# coding: utf-8

import os
import time
import json

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from model_config import LocalConfig
from model_helper import ModelHelper
from data_util import DataUtil

class LocalModel(object):
    def __init__(self, config, model_helper):
        self.config = config
        self.model_helper = model_helper

        # 输入的数据
        self.input_mention = tf.placeholder(tf.float32, [None, self.config.candidate_num, self.config.seq_length,
                                                       self.config.word_embedd_dim], name="input_x")

        self.entity_desc = tf.placeholder(tf.float32, [None, self.config.candidate_num, self.config.seq_length,
                                                     self.config.word_embedd_dim],name="entity_desc")

        self.entity_embed = tf.placeholder(tf.float32, [None, self.config.candidate_num,
                                                     self.config.word_embedd_dim],name="entity_embed")

        self.entity_fea = tf.placeholder(tf.float32, [None, self.config.candidate_num,
                                                          self.config.fea_dim], name="entity_fea")

        self.xgb_score = tf.placeholder(tf.float32, [None, self.config.candidate_num], name="xgb_score")

        self.input_label = tf.placeholder(tf.int64, [None, self.config.candidate_num], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        self.rnn_model()

    def rnn_model(self):
        """
        构造本地表示网络
        :return:
        """

        # add dropout
        def dropout():
            cell = tf.contrib.rnn.BasicLSTMCell(self.config.rnn_hidden_size, state_is_tuple=True)
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # rnn
        def rnn_output(input):
            cells = [dropout() for _ in range(self.config.rnn_num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            _output, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=input, dtype=tf.float32)
            return _output

        # two LSTM network shared variables
        with tf.variable_scope("sent_encode"):
            # encode mention context
            input_mention = tf.reshape(self.input_mention, [-1, self.config.seq_length, self.config.word_embedd_dim])
            _context_output = rnn_output(input_mention)
            context_last = _context_output[:, -1, :]
            context_last = tf.reshape(context_last, [-1, self.config.candidate_num, self.config.rnn_hidden_size])

        with tf.variable_scope("sent_encode", reuse=True):
            # encode entity desc
            input_desc = tf.reshape(self.entity_desc, [-1, self.config.seq_length, self.config.word_embedd_dim])
            _desc_output = rnn_output(input_desc)
            desc_last = _desc_output[:, -1, :]
            desc_last = tf.reshape(desc_last, [-1, self.config.candidate_num, self.config.rnn_hidden_size])

        # shape=(B, C, H)
        self.mention_context_rep = context_last
        self.candidate_desc_rep = desc_last

        semantic_embed = tf.concat([context_last, desc_last, self.entity_embed], axis=-1)

        for l_size in ([256, 128, 64][:3]):
            semantic_embed = slim.fully_connected(semantic_embed, l_size, activation_fn=tf.nn.softplus)
            semantic_embed = tf.nn.dropout(semantic_embed, keep_prob=self.keep_prob)

        score_in = tf.tile(tf.expand_dims(self.xgb_score, -1), [1, 1, 20])
        mlp_output = tf.concat([semantic_embed, self.entity_fea, score_in], axis=-1)

        # (B,C,2)
        for l_size in ([32, 8][:self.config.mlp_metric_layer] + [2]):
            mlp_output = slim.fully_connected(mlp_output, l_size, activation_fn=tf.nn.softplus)
            mlp_output = tf.nn.dropout(mlp_output, keep_prob=self.keep_prob)

        batch_loss_list = []
        batch_prob_list = []
        for batch_index in range(self.config.local_batch_size):
            logits = mlp_output[batch_index, :, :]
            # [C, 2]
            prob = tf.nn.softmax(logits, axis=-1)
            # [C]
            label = self.input_label[batch_index, :]
            loss = self.rank_loss(prob, label)
            batch_loss_list.append(loss)
            batch_prob_list.append(prob)

        self.batch_prob = tf.stack(batch_prob_list)
        self.group_acc = self.group_accuracy()
        self.total_loss = tf.reduce_sum(batch_loss_list)
        self.optim = tf.train.AdamOptimizer(learning_rate=self.config.local_learning_rate).minimize(self.total_loss)

    def rank_loss(self, prob, label):
        # label_in:[C]
        # [P]
        pos_indices = tf.where(tf.equal(label, 1))
        # [N]
        neg_indices = tf.where(tf.equal(label, 0))

        # [P,2]
        pos_metric = tf.gather_nd(prob, pos_indices)
        # [N,2]
        neg_metric = tf.gather_nd(prob, neg_indices)

        pos_one_hot = tf.constant([[0, 1]], dtype=tf.float32)
        # [P,2]
        pos_one_hot_labels = tf.tile(pos_one_hot, [tf.shape(pos_indices)[0], 1])
        # [N,2]
        neg_one_hot_labels = tf.tile(pos_one_hot, [tf.shape(neg_indices)[0], 1])

        # only calculate the probability of label 1
        # [P]
        pos_metric = tf.reduce_sum(pos_metric * pos_one_hot_labels, axis=-1)
        # [N]
        neg_metric = tf.reduce_sum(neg_metric * neg_one_hot_labels, axis=-1)

        # do the substraction
        # [P, N]
        pos_metric = tf.tile(tf.expand_dims(pos_metric, 1), [1, tf.shape(neg_indices)[0]])
        # [P, N]
        neg_metric = tf.tile(tf.expand_dims(neg_metric, 0), [tf.shape(pos_indices)[0], 1])
        # [P, N]
        delta = neg_metric - pos_metric

        loss = tf.reduce_mean(tf.nn.relu(self.config.local_margin + delta))

        return loss

    def group_accuracy(self):
        """
        calculate group acc for each mention
        :return:
        """
        # shape=(B,C)
        self.mention_logits = self.batch_prob[:, :, -1]
        # shape=(B)
        logits_max = tf.argmax(self.mention_logits, -1)
        label_max = tf.argmax(self.input_label, -1)
        self.correct_prediction = tf.equal(logits_max, label_max)
        accuracy_all = tf.cast(self.correct_prediction, tf.float32)
        return tf.reduce_mean(accuracy_all)

class LocalModelProcess(object):
    """
    process local model
    """

    def __init__(self, local_config, model_helper, data_util, local_model):
        self.local_config = local_config
        self.model_helper = model_helper
        self.local_model = local_model
        self.data_util = data_util

    def load_local_data(self, data_path, data_name="Train or Test", is_filter=True):
        """
        load data for local model
        :return:
        """
        print("Loading {0} data...".format(data_name))
        start_time = time.time()

        # load data
        mention_context_embed, entity_desc_embed, entity_embed, feas, xgb_scores, labels = self.model_helper.load_local_data(data_path, is_load_tmp=True, is_filter=is_filter)

        time_dif = self.data_util.get_time_dif(start_time)
        print("Time usage:{0}".format(time_dif))

        data_batch = [mention_context_embed, entity_desc_embed, entity_embed, feas, xgb_scores, labels]

        return data_batch

    def evaluate(self, sess, eval_data, is_cal_loss=True, is_test=False):
        """
        evaluate model
        :param sess:
        :param eval_data:
        :param is_cal_loss: when test data, one group may not contain label 0 and 1, so the rank loss can't calculate
        :return:
        """
        val_mention, val_candidate_desc, val_candidate_embed, \
        val_candidate_fea, val_score, val_label = eval_data

        data_len = val_mention.shape[0]
        batch_eval = self.model_helper.local_batch_iter(eval_data)
        total_loss = 0.0
        total_acc = 0.0

        all_logist_list = []
        all_pred_list = []
        for mention_batch, candidate_desc_batch, candidate_embed_batch, \
            fea_batch, score_batch, label_batch in batch_eval:
            batch_len = mention_batch.shape[0]
            feed_dict = {
                self.local_model.input_mention: mention_batch,
                self.local_model.entity_desc: candidate_desc_batch,
                self.local_model.entity_embed: candidate_embed_batch,
                self.local_model.entity_fea: fea_batch,
                self.local_model.xgb_score: score_batch,
                self.local_model.input_label: label_batch,
                self.local_model.keep_prob: 1.0
            }
            if is_cal_loss:
                loss, acc, logits, pred = sess.run([self.local_model.total_loss,
                                                        self.local_model.group_acc, self.local_model.mention_logits,
                                                        self.local_model.correct_prediction], feed_dict=feed_dict)
                total_loss += loss * batch_len
                all_logist_list.extend(logits.tolist())
                all_pred_list.extend(pred.tolist())

            else:
                acc = sess.run(self.local_model.group_acc, feed_dict=feed_dict)

            total_acc += acc * batch_len

        if is_test:
            self.save_model_pred(all_logist_list)
            self.error_analyse(all_pred_list)

        return total_loss / data_len, total_acc / data_len

    def save_model_pred(self, all_logist_list):
        """
        save the lstm result
        :param all_logist_list:
        :return:
        """
        data_group_path = self.local_config.test_path + "_lstm_group"
        group_list = json.load(open(data_group_path, "r"))

        group_pred_score = {}
        for index, group_id in enumerate(group_list):
            group_pred_score[group_id] = all_logist_list[index]

        group_item_dict = {}
        with open(self.local_config.test_path, "r", encoding="utf-8") as test_file:
            for item in test_file:
                item = item.strip()
                group_id = int(item.strip().split("\t")[0])

                if group_id not in group_item_dict:
                    group_item_dict[group_id] = [item]
                else:
                    group_item_dict[group_id].append(item)

        with open(self.local_config.test_local_pred_path, "w", encoding="utf-8") as lstm_pred_file:
            for group_id, item_list in group_item_dict.items():
                for index, item in enumerate(item_list):
                    item = item.strip()
                    if group_id in group_pred_score:
                        group_str, label_str, fea_str, mention_str, entity_str = item.split("\t")
                        entity_json = json.loads(entity_str)
                        entity_json["local_pred"] = group_pred_score[group_id][index]

                        lstm_pred_file.write("\t".join([group_str, label_str, fea_str,
                                                        mention_str, json.dumps(entity_json)]) + "\n")

                        lstm_pred_file.flush()
                    else:
                        lstm_pred_file.write(item + "\n")
                        lstm_pred_file.flush()

    def error_analyse(self, all_pred_list):
        data_group_path = self.local_config.test_path + "_lstm_group"
        group_list = json.load(open(data_group_path, "r"))

        for index, pred in enumerate(all_pred_list[:len(group_list)]):
            if not pred:
                print("error group: {0}".format(group_list[index]))

    def train_local_model(self):
        """
        train local lstm model
        :return:
        """
        # 配置本地模型保存路径
        saver = tf.train.Saver()
        if not os.path.exists(self.local_config.save_local_dir):
            os.makedirs(self.local_config.save_local_dir)

        # 加载训练数据
        train_data_batch = self.load_local_data(self.local_config.train_path, "Train")

        #加载验证数据
        val_data_batch = self.load_local_data(self.local_config.val_path, "Test")

        # 创建session
        session = tf.Session()
        session.run(tf.global_variables_initializer())

        print('Training Local Model...')
        start_time = time.time()
        # 总批次
        total_batch = 0
        # 最佳验证集准确率
        best_acc_val = 0.0
        # 记录上一次提升批次
        last_improved = 0

        # early stopping的标志位
        early_stop_flag = False
        for epoch in range(self.local_config.local_num_epochs):
            print('Epoch:', epoch + 1)
            batch_train = self.model_helper.local_batch_iter(train_data_batch)
            for mention_batch, entity_desc_batch, entity_embed_batch, fea_batch, score_batch, label_batch in batch_train:
                feed_dict = {
                    self.local_model.input_mention: mention_batch,
                    self.local_model.entity_desc: entity_desc_batch,
                    self.local_model.entity_embed: entity_embed_batch,
                    self.local_model.entity_fea: fea_batch,
                    self.local_model.xgb_score: score_batch,
                    self.local_model.input_label: label_batch,
                    self.local_model.keep_prob: self.local_config.dropout_keep_prob
                }

                # the loss in train data
                if total_batch % self.local_config.print_per_batch == 0:
                    loss_train, acc_train = session.run([self.local_model.total_loss, self.local_model.group_acc], feed_dict=feed_dict)
                    time_dif = self.data_util.get_time_dif(start_time)
                    msg = "Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%}," + " Time: {3}"
                    print(msg.format(total_batch, loss_train, acc_train, time_dif))

                    if total_batch % 50 == 0:
                        loss_val, acc_val = self.evaluate(session, val_data_batch)
                        # 保存最好结果
                        if acc_val > best_acc_val:
                            best_acc_val = acc_val
                            last_improved = total_batch
                            saver.save(sess=session, save_path=self.local_config.save_local_dir)
                            improved_str = '*'
                        else:
                            improved_str = ''

                        time_dif = self.data_util.get_time_dif(start_time)
                        msg = "Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%}," \
                              + " Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}"
                        print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

                # 对loss进行优化
                session.run(self.local_model.optim, feed_dict=feed_dict)
                total_batch += 1

                # 正确率长期不提升，提前结束训练
                if total_batch - last_improved > self.local_config.local_require_improvement:
                    print("No optimization for a long time, auto-stopping...")
                    early_stop_flag = True
                    break

            # early stopping
            if early_stop_flag:
                break

        session.close()

    def test_local_model(self):
        """
        test local model
        :return:
        """
        print("Test Local Model...")
        # create session
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # load local model
        saver.restore(sess=session, save_path=self.local_config.save_local_dir)

        test_data_batch = self.load_local_data(self.local_config.test_path, "Test")

        loss_test, acc_test = self.evaluate(session, test_data_batch, is_test=True)
        print("loss_test:{0}, acc_test:{1}".format(loss_test, acc_test))

        session.close()

    def save_local_representation(self, data_list, save_path):
        """
        save local encoding of mention context and entity desc
        :param data_list: cut_candidate list, numpy
        :param save_path:
        :return:
        """
        print('Save Local Representation...')
        start_time = time.time()

        # create session
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # load local model
        saver.restore(sess=session, save_path=self.local_config.save_local_dir)

        local_represent_list = []
        batch_eval = self.model_helper.local_batch_iter(data_list)

        for mention_batch, entity_desc_batch, entity_embed_batch, fea_batch, score_batch, label_batch in batch_eval:
            feed_dict = {
                self.local_model.input_mention: mention_batch,
                self.local_model.entity_desc: entity_desc_batch,
                self.local_model.entity_embed: entity_embed_batch,
                self.local_model.entity_fea: fea_batch,
                self.local_model.xgb_score: score_batch,
                self.local_model.input_label: label_batch,
                self.local_model.keep_prob: 1.0
            }
            # mention_rep_shape=(B,C,H), entity_rep_shape=(B,C,H）
            mention_rep_batch, entity_rep_batch = session.run(
                [self.local_model.mention_context_rep, self.local_model.candidate_desc_rep], feed_dict=feed_dict)

            for mention_index in range(len(mention_batch)):
                for entity_index in range(self.local_config.candidate_num):
                    # shape=(H)
                    mention_rep = mention_rep_batch[mention_index, entity_index, :]
                    # shape=(H)
                    entity_rep = entity_rep_batch[mention_index, entity_index, :]

                    mention_entity = np.vstack((mention_rep, entity_rep))

                    local_represent_list.append(mention_entity)

        # save local rep, shape=(batch*candidate_num, local_representation_size*2)
        print("local_represent_list:{0}".format(len(local_represent_list)))
        local_represent_np = np.array(local_represent_list)
        np.save(save_path, local_represent_np)

        time_dif = self.data_util.get_time_dif(start_time)
        print("Save Representation Time usage:{0}".format(time_dif))

        session.close()

    def controller(self):
        """
        main process
        :return:
        """
        # 训练本地模型
        local_model_process.train_local_model()

        # 测试本地模型
        local_model_process.test_local_model()

        # 保存测试文件中间表示
        test_data_list = self.load_local_data(self.local_config.test_local_mention_rank_path, "Test", is_filter=False)
        local_model_process.save_local_representation(test_data_list, self.local_config.local_rep_test_path)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    data_util = DataUtil()
    model_helper = ModelHelper(data_util)
    local_config = LocalConfig()

    local_model = LocalModel(local_config, model_helper)
    local_model_process = LocalModelProcess(local_config, model_helper, data_util, local_model)
    local_model_process.controller()

