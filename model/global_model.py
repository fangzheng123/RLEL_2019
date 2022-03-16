# encoding: utf-8

import time
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim

from model_config import GlobalConfig
from model_helper import ModelHelper
from data_util import DataUtil

class GlobalModel(object):

    def __init__(self, global_config):
        self.global_config = global_config

        # mention context embedding, shape=(B,S,H)
        self.input_mention_context = tf.placeholder(tf.float32, [None, None, self.global_config.local_rep_dim],
                                                    name="input_mention_context")
        # entity desc embedding
        self.input_entity_desc = tf.placeholder(tf.float32, [None, None, self.global_config.local_rep_dim],
                                                     name="input_entity_desc")

        # entity embedding
        self.input_entity_embedding = tf.placeholder(tf.float32, [None, None, self.global_config.word_embedd_dim],
                                               name="input_entity_embedding")

        # entity fea
        self.input_entity_fea = tf.placeholder(tf.float32, [None, None, self.global_config.fea_dim],
                                                            name="input_entity_fea")
        # entity score
        self.input_entity_score = tf.placeholder(tf.float32, [None, self.global_config.sequence_len], name="input_entity_score")

        # 输出类别
        self.input_label = tf.placeholder(tf.int32, [None, None, self.global_config.class_num], name='input_label')
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        self.rnn_model()



    def rnn_model(self):
        """
        构造全局表示网络
        :return:
        """
        # add dropout
        def dropout():
            cell = tf.contrib.rnn.BasicLSTMCell(self.global_config.rnn_hidden_size, state_is_tuple=True)
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # rnn
        def rnn_output(input):
            cells = [dropout() for _ in range(self.global_config.rnn_num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            _output, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=input, dtype=tf.float32)
            return _output

        new_input = tf.concat([self.input_mention_context, self.input_entity_desc, self.input_entity_embedding], axis=-1)

        _output = rnn_output(new_input)
        self.rnn_output = _output

        mlp_output = _output
        # for l_size in ([128, 64, 32][:self.global_config.mlp_metric_layer]):
        #     mlp_output = slim.fully_connected(mlp_output, l_size)
        # mlp_output = tf.concat([mlp_output, self.input_entity_fea], axis=-1)

        self.output_logits = slim.fully_connected(mlp_output, self.global_config.class_num, activation_fn=None)
        # predict
        self.y_pred_cls = tf.argmax(tf.nn.softmax(self.output_logits), -1)
        # cross_entropy
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.output_logits, labels=self.input_label)
        self.loss = tf.reduce_mean(cross_entropy)
        # Optimizer
        self.optim = tf.train.AdamOptimizer(learning_rate=self.global_config.global_learning_rate).minimize(self.loss)

        # acc
        correct_pred = tf.equal(tf.argmax(self.input_label, -1), self.y_pred_cls)
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


class GlobalModelProcess(object):
    """
    全局模型处理类
    """

    def __init__(self, global_config, model_helper, data_util, global_model):
        """

        :param global_config:
        :param data_helper:
        :param global_model:
        """
        self.global_config = global_config
        self.model_helper = model_helper
        self.data_util = data_util
        self.global_model = global_model

    def load_global_data(self, data_path, data_rep_path, data_name="Train or Test"):
        """
        load data for global model
        :return:
        """
        print("Loading {0} data...".format(data_name))
        start_time = time.time()

        # load data
        mention_context_embed, entity_desc_embed, entity_embed, feas, xgb_scores, labels = self.model_helper.load_global_data(data_path, data_rep_path)

        time_dif = self.data_util.get_time_dif(start_time)
        print("Time usage:{0}".format(time_dif))
        print("mention:{0}, entity_desc:{1}, entity_embed:{2}, fea:{3}, xgb_scores: {4}, label:{5}"
              .format(len(mention_context_embed), len(entity_desc_embed), len(entity_embed), len(feas), len(xgb_scores), len(labels)))

        data_batch = [mention_context_embed, entity_desc_embed, entity_embed, feas, xgb_scores, labels]

        return data_batch

    def evaluate(self, sess, eva_data):
        """
        评估在某一数据上的准确率和损失
        :param sess:
        :param eva_data:
        :return:
        """
        mention_eval, entity_desc_eval, embedd_eval, fea_eval, xgb_scores, label_eval = eva_data

        data_len = len(mention_eval) / self.global_config.sequence_len
        batch_eval = self.model_helper.global_batch_iter(eva_data)
        total_loss = 0.0
        total_acc = 0.0
        for mention_batch, entity_desc_batch, embedd_batch, fea_batch, score_batch, label_batch in batch_eval:
            batch_len = len(mention_batch)
            feed_dict = {
                self.global_model.input_mention_context: mention_batch,
                self.global_model.input_entity_desc: entity_desc_batch,
                self.global_model.input_entity_embedding: embedd_batch,
                self.global_model.input_entity_fea: fea_batch,
                self.global_model.input_entity_score: score_batch,
                self.global_model.input_label: label_batch,
                self.global_model.keep_prob: 1.0
            }
            loss, acc = sess.run([self.global_model.loss, self.global_model.acc], feed_dict=feed_dict)
            total_loss += loss * batch_len
            total_acc += acc * batch_len

        return total_loss / data_len, total_acc / data_len

    def train_global_model(self):
        """
        train global model
        :return:
        """
        # 配置全局模型保存路径
        saver = tf.train.Saver()
        if not os.path.exists(self.global_config.save_global_dir):
            os.makedirs(self.global_config.save_global_dir)

        # 加载训练数据
        train_data_list = self.load_global_data(
            self.global_config.train_local_mention_rank_path,
            self.global_config.local_rep_train_path,
            data_name="Train"
        )

        # 加载测试数据
        test_data_list = self.load_global_data(
            self.global_config.test_local_mention_rank_path,
            self.global_config.local_rep_test_path,
            data_name="Test"
        )

        # 创建session
        session = tf.Session()
        session.run(tf.global_variables_initializer())

        print('Training Global Model...')
        start_time = time.time()
        # 总批次
        total_batch = 0
        # 最佳验证集准确率
        best_acc_val = 0.0
        # 记录上一次提升批次
        last_improved = 0

        # early stopping
        early_stop_flag = False
        for epoch in range(self.global_config.global_num_epochs):
            print('Epoch:', epoch + 1)
            batch_train = self.model_helper.global_batch_iter(train_data_list)
            for mention_batch, entity_desc_batch, embedd_batch, fea_batch, score_batch, label_batch in batch_train:
                feed_dict = {
                    self.global_model.input_mention_context: mention_batch,
                    self.global_model.input_entity_desc: entity_desc_batch,
                    self.global_model.input_entity_embedding: embedd_batch,
                    self.global_model.input_entity_fea: fea_batch,
                    self.global_model.input_entity_score: score_batch,
                    self.global_model.input_label: label_batch,
                    self.global_model.keep_prob: self.global_config.dropout_keep_prob
                }

                if total_batch % self.global_config.print_per_batch == 0:
                    feed_dict[self.global_model.keep_prob] = 1.0
                    loss_train, acc_train = session.run([self.global_model.loss, self.global_model.acc], feed_dict=feed_dict)
                    time_dif = self.data_util.get_time_dif(start_time)
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                          + ' Time: {3}'
                    print(msg.format(total_batch, loss_train, acc_train, time_dif))

                    if total_batch % 25 == 0:
                        loss_val, acc_val = self.evaluate(session, test_data_list)

                        # 保存最好结果
                        if acc_val > best_acc_val:
                            best_acc_val = acc_val
                            last_improved = total_batch
                            saver.save(sess=session, save_path=self.global_config.save_global_dir)
                            improved_str = '*'
                        else:
                            improved_str = ''

                        time_dif = self.data_util.get_time_dif(start_time)
                        msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                              + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                        print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

                # 对loss进行优化
                session.run(self.global_model.optim, feed_dict=feed_dict)
                total_batch += 1

                # 验证集正确率长期不提升，提前结束训练
                if total_batch - last_improved > self.global_config.local_require_improvement:
                    print("No optimization for a long time, auto-stopping...")
                    early_stop_flag = True
                    break

            # early stopping
            if early_stop_flag:
                break

        session.close()

    def controller(self):
        """

        :return:
        """
        # 训练全局模型
        self.train_global_model()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    data_util = DataUtil()
    model_helper = ModelHelper(data_util)

    global_config = GlobalConfig()
    global_model = GlobalModel(global_config)

    global_model_process = GlobalModelProcess(global_config, model_helper, data_util, global_model)
    global_model_process.controller()