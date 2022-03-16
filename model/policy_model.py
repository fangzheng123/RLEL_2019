# encoding: utf-8

import time
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from data_util import DataUtil
from global_model import GlobalModel
from model_config import GlobalConfig, PolicyConfig
from model_helper import ModelHelper

class Agent():
    """
    强化学习中agent类，与环境进行交互，此处即为一个神经网络
    """

    def __init__(self, policy_config):
        self.policy_config = policy_config

        # mention上下文表示, shape=(B, S, C, H)
        self.input_mention_context = tf.placeholder(tf.float32, [None, None, self.policy_config.candidate_num, self.policy_config.local_rep_dim],
                                                    name="input_mention_context")
        # 候选实体本地表示
        self.input_entity_desc = tf.placeholder(tf.float32, [None, None, self.policy_config.candidate_num, self.policy_config.local_rep_dim],
                                                     name="input_entity_desc")
        # 候选实体向量
        self.input_entity_embedding = tf.placeholder(tf.float32, [None, None, self.policy_config.candidate_num, self.policy_config.word_embedd_dim],
                                               name="input_entity_embedding")

        # 候选实体相关特征
        self.input_entity_fea = tf.placeholder(tf.float32, [None, None, self.policy_config.candidate_num, self.policy_config.fea_dim],
                                                            name="input_entity_fea")

        # xgb score
        self.xgb_score = tf.placeholder(tf.float32, [None, None, self.policy_config.candidate_num], name="xgb_score")

        # 上一次决策后的状态
        self.last_state = tf.placeholder(tf.float32, [None, None, self.policy_config.global_rep_dim], name="last_state")

        # action和reward
        self.action_holder = tf.placeholder(shape=[None, self.policy_config.sequence_len], dtype=tf.int32)
        self.reward_holder = tf.placeholder(shape=[None, self.policy_config.sequence_len], dtype=tf.float32)

        self.policy_model()

    def policy_model(self):
        """
        构造policy网络，选择action
        :return:
        """
        # shape=(B, S, C, H)
        mention_entity = tf.concat([self.input_mention_context, self.input_entity_desc, self.input_entity_embedding], axis=-1)

        mlp_output = mention_entity
        for l_size in ([256, 128][:self.policy_config.policy_mlp_layer]):
            mlp_output = slim.fully_connected(mlp_output, l_size, activation_fn=tf.nn.softplus)

        mlp_output = tf.concat([tf.tile(tf.expand_dims(self.last_state, 2), [1, 1, self.policy_config.candidate_num, 1]), mlp_output], axis=-1)

        for l_size in ([128, 64][:self.policy_config.policy_mlp_layer]):
            mlp_output = slim.fully_connected(mlp_output, l_size, activation_fn=tf.nn.softplus)

        score_in = tf.tile(tf.expand_dims(self.xgb_score, -1), [1, 1, 1, 20])
        # mlp_output = tf.concat([score_in, mlp_output], axis=-1)
        mlp_output = tf.concat([self.input_entity_fea, score_in, mlp_output], axis=-1)

        for l_size in ([32, 16, 8][:self.policy_config.policy_mlp_layer] + [1]):
            mlp_output = slim.fully_connected(mlp_output, l_size, activation_fn=tf.nn.softplus)

        # shape=(B,S,C)
        self.action_output = tf.nn.softmax((tf.squeeze(mlp_output, squeeze_dims=[-1])))

        # 根据action选择当前候选实体
        self.chosen_action = tf.argmax(self.action_output, -1)

        # 获取文档中每次选择action的下标
        self.indexes = tf.range(0, tf.shape(self.action_output)[0] * tf.shape(self.action_output)[1]) \
                       * tf.shape(self.action_output)[2] + tf.reshape(self.action_holder, [-1])
        # 根据indexs获取output中相应的值
        self.responsible_outputs = tf.gather(tf.reshape(self.action_output, [-1]), self.indexes)
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * tf.reshape(self.reward_holder, [-1]))

        # 优化器
        self.optim = tf.train.AdamOptimizer(learning_rate=self.policy_config.policy_learning_rate).minimize(self.loss)


class AgentProcess(object):
    """
    agent处理类
    """

    def __init__(self, global_config, policy_config, model_helper, data_util):
        self.global_config = global_config
        self.policy_config = policy_config
        self.model_helper = model_helper
        self.data_util = data_util

    def load_policy_data(self, data_path, data_rep_path, data_name="Train or Test", is_train=False):
        """
        load data for policy model
        :param data_path:
        :param data_rep_path:
        :param data_name:
        :return:
        """
        print("Loading {0} data...".format(data_name))
        start_time = time.time()

        # load data
        mention_context_embed, entity_desc_embed, entity_embed, feas, xgb_scores, labels = self.model_helper.load_policy_data(data_path, data_rep_path, is_train=is_train)

        time_dif = self.data_util.get_time_dif(start_time)
        print("Time usage:{0}".format(time_dif))
        print("mention_seq:{0}, entity_desc_seq:{1}, entity_embed_seq:{2}, fea_seq:{3}, xgb_scores_seq: {4}, label_seq:{5}"
              .format(len(mention_context_embed), len(entity_desc_embed), len(entity_embed), len(feas), len(xgb_scores), len(labels)))

        data_batch = [mention_context_embed, entity_desc_embed, entity_embed, feas, xgb_scores, labels]

        return data_batch

    def discount_rewards(self, reward_np):
        """
        计算action_value
        :param reward_np: shape=(B,S)
        :return:
        """
        dicount_reward_list = []
        for reward_seq in reward_np:
            discounted_reward = np.zeros_like(reward_seq)
            penalty = 0
            add_reward = 0
            for time_step in reversed(range(0, len(reward_seq))):
                if reward_seq[time_step] == 0:
                    penalty += -1
                    discounted_reward[time_step] = penalty
                elif reward_seq[time_step] == 1:
                    add_reward += 1
                    discounted_reward[time_step] = add_reward
            dicount_reward_list.append(discounted_reward)

        return np.array(dicount_reward_list)

    def evaluate(self, policy_sess, global_sess, my_agent, global_model, data_batch, is_test=False):
        """
        评估在某一数据上的准确率和损失
        :param policy_sess:
        :param global_sess:
        :param my_agent:
        :param global_model:
        :param data_batch:
        :param is_test:
        :return:
        """
        mention_context_embed, entity_desc_embed, entity_embed, feas, xgb_scores, labels = data_batch
        data_len = len(mention_context_embed)
        total_loss = 0.0
        total_acc = 0.0

        batch_val = self.model_helper.policy_batch_iter(data_batch)

        all_reward_list = []
        # (mention_batch, entity_batch,fea_batch,label_batch):shape=(B,S,C,H)
        for mention_batch, entity_batch, embedd_batch, fea_batch, score_batch, label_batch in batch_val:
            # global模型中上一时刻的输出
            global_last_state = np.zeros([len(mention_batch), 1, self.policy_config.global_rep_dim])

            seq_mention_list = []
            seq_select_entity_list = []
            seq_select_embedd_list = []
            seq_select_fea_list = []
            seq_select_score_list = []
            seq_last_state_list = []
            policy_action_list = []
            policy_reward_list = []

            for mention_index in range(self.policy_config.sequence_len):
                current_mention_batch = mention_batch[:, mention_index, :, :]
                current_mention_batch = np.expand_dims(current_mention_batch, axis=1)
                cuurent_entity_batch = entity_batch[:, mention_index, :, :]
                cuurent_entity_batch = np.expand_dims(cuurent_entity_batch, axis=1)
                current_entity_embedd_batch = embedd_batch[:, mention_index, :, :]
                current_entity_embedd_batch = np.expand_dims(current_entity_embedd_batch, axis=1)
                current_entity_fea_batch = fea_batch[:, mention_index, :, :]
                current_entity_fea_batch = np.expand_dims(current_entity_fea_batch, axis=1)
                current_entity_score_batch = score_batch[:, mention_index, :]
                current_entity_score_batch = np.expand_dims(current_entity_score_batch, axis=1)

                policy_feed_dict = {
                    my_agent.input_mention_context: current_mention_batch,
                    my_agent.input_entity_desc: cuurent_entity_batch,
                    my_agent.input_entity_embedding: current_entity_embedd_batch,
                    my_agent.input_entity_fea: current_entity_fea_batch,
                    my_agent.xgb_score:current_entity_score_batch,
                    my_agent.last_state: global_last_state
                }

                # shape=(S,B,H)
                seq_last_state_list.append(np.squeeze(global_last_state, axis=1))

                # 选择概率最大的action
                chosen_actions = policy_sess.run(my_agent.chosen_action, feed_dict=policy_feed_dict)
                chosen_actions = np.squeeze(chosen_actions, axis=1)
                # print chosen_actions

                current_action_batch = [action for action in chosen_actions]

                # 获取即时reward
                current_label = label_batch[:, mention_index, :, :]
                current_label_batch = [np.argmax(current_label[batch_index, action_index, :], axis=-1)
                                       for batch_index, action_index in enumerate(current_action_batch)]
                current_reward_batch = []
                for label in current_label_batch:
                    # 正例
                    if label == 1:
                        current_reward_batch.append(1)
                    # 负例
                    elif label == 0:
                        current_reward_batch.append(0)

                # shape=(S,B)
                policy_action_list.append(current_action_batch)
                policy_reward_list.append(current_reward_batch)

                # 构造global模型的输入数据
                seq_mention_list.append(mention_batch[:, mention_index, 0, :])
                global_mention_input = np.stack(seq_mention_list, axis=1)

                candidate_entity_batch = entity_batch[:, mention_index, :, :]
                entity_fea_batch = fea_batch[:, mention_index, :, :]
                entity_embedd_batch = embedd_batch[:, mention_index, :, :]
                select_entity_batch = [candidate_entity_batch[batch_index, select_index, :]
                                       for batch_index, select_index in enumerate(current_action_batch)]
                select_entity_fea_batch = [entity_fea_batch[batch_index, select_index, :]
                                           for batch_index, select_index in enumerate(current_action_batch)]
                select_entity_embedd_batch = [entity_embedd_batch[batch_index, select_index, :]
                                           for batch_index, select_index in enumerate(current_action_batch)]

                seq_select_entity_list.append(select_entity_batch)
                seq_select_fea_list.append(select_entity_fea_batch)
                seq_select_embedd_list.append(select_entity_embedd_batch)

                global_entity_input = np.stack(seq_select_entity_list, axis=1)
                global_fea_input = np.stack(seq_select_fea_list, axis=1)
                global_embedd_input = np.stack(seq_select_embedd_list, axis=1)

                global_feed_dict = {
                    global_model.input_mention_context: global_mention_input,
                    global_model.input_entity_desc: global_entity_input,
                    global_model.input_entity_embedding: global_embedd_input,
                    global_model.input_entity_fea: global_fea_input,
                    global_model.keep_prob: 1.0
                }

                # 获取global模型的输出
                global_cell_output = global_sess.run(global_model.rnn_output, feed_dict=global_feed_dict)
                global_last_state = global_cell_output[:, -1, :]
                global_last_state = np.expand_dims(global_last_state, axis=1)

            # 周期结束
            # shape=(B,S)
            policy_action_np = np.stack(policy_action_list, axis=1)
            # 计算action_value
            policy_reward_np = np.stack(policy_reward_list, axis=1)
            policy_action_value = self.discount_rewards(policy_reward_np)

            all_reward_list.extend(policy_reward_np.tolist())

            seq_feed_dict = {
                my_agent.action_holder: policy_action_np,
                my_agent.reward_holder: policy_action_value,
                my_agent.input_mention_context: mention_batch,
                my_agent.input_entity_desc: entity_batch,
                my_agent.input_entity_embedding: embedd_batch,
                my_agent.input_entity_fea: fea_batch,
                my_agent.xgb_score: score_batch,
                my_agent.last_state: np.stack(seq_last_state_list, axis=1)
            }

            batch_len = len(mention_batch)
            loss = policy_sess.run(my_agent.loss, feed_dict=seq_feed_dict)
            acc = float(sum([sum(seq_reward) for seq_reward in policy_reward_np])) / policy_reward_np.size
            total_loss += loss * batch_len
            total_acc += acc * batch_len

        # 保存预测结果
        if is_test:
            print("mention num:{0}".format(len(all_reward_list) * self.policy_config.sequence_len))
            np.save(self.policy_config.test_policy_reward_path, np.array(all_reward_list))

        return total_loss / data_len, total_acc / data_len


    def train_agent(self):
        """
        训练agent
        :return:
        """
        # create global graph
        global_graph = tf.Graph()
        global_session = tf.Session(graph=global_graph)
        # 加载全局模型
        with global_session.as_default():
            with global_graph.as_default():
                global_model = GlobalModel(self.global_config)
                global_session.run(tf.global_variables_initializer())
                global_saver = tf.train.Saver()
                global_saver.restore(sess=global_session, save_path=self.global_config.save_global_dir)

        # create policy graph
        policy_graph = tf.Graph()
        policy_session = tf.Session(graph=policy_graph)
        # train policy model
        with policy_session.as_default():
            with policy_graph.as_default():
                # init policy network
                my_agent = Agent(self.policy_config)
                policy_saver = tf.train.Saver()

                policy_session.run(tf.global_variables_initializer())

                # load train data
                train_data_list = self.load_policy_data(
                    self.policy_config.train_local_mention_rank_path,
                    self.policy_config.local_rep_train_path,
                    data_name="Train", is_train=True)

                # load test data
                test_data_list = self.load_policy_data(
                    self.policy_config.test_local_mention_rank_path,
                    self.policy_config.local_rep_test_path,
                    data_name="Test")

                print('Training Policy Model...')
                start_time = time.time()
                # 总批次
                total_batch = 0
                # 最佳验证集准确率
                best_acc_val = 0.0
                # 记录上一次提升批次
                last_improved = 0

                # early stopping的标志位
                early_stop_flag = False
                for epoch in range(self.policy_config.policy_num_epochs):
                    print('Epoch:', epoch + 1)
                    batch_train = self.model_helper.policy_batch_iter(train_data_list, is_random=True)

                    # (mention_batch,entity_batch,fea_batch,label_batch):shape=(B,S,C,H), (score_batch):shape=(B,S,C)
                    for mention_batch, entity_batch, entity_embedd_batch, fea_batch, score_batch, label_batch in batch_train:

                        # global模型中上一时刻的输出
                        global_last_state = np.zeros([len(mention_batch), 1, self.policy_config.global_rep_dim])
                        # 采样得到的序列结果
                        seq_mention_list = []
                        seq_select_entity_list = []
                        seq_select_embedd_list = []
                        seq_select_fea_list = []
                        seq_select_score_list = []
                        seq_last_state_list = []
                        policy_action_list = []
                        policy_reward_list = []

                        # 采样一个周期
                        for mention_index in range(self.policy_config.sequence_len):
                            current_mention_batch = mention_batch[:, mention_index, :, :]
                            current_mention_batch = np.expand_dims(current_mention_batch, axis=1)
                            cuurent_entity_batch = entity_batch[:, mention_index, :, :]
                            cuurent_entity_batch = np.expand_dims(cuurent_entity_batch, axis=1)
                            current_entity_embedd_batch = entity_embedd_batch[:, mention_index, :, :]
                            current_entity_embedd_batch = np.expand_dims(current_entity_embedd_batch, axis=1)
                            current_entity_fea_batch = fea_batch[:, mention_index, :, :]
                            current_entity_fea_batch = np.expand_dims(current_entity_fea_batch, axis=1)
                            current_score_batch = score_batch[:, mention_index, :]
                            current_score_batch = np.expand_dims(current_score_batch, axis=1)

                            policy_feed_dict = {
                                my_agent.input_mention_context: current_mention_batch,
                                my_agent.input_entity_desc: cuurent_entity_batch,
                                my_agent.input_entity_embedding: current_entity_embedd_batch,
                                my_agent.input_entity_fea: current_entity_fea_batch,
                                my_agent.xgb_score: current_score_batch,
                                my_agent.last_state: global_last_state
                            }

                            # shape=(S,B,H)
                            seq_last_state_list.append(np.squeeze(global_last_state, axis=1))

                            # 通过output以一定概率形式选择action
                            action_dist = policy_session.run(my_agent.action_output, feed_dict=policy_feed_dict)
                            action_dist = np.squeeze(action_dist, axis=1)
                            current_action_batch = []
                            for action_prob in action_dist:
                                new_action_prob = [prob/sum(action_prob) for prob in action_prob]
                                # 直接按照生成的概率随机选择
                                action = np.random.choice(len(new_action_prob), p=new_action_prob)
                                current_action_batch.append(action)

                            # 获取即时reward
                            current_label = label_batch[:, mention_index, :, :]
                            current_label_batch = [np.argmax(current_label[batch_index, action_index, :], axis=-1)
                                            for batch_index, action_index in enumerate(current_action_batch)]

                            current_reward_batch = []
                            for label in current_label_batch:
                                # 正例
                                if label == 1:
                                    current_reward_batch.append(1)
                                # 负例
                                elif label == 0:
                                    current_reward_batch.append(0)

                            # shape=(S,B)
                            policy_action_list.append(current_action_batch)
                            policy_reward_list.append(current_reward_batch)

                            # 构造global模型的输入数据
                            # shape=(S, B, H)
                            seq_mention_list.append(mention_batch[:, mention_index, 0, :])
                            # shape=(B, S, H)
                            global_mention_input = np.stack(seq_mention_list, axis=1)

                            entity_desc_batch = entity_batch[:, mention_index, :, :]
                            embedd_batch = entity_embedd_batch[:, mention_index, :, :]
                            entity_fea_batch = fea_batch[:, mention_index, :, :]
                            entity_score_batch = score_batch[:, mention_index, :]

                            select_entity_batch = [entity_desc_batch[batch_index, select_index, :]
                                                   for batch_index, select_index in enumerate(current_action_batch)]
                            select_entity_embedd_batch = [embedd_batch[batch_index, select_index, :]
                                                       for batch_index, select_index in enumerate(current_action_batch)]
                            select_entity_fea_batch = [entity_fea_batch[batch_index, select_index, :]
                                                       for batch_index, select_index in enumerate(current_action_batch)]
                            select_entity_score_batch = [entity_score_batch[batch_index, select_index]
                                                       for batch_index, select_index in enumerate(current_action_batch)]

                            # shape=(S, B, H)
                            seq_select_entity_list.append(select_entity_batch)
                            seq_select_fea_list.append(select_entity_fea_batch)
                            seq_select_embedd_list.append(select_entity_embedd_batch)
                            # shape = (S,B)
                            seq_select_score_list.append(select_entity_score_batch)

                            # shape=(B,S,H)
                            global_entity_input = np.stack(seq_select_entity_list, axis=1)
                            global_fea_input = np.stack(seq_select_fea_list, axis=1)
                            global_embedd_input = np.stack(seq_select_embedd_list, axis=1)

                            global_feed_dict = {
                                global_model.input_mention_context: global_mention_input,
                                global_model.input_entity_desc: global_entity_input,
                                global_model.input_entity_embedding: global_embedd_input,
                                global_model.input_entity_fea: global_fea_input,
                                global_model.keep_prob: 1.0
                            }
                            # 获取global模型的输出
                            global_cell_output = global_session.run(global_model.rnn_output, feed_dict=global_feed_dict)
                            global_last_state = global_cell_output[:, -1, :]
                            global_last_state = np.expand_dims(global_last_state, axis=1)

                        # 周期结束,更新网络
                        # shape=(B,S)
                        policy_action_np = np.stack(policy_action_list, axis=1)
                        # print policy_action_np

                        # 计算action_value
                        policy_reward_np = np.stack(policy_reward_list, axis=1)
                        policy_action_value = self.discount_rewards(policy_reward_np)

                        seq_feed_dict = {
                            my_agent.action_holder: policy_action_np,
                            my_agent.reward_holder: policy_action_value,
                            my_agent.input_mention_context: mention_batch,
                            my_agent.input_entity_desc: entity_batch,
                            my_agent.input_entity_embedding: entity_embedd_batch,
                            my_agent.input_entity_fea: fea_batch,
                            my_agent.xgb_score: score_batch,
                            my_agent.last_state: np.stack(seq_last_state_list, axis=1)
                        }

                        # 输出在测试集上的性能
                        if total_batch % self.policy_config.print_per_batch == 0:
                            loss_train = policy_session.run(my_agent.loss, feed_dict=seq_feed_dict)
                            acc_train = float(sum([sum(seq_reward) for seq_reward in policy_reward_np])) / policy_reward_np.size

                            time_dif = self.data_util.get_time_dif(start_time)
                            msg = 'Iter: {0:>6}, Train Loss: {1:>8.2}, Train Acc: {2:>7.2%},' \
                                  + 'Time: {3}'

                            print(msg.format(total_batch, loss_train, acc_train, time_dif))

                            if total_batch % 25 == 0:
                                loss_val, acc_val = self.evaluate(policy_session, global_session, my_agent, global_model, test_data_list)

                                # 保存最好结果
                                if acc_val > best_acc_val:
                                    best_acc_val = acc_val
                                    last_improved = total_batch
                                    policy_saver.save(sess=policy_session, save_path=self.policy_config.save_policy_dir)
                                    improved_str = '*'
                                else:
                                    improved_str = ''

                                time_dif = self.data_util.get_time_dif(start_time)
                                msg = 'Iter: {0:>6}, Train Loss: {1:>8.2}, Train Acc: {2:>7.2%},' \
                                      + ' Val Loss: {3:>8.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'

                                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif,
                                                 improved_str))

                        # 对loss进行优化
                        policy_session.run(my_agent.optim, feed_dict=seq_feed_dict)
                        total_batch += 1

                        # 验证集正确率长期不提升，提前结束训练
                        if total_batch - last_improved > self.policy_config.policy_require_improvement:
                            print("No optimization for a long time, auto-stopping...")
                            early_stop_flag = True
                            break

                    # early stopping
                    if early_stop_flag:
                        break

        global_session.close()
        policy_session.close()

    def test_agent(self):
        """
        测试训练好的模型
        :return:
        """
        print("Test Agent...")

        # 加载测试数据
        test_data = self.load_policy_data(
                    self.policy_config.test_local_mention_rank_path,
                    self.policy_config.local_rep_test_path,
                    data_name="Test")

        # 创建全局模型的graph
        global_graph = tf.Graph()
        global_session = tf.Session(graph=global_graph)
        # 加载全局模型
        with global_session.as_default():
            with global_graph.as_default():
                global_model = GlobalModel(self.global_config)
                global_session.run(tf.global_variables_initializer())
                global_saver = tf.train.Saver()
                global_saver.restore(sess=global_session, save_path=self.global_config.save_global_dir)

        # 创建policy模型的graph
        policy_graph = tf.Graph()
        policy_session = tf.Session(graph=policy_graph)
        with policy_session.as_default():
            with policy_graph.as_default():
                my_agent = Agent(self.policy_config)
                policy_session.run(tf.global_variables_initializer())
                policy_saver = tf.train.Saver()
                policy_saver.restore(sess=policy_session, save_path=self.policy_config.save_policy_dir)
                loss_val, acc_val = self.evaluate(policy_session, global_session, my_agent, global_model, test_data, is_test=True)

                print(loss_val, acc_val)

    def load_policy_result(self):
        # shape=(B,S)
        batch_seq_reward = np.load(self.policy_config.test_policy_reward_path)

        file_group_dict = self.data_util.get_file_group(self.policy_config.test_local_mention_rank_path)

        for s in batch_seq_reward:
            print(s)

        start_id = 0
        group_reward_dict = {}
        for mention_file, group_list in file_group_dict.items():
            seq_num = ((len(group_list) - 1) // self.policy_config.sequence_len) + 1
            reward_batch = batch_seq_reward[start_id:start_id+seq_num]
            start_id = start_id+seq_num

            group_start_index = 0
            for seq_index in range(seq_num):
                seq_group_list = group_list[group_start_index:group_start_index+self.policy_config.sequence_len]
                seq_reward_list = reward_batch[seq_index].tolist()
                for i, group_id in enumerate(seq_group_list):
                    group_reward_dict[group_id] = seq_reward_list[i]

                group_start_index = group_start_index+self.policy_config.sequence_len

        group_item_dict = self.data_util.get_group_list(self.policy_config.test_local_mention_rank_path)
        with open(self.policy_config.test_policy_pred_path, "w", encoding="utf-8") as policy_pred_file:
            for group_id, item_list in group_item_dict.items():
                if group_id not in group_reward_dict:
                    continue
                reward = group_reward_dict[group_id]

                if reward == 1:
                    for index, item in enumerate(item_list):
                        group_str, label_str, fea_str, mention_str, entity_str = item.split("\t")
                        if int(label_str) == 1:
                            policy_pred_file.write(item + "\n")
                else:
                    policy_pred_file.write(item_list[0].strip() + "\n")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    data_util = DataUtil()
    global_config = GlobalConfig()
    policy_config = PolicyConfig()
    model_helper = ModelHelper(data_util)

    agent_process = AgentProcess(global_config, policy_config, model_helper, data_util)

    agent_process.train_agent()
    agent_process.test_agent()
    agent_process.load_policy_result()
    data_util.cal_recall(policy_config.test_policy_pred_path)