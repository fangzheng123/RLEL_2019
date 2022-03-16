# encoding: utf-8


import json
import os

import numpy as np
from sklearn import preprocessing

from data_util import DataUtil
from model_config import LocalConfig, GlobalConfig, PolicyConfig

class ModelHelper(object):

    def __init__(self, data_util):
        self.data_util = data_util
        self.local_config = LocalConfig()
        self.global_config = GlobalConfig()
        self.policy_config = PolicyConfig()

        self.word_embed = {}
        self.all_entity_embed = {}
        # self.word_embed = self.data_util.load_word_embed()
        # self.all_entity_embed = self.data_util.load_entity_embed()

    def get_entity_desc(self, entity_obj):
        """
        build entity description
        :param entity_obj:
        :return:
        """
        summary_keywords = ""
        if "summary_keywords" in entity_obj:
            summary_keywords = " ".join(entity_obj["summary_keywords"]).lower()
            summary_keywords = self.data_util.remove_special_char(summary_keywords)

        category = ""
        if "category" in entity_obj:
            category = " ".join(entity_obj["category"]).lower()
            category = self.data_util.remove_special_char(category)

        entity_desc = " ".join(summary_keywords.split(" ")[:30]) + " " + category

        return entity_desc

    def get_mention_context(self, mention_obj):
        """
        build mention context
        :param mention_obj:
        :return:
        """
        if "mention_context" in mention_obj:
            mention_context = mention_obj["mention_context"].lower()
            tmp_context = " ".join(mention_context.split(" ")[:-3])
            mention_context = " ".join(mention_context.split(" ")[-3:]) + " " + tmp_context
        else:
            mention_context = " ".join(
                [" ".join(mention_obj["mention_left_context"].split(" ")[::-1][:30][::-1]),
                 " ".join(mention_obj["mention_right_context"].split(" ")[:30])]).lower()

        mention_context = mention_context.replace("soccer", "football").replace("nfl", "football").replace("nba", "basketball")

        return mention_context

    def normalize_feature(self, all_feature_list):
        """
        Min-Max normalization
        :param all_feature_list:
        :return:
        """
        mention_num = len(all_feature_list)
        candidate_num = self.local_config.candidate_num

        all_feas = [feas for candidate_fea_list in all_feature_list for feas in candidate_fea_list]
        norm_feas = preprocessing.minmax_scale(np.array(all_feas))

        all_norm_fea = np.reshape(norm_feas, (mention_num, candidate_num, -1))

        return all_norm_fea

    def sent2embed(self, sent):
        """
        convert sent to embedding
        :param sent:
        :return:
        """
        sent = sent.lower()
        word_list = [word for word in sent.split(" ") if word != ""]
        word_embedd_list = [self.word_embed[word] for word in word_list if word in self.word_embed]

        return word_embedd_list

    def load_tmp_data(self, candidate_rank_path):
        """

        :param candidate_rank_path:
        :return:
        """
        print("load tmp data")
        mention_arr = np.load(candidate_rank_path + "_data_tmp_mention.npy")
        desc_arr = np.load(candidate_rank_path + "_data_tmp_desc.npy")
        embed_arr = np.load(candidate_rank_path + "_data_tmp_embed.npy")
        fea_arr = np.load(candidate_rank_path + "_data_tmp_fea.npy")
        score_arr = np.load(candidate_rank_path + "_data_tmp_score.npy")
        label_arr = np.load(candidate_rank_path + "_data_tmp_label.npy")

        # fea_list = fea_arr.tolist()
        # new_fea_list = []
        # for candidate_list in fea_list:
        #     new_can_list = []
        #     for feas in candidate_list:
        #         new_feas = [val for index, val in enumerate(feas) if index in [1, 2]]
        #         new_feas = new_feas * 10
        #         new_can_list.append(new_feas)
        #
        #     new_fea_list.append(new_can_list)
        #
        # fea_arr = np.array(new_fea_list)

        return mention_arr, desc_arr, embed_arr, fea_arr, score_arr, label_arr

    def load_local_data(self, candidate_rank_path, is_save_tmp=False, is_load_tmp=False, is_filter=True):
        """
        load data for local model
        :param candidate_rank_path:
        :return:
        """
        if is_load_tmp and is_filter:
            return self.load_tmp_data(candidate_rank_path)

        all_mention_context_list = []
        all_entity_desc_list = []
        all_entity_embed_list = []
        all_fea_list = []
        all_score_list = []
        all_label_list = []

        group_dict = {}
        with open(candidate_rank_path, "r", encoding="utf-8") as candidate_rank_file:
            for item in candidate_rank_file:
                item = item.strip()

                group_str, label_str, fea_str, mention_str, entity_str = item.split("\t")
                group_id = int(group_str)

                if group_id not in group_dict:
                    group_dict[group_id] = [item]
                else:
                    group_dict[group_id].append(item)

        filter_group_dict = {}
        if is_filter:
            not_recall_count = 0
            # filter data which has only one candidate or are all false candidate
            for group_id, item_list in group_dict.items():
                label_list = [int(item.split("\t")[1]) for item in item_list]
                if sum(label_list) == 0:
                    not_recall_count += 1
                if sum(label_list) == 0 or sum(label_list) == len(label_list):
                    continue

                filter_group_dict[group_id] = item_list

            pred_right_num = 0
            for group_id, item_list in filter_group_dict.items():
                label_list = [int(item.split("\t")[1]) for item in item_list]
                if label_list[0] == 1:
                    pred_right_num += 1

            print("all num :{0}, not recall num:{1}, pred acc:{2}".format(sum([len(item_list) for _, item_list in group_dict.items()]),
                                                            not_recall_count, pred_right_num/len(filter_group_dict)))

            # save groups which are tested in local model
            json.dump(list(filter_group_dict.keys()), open(candidate_rank_path+"_lstm_group", "w"))
        else:
            # for saving local representations
            filter_group_dict = group_dict

        for group_id, item_list in filter_group_dict.items():
            candidate_mention_list = []
            candidate_entity_list = []
            candidate_embed_list = []
            candidate_fea_list = []
            candidate_score_list = []
            candidate_label_list = []
            group_num = len(item_list)

            for item in item_list:
                item = item.strip()

                group_str, label_str, fea_str, mention_str, entity_str = item.split("\t")
                mention_obj = json.loads(mention_str)
                entity_obj = json.loads(entity_str)

                mention_context = self.get_mention_context(mention_obj)
                mention_context = " ".join(mention_context.split(" ")[:60])
                mention_context_embed = self.sent2embed(mention_context)[:self.local_config.seq_length]

                entity_desc = self.get_entity_desc(entity_obj)
                entity_desc = " ".join(entity_desc.split(" ")[:60])
                entity_desc_embed = self.sent2embed(entity_desc)[:self.local_config.seq_length]

                # padding sent
                zero_padding = list(np.zeros(self.local_config.word_embedd_dim))
                if len(mention_context_embed) < self.local_config.seq_length:
                    mention_context_embed.extend([zero_padding for i in range(self.local_config.seq_length - len(mention_context_embed))])

                if len(entity_desc_embed) < self.local_config.seq_length:
                    entity_desc_embed.extend([zero_padding for i in range(self.local_config.seq_length - len(entity_desc_embed))])

                entity_embed = zero_padding
                if "url" in entity_obj:
                    entity_url = entity_obj["url"].split("/")[-1].lower()
                    if entity_url in self.all_entity_embed:
                        entity_embed = self.all_entity_embed[entity_url]

                fea_obj = json.loads(fea_str)
                feature = [val for name, val in fea_obj.items()]
                # xgb_score = entity_obj["xgboost_pred"]

                xgb_score = (5 - entity_obj["xgboost_rank_position"]) * 0.2

                # shape = (S, H)
                candidate_mention_list.append(mention_context_embed)
                candidate_entity_list.append(entity_desc_embed)
                candidate_embed_list.append(entity_embed)
                candidate_fea_list.append(feature)
                candidate_score_list.append(xgb_score)
                candidate_label_list.append(int(label_str))

            # padding entity
            if group_num != self.local_config.candidate_num:
                pad_mention = candidate_mention_list[-1]
                pad_entity = candidate_entity_list[-1]
                pad_embed = candidate_embed_list[-1]
                pad_fea = candidate_fea_list[-1]
                pad_score = candidate_score_list[-1]
                pad_label = candidate_label_list[-1]
                for i in range(self.local_config.candidate_num-group_num):
                    candidate_mention_list.append(pad_mention)
                    candidate_entity_list.append(pad_entity)
                    candidate_embed_list.append(pad_embed)
                    candidate_fea_list.append(pad_fea)
                    candidate_score_list.append(pad_score)
                    candidate_label_list.append(pad_label)

            # # label to one hot
            # candidate_label_np = np.eye(self.local_config.class_num)[candidate_label_list]

            all_mention_context_list.append(candidate_mention_list)
            all_entity_desc_list.append(candidate_entity_list)
            all_entity_embed_list.append(candidate_embed_list)
            all_fea_list.append(candidate_fea_list)
            all_score_list.append(candidate_score_list)
            # shape=(B,C,1)
            all_label_list.append(candidate_label_list)

        # normalize feature
        # norm_feas = self.normalize_feature(all_fea_list)

        if is_save_tmp:
            np.save(candidate_rank_path + "_data_tmp_mention", np.array(all_mention_context_list))
            np.save(candidate_rank_path + "_data_tmp_desc", np.array(all_entity_desc_list))
            np.save(candidate_rank_path + "_data_tmp_embed", np.array(all_entity_embed_list))
            np.save(candidate_rank_path + "_data_tmp_fea", all_fea_list)
            np.save(candidate_rank_path + "_data_tmp_score", np.array(all_score_list))
            np.save(candidate_rank_path + "_data_tmp_label", np.array(all_label_list))

        return np.array(all_mention_context_list), np.array(all_entity_desc_list), \
               np.array(all_entity_embed_list), np.array(all_fea_list), np.array(all_score_list), np.array(all_label_list)

    def local_batch_iter(self, data_list, is_random=False):
        """
        build batch data for local model
        :param data_list:
        :return:
        """
        mention_context_array, entity_desc_array, entity_embed_array, fea_array, score_array, label_array = data_list

        batch_size = self.local_config.local_batch_size
        all_data_num = mention_context_array.shape[0]
        batch_num = int((all_data_num - 1) / batch_size) + 1

        # shuffle
        if is_random:
            indices = np.random.permutation(np.arange(all_data_num))
            mention_context_array = mention_context_array[indices]
            entity_desc_array = entity_desc_array[indices]
            entity_embed_array = entity_embed_array[indices]
            fea_array = fea_array[indices]
            score_array = score_array[indices]
            label_array = label_array[indices]

        for i in range(batch_num):
            start_id = i * self.local_config.local_batch_size
            end_id = min((i + 1) * batch_size, all_data_num)
            if end_id == all_data_num:
                yield mention_context_array[max(end_id - batch_size, 0):end_id], \
                      entity_desc_array[max(end_id - batch_size, 0):end_id], \
                      entity_embed_array[max(end_id - batch_size, 0):end_id], \
                      fea_array[max(end_id - batch_size, 0):end_id], \
                      score_array[max(end_id - batch_size, 0):end_id], \
                      label_array[max(end_id - batch_size, 0):end_id]
            else:
                yield mention_context_array[start_id:end_id], entity_desc_array[start_id:end_id], \
                      entity_embed_array[start_id:end_id], fea_array[start_id:end_id], \
                      score_array[start_id:end_id], label_array[start_id:end_id]

    def load_global_data(self, rank_mention_path, rank_rep_path):
        """

        :param rank_file:
        :param rank_rep_path:
        :return:
        """
        rank_rep_array = np.load(rank_rep_path)

        group_item_dict = self.data_util.get_group_list(rank_mention_path)
        file_group_dict = self.data_util.get_file_group(rank_mention_path)

        index = 0
        all_group = 0
        all_mention_context_list = []
        all_entity_desc_list = []
        all_entity_embed_list = []
        all_fea_list = []
        all_score_list = []
        all_label_list = []
        for mention_file, group_list in file_group_dict.items():
            all_group += len(group_list)

            for group_id in group_list:
                item_list = group_item_dict[group_id]

                candidate_mention_list = []
                candidate_entity_list = []
                candidate_embed_list = []
                candidate_fea_list = []
                candidate_score_list = []
                candidate_label_list = []
                group_num = len(item_list)

                for item in item_list:
                    item = item.strip()

                    group_str, label_str, fea_str, mention_str, entity_str = item.split("\t")
                    entity_obj = json.loads(entity_str)
                    fea_obj = json.loads(fea_str)

                    if entity_obj["xgboost_rank_position"] == 1 and int(label_str) == 1 \
                            and fea_obj["same_candidate_word_num"] > 0:
                        entity_obj["xgboost_rank_position"] = 0

                    zero_padding = list(np.zeros(self.local_config.word_embedd_dim))
                    entity_embed = zero_padding
                    if "url" in entity_obj:
                        entity_url = entity_obj["url"].split("/")[-1].lower()
                        if entity_url in self.all_entity_embed:
                            entity_embed = self.all_entity_embed[entity_url]

                    feature = [val for name, val in fea_obj.items()]
                    xgb_score = (5 - entity_obj["xgboost_rank_position"]) * 0.2

                    # shape = (S, H)
                    candidate_mention_list.append(rank_rep_array[index][0])
                    candidate_entity_list.append(rank_rep_array[index][1])
                    candidate_embed_list.append(entity_embed)
                    candidate_fea_list.append(feature)
                    candidate_score_list.append(xgb_score)
                    candidate_label_list.append(int(label_str))
                    index += 1

                # entity padding
                if group_num != self.local_config.candidate_num:
                    pad_mention = candidate_mention_list[-1]
                    pad_entity = candidate_entity_list[-1]
                    pad_embed = candidate_embed_list[-1]
                    pad_fea = candidate_fea_list[-1]
                    pad_score = candidate_score_list[-1]
                    pad_label = candidate_label_list[-1]
                    for i in range(self.local_config.candidate_num - group_num):
                        candidate_mention_list.append(pad_mention)
                        candidate_entity_list.append(pad_entity)
                        candidate_embed_list.append(pad_embed)
                        candidate_fea_list.append(pad_fea)
                        candidate_score_list.append(pad_score)
                        candidate_label_list.append(pad_label)
                        index += 1

                # label to one hot, shape=(C,class_num)
                candidate_label_np = np.eye(self.local_config.class_num)[candidate_label_list]

                all_mention_context_list.append(candidate_mention_list)
                all_entity_desc_list.append(candidate_entity_list)
                all_entity_embed_list.append(candidate_embed_list)
                all_fea_list.append(candidate_fea_list)
                all_score_list.append(candidate_score_list)
                all_label_list.append(candidate_label_np)

            if len(group_list) % self.local_config.sequence_len != 0:
                remainder = len(group_list) % self.local_config.sequence_len

                # mention padding
                pad_mention = all_mention_context_list[-1]
                pad_entity = all_entity_desc_list[-1]
                pad_embed = all_entity_embed_list[-1]
                pad_fea = all_fea_list[-1]
                pad_score = all_score_list[-1]
                pad_label = all_label_list[-1]
                for i in range(self.local_config.sequence_len - remainder):
                    all_mention_context_list.append(pad_mention)
                    all_entity_desc_list.append(pad_entity)
                    all_entity_embed_list.append(pad_embed)
                    all_fea_list.append(pad_fea)
                    all_score_list.append(pad_score)
                    all_label_list.append(pad_label)

        print(len(all_mention_context_list), len(all_entity_desc_list),
              len(all_entity_embed_list), len(all_fea_list), len(all_score_list), len(all_label_list),
              all_group*self.local_config.candidate_num, index, len(rank_rep_array))

        # shape=(B,C,H)
        return np.array(all_mention_context_list), np.array(all_entity_desc_list), np.array(all_entity_embed_list), \
               np.array(all_fea_list), np.array(all_score_list), np.array(all_label_list)

    def global_batch_iter(self, data_list):
        """
        build batch data for global model
        :param data_list:
        :return:
        """
        # shape = (B,C,H)
        mention_context_array, entity_desc_array, entity_embed_array, fea_array, score_array, label_array = data_list

        # print(mention_context_array.shape, entity_desc_array.shape, entity_embed_array.shape, fea_array.shape, score_array.shape, label_array.shape)

        new_mention_context_list = []
        new_entity_desc_list = []
        new_entity_embed_list = []
        new_fea_list = []
        new_score_list = []
        new_label_list = []

        for index in range(len(mention_context_array)):
            label_list = np.argmax(label_array[index], axis=-1).tolist()
            right_index = 1
            if 1 in label_list:
                right_index = label_list.index(1)
                # if right_index > 2:
                #     right_index = 1

            # 以一定概率生成候选下标
            # p_list = [0.5 for i in range(2)]
            # p_list[right_index] = 0.6
            choose_index = np.random.choice(2, p=[0.7, 0.3]) + 1

            if right_index == 0:
                choose_index = right_index

            new_mention_context_list.append(mention_context_array[index, choose_index, :])
            new_entity_desc_list.append(entity_desc_array[index, choose_index, :])
            new_entity_embed_list.append(entity_embed_array[index, choose_index, :])
            new_fea_list.append(fea_array[index, choose_index, :])
            new_score_list.append(score_array[index, choose_index])
            new_label_list.append(label_array[index, choose_index, :])

        mention_context_array = np.reshape(np.array(new_mention_context_list), (-1, self.global_config.sequence_len, self.global_config.local_rep_dim))
        entity_desc_array = np.reshape(np.array(new_entity_desc_list), (-1, self.global_config.sequence_len, self.global_config.local_rep_dim))
        entity_embed_array = np.reshape(np.array(new_entity_embed_list), (-1, self.global_config.sequence_len, self.global_config.word_embedd_dim))
        fea_array = np.reshape(np.array(new_fea_list), (-1, self.global_config.sequence_len, self.global_config.fea_dim))
        score_array = np.reshape(np.array(new_score_list), (-1, self.global_config.sequence_len))
        label_array = np.reshape(np.array(new_label_list), (-1, self.global_config.sequence_len, self.global_config.class_num))

        # print(mention_context_array.shape, entity_desc_array.shape, entity_embed_array.shape, fea_array.shape, score_array.shape, label_array.shape)

        batch_size = self.global_config.global_batch_size
        all_data_num = mention_context_array.shape[0]
        batch_num = int((all_data_num - 1) / batch_size) + 1

        for i in range(batch_num):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, all_data_num)
            if end_id == all_data_num:
                yield mention_context_array[max(end_id - batch_size, 0):end_id], \
                      entity_desc_array[max(end_id - batch_size, 0):end_id], \
                      entity_embed_array[max(end_id - batch_size, 0):end_id], \
                      fea_array[max(end_id - batch_size, 0):end_id], \
                      score_array[max(end_id - batch_size, 0):end_id], \
                      label_array[max(end_id - batch_size, 0):end_id]
            else:
                yield mention_context_array[start_id:end_id], entity_desc_array[start_id:end_id], \
                      entity_embed_array[start_id:end_id], fea_array[start_id:end_id], \
                      score_array[start_id:end_id], label_array[start_id:end_id]

    def load_policy_data(self, rank_mention_path, rank_rep_path, is_train=False):
        """
        load data for policy model
        :param rank_mention_path:
        :param rank_rep_path:
        :param is_train:
        :return:
        """
        # shape = (B, C, H)
        mention_context_array, entity_desc_array, entity_embed_array, fea_array, score_array, label_array = \
            self.load_global_data(rank_mention_path, rank_rep_path)

        # shape=(B,S,C,H)
        mention_context_array = np.reshape(mention_context_array, (-1, self.policy_config.sequence_len,
                                                                   self.policy_config.candidate_num,
                                                                   self.policy_config.local_rep_dim))
        entity_desc_array = np.reshape(entity_desc_array, (-1, self.policy_config.sequence_len,
                                                                   self.policy_config.candidate_num,
                                                                   self.policy_config.local_rep_dim))
        entity_embed_array = np.reshape(entity_embed_array, (-1, self.policy_config.sequence_len,
                                                                   self.policy_config.candidate_num,
                                                                   self.policy_config.word_embedd_dim))
        fea_array = np.reshape(fea_array, (-1, self.policy_config.sequence_len, self.policy_config.candidate_num,
                                                                   self.policy_config.fea_dim))
        # shape=(B,S,C)
        score_array = np.reshape(score_array, (-1, self.policy_config.sequence_len, self.policy_config.candidate_num))

        label_array = np.reshape(label_array, (-1, self.policy_config.sequence_len, self.policy_config.candidate_num, self.policy_config.class_num))

        return mention_context_array, entity_desc_array, entity_embed_array, fea_array, score_array, label_array

    def policy_batch_iter(self, data_list, is_random=False):
        """
        build batch data for policy model
        :param data_list:
        :return:
        """
        # shape=(B,S,C,H)
        mention_context_array, entity_desc_array, entity_embed_array, fea_array, score_array, label_array = data_list

        batch_size = self.policy_config.policy_batch_size
        all_data_num = mention_context_array.shape[0]
        batch_num = int((all_data_num - 1) / batch_size) + 1

        # shuffle
        if is_random:
            indices = np.random.permutation(np.arange(all_data_num))
            mention_context_array = mention_context_array[indices]
            entity_desc_array = entity_desc_array[indices]
            entity_embed_array = entity_embed_array[indices]
            fea_array = fea_array[indices]
            score_array = score_array[indices]
            label_array = label_array[indices]

        for i in range(batch_num):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, all_data_num)
            if end_id == all_data_num:
                yield mention_context_array[max(end_id - batch_size, 0):end_id], \
                      entity_desc_array[max(end_id - batch_size, 0):end_id], \
                      entity_embed_array[max(end_id - batch_size, 0):end_id], \
                      fea_array[max(end_id - batch_size, 0):end_id], \
                      score_array[max(end_id - batch_size, 0):end_id], \
                      label_array[max(end_id - batch_size, 0):end_id]
            else:
                yield mention_context_array[start_id:end_id], entity_desc_array[start_id:end_id], \
                      entity_embed_array[start_id:end_id], fea_array[start_id:end_id], \
                      score_array[start_id:end_id], label_array[start_id:end_id]


if __name__ == "__main__":
    data_util = DataUtil()
    model_helper = ModelHelper(data_util)
    local_config = LocalConfig()

    # model_helper.load_local_data(local_config.train_path, is_save_tmp=True)
    # model_helper.load_local_data(local_config.test_path, is_save_tmp=True)

    # data_list = model_helper.load_global_data(local_config.test_local_mention_rank_path, local_config.local_rep_test_path)
    # for i in model_helper.global_batch_iter(data_list):
    #     pass