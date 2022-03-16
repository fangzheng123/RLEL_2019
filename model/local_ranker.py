# encoding: utf-8

import json
import numpy as np

from model_config import LocalConfig
from data_util import DataUtil


class LocalRanker(object):
    """
    rank mention
    """

    def __init__(self, local_config):
        self.local_config = local_config
        self.data_util = DataUtil()

    def cal_recall(self, data_path):
        """

        :param data_path:
        :return:
        """
        right_num = 0
        group_set = set()
        with open(data_path, "r", encoding="utf-8") as data_file:
            for item in data_file:
                item = item.strip()

                group_str, label_str, fea_str, mention_str, entity_str = item.split("\t")
                group_id = int(group_str)

                if group_id not in group_set:
                    if int(label_str) == 1:
                        right_num += 1

                    group_set.add(group_id)

        print(right_num, len(group_set), right_num/len(group_set))

    def get_not_recall(self, data_path):
        """

        :param data_path:
        :return:
        """
        group_set = set()
        not_recall_set = set()
        with open(data_path, "r", encoding="utf-8") as data_file:
            for item in data_file:
                item = item.strip()

                group_str, label_str, fea_str, mention_str, entity_str = item.split("\t")
                group_id = int(group_str)

                if group_id not in group_set:
                    if int(label_str) != 1:
                        not_recall_set.add(group_id)

                    group_set.add(group_id)

        return not_recall_set

    def rank_candidate(self):
        """
        Sort the candidates according to the prediction results of the local model
        :param data_list:
        :param local_model:
        :param is_random:
        :return:
        """
        print('Rank Candidate...')

        group_dict = self.data_util.get_group_list(self.local_config.test_local_pred_path)
        with open(self.local_config.test_local_candidate_rank_path, "w", encoding="utf-8") as local_candidate_rank_file:
            for group_id, item_list in group_dict.items():
                pred_dict = {}
                new_item_list = []
                for index, item in enumerate(item_list):
                    group_str, label_str, fea_str, mention_str, entity_str = item.split("\t")
                    entity_obj = json.loads(entity_str)

                    if "local_pred" in entity_obj:
                        pred_dict[index] = entity_obj["local_pred"]

                pred_sort_list = [i for i, val in sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)]
                if len(pred_sort_list) > 0:
                    for index in pred_sort_list:
                        new_item_list.append(item_list[index])
                else:
                    new_item_list = item_list

                for item in new_item_list:
                    local_candidate_rank_file.write(item + "\n")


    def add_father_candidate(self):
        """
        Make the father target entity as the current mention's target entity
        :return:
        """
        mention_candidate_dict = {}
        mention_form_dict = {}
        with open(self.local_config.test_local_candidate_rank_path, "r", encoding="utf-8") as local_rank_file:
            for item in local_rank_file:
                item = item.strip()

                group_str, label_str, fea_str, mention_str, entity_str = item.split("\t")
                group_id = int(group_str)
                mention_obj = json.loads(mention_str)
                entity_obj = json.loads(entity_str)

                if group_id not in mention_candidate_dict:
                    mention_candidate_dict[group_id] = [entity_obj]
                    mention_form_dict[mention_obj["mention_form"]] = [group_id]
                else:
                    mention_candidate_dict[group_id].append(entity_obj)
                    mention_form_dict[mention_obj["mention_form"]].append(group_id)

        group_item_dict = {}
        with open(self.local_config.test_local_candidate_rank_path, "r", encoding="utf-8") as local_rank_file:
            for item in local_rank_file:
                item = item.strip()

                group_str, label_str, fea_str, mention_str, entity_str = item.split("\t")

                group_id = int(group_str)
                label = int(label_str)
                mention_obj = json.loads(mention_str)
                entity_obj = json.loads(entity_str)
                father_mention_list = mention_obj["father_mention"]
                if len(father_mention_list) > 0 and father_mention_list[0] in mention_form_dict:
                    father_target = mention_candidate_dict[mention_form_dict[father_mention_list[0]][0]][0]

                    if father_target["source_name"] == mention_obj["target_name"]:
                        label = 1

                    new_item = "\t".join([group_str, str(label), fea_str, mention_str, json.dumps(father_target)])
                    group_item_dict[group_id] = [new_item]

                else:
                    if group_id not in group_item_dict:
                        group_item_dict[group_id] = [item]
                    else:
                        group_item_dict[group_id].append(item)

        with open(self.local_config.test_local_candidate_rank_path, "w", encoding="utf-8") as local_rank_file:
            for group_id, item_list in group_item_dict.items():
                for item in item_list:
                    local_rank_file.write(item + "\n")

            print(len(group_item_dict))

    def rank_mention(self, is_train=False):
        """
        Sort the mentions according to the prediction results of the local model
        :param data_path:
        :param is_train: is train data
        :return:
        """
        print('Rank Mention...')

        # rank train data
        if is_train:
            rank_candidate_path = self.local_config.train_local_candidate_rank_path
        # rank test data
        else:
            rank_candidate_path = self.local_config.test_local_candidate_rank_path

        group_item_dict = self.data_util.get_group_list(rank_candidate_path)
        file_group_dict = self.data_util.get_file_group(rank_candidate_path)
        pre_group_list = []
        next_group_list = []
        all_next_group_list = []
        rank_group_list = []
        rank_file_group_dict = {}
        for group_id, item_list in group_item_dict.items():
            group_str, label_str, fea_str, mention_str, entity_str = item_list[0].split("\t")

            group_id = int(group_str)
            label = int(label_str)
            fea = json.loads(fea_str)
            mention_obj = json.loads(mention_str)
            mention_file = mention_obj["mention_file"]

            current_file_group_list = file_group_dict[mention_file]

            if is_train:
                if label == 1:
                    pre_group_list.append(group_id)
                else:
                    next_group_list.append(group_id)
            else:
                if fea["same_candidate_word_num"] > 0:
                    pre_group_list.append(group_id)
                else:
                    next_group_list.append(group_id)

            if len(pre_group_list) + len(next_group_list) == self.local_config.sequence_len or \
                    len(rank_group_list) + self.local_config.sequence_len > len(current_file_group_list):
                rank_group_list.extend(pre_group_list.copy())
                rank_group_list.extend(next_group_list.copy())
                pre_group_list = []
                next_group_list = []

            if len(rank_group_list) == len(current_file_group_list):
                rank_file_group_dict[mention_file] = rank_group_list.copy()
                rank_group_list = []

        print("source mention num:{0}, rank mention num: {1}".format(sum([len(groups) for _, groups in file_group_dict.items()]),
                                                                     sum([len(groups) for _, groups in rank_file_group_dict.items()])))

        if is_train:
            rank_mention_path = self.local_config.train_local_mention_rank_path
        else:
            rank_mention_path = self.local_config.test_local_mention_rank_path

        with open(rank_mention_path, "w", encoding="utf-8") as rank_mention_file:
            for _, group_list in rank_file_group_dict.items():
                for group_id in group_list:
                    item_list = group_item_dict[group_id]
                    for item in item_list:
                        rank_mention_file.write(item + "\n")

    def controller(self):
        self.rank_candidate()
        self.cal_recall(self.local_config.test_local_candidate_rank_path)

        self.add_father_candidate()
        self.cal_recall(self.local_config.test_local_candidate_rank_path)

        self.rank_mention()

if __name__ == "__main__":
    local_config = LocalConfig()
    local_ranker = LocalRanker(local_config)

    local_ranker.controller()