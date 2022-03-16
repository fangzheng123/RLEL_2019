# encoding: utf-8

import json
import numpy as np
import xgboost as xgb
import data_util

class XgboostRank(object):
    def __init__(self, model_path, data_util):
        self.model = None
        self.param = {'booster': 'gbtree', 'max_depth': 4, 'eta': 0.01, 'verbosity': 1, 'objective': 'rank:pairwise',
                      'gamma': 0.2, 'lambda': 500, 'subsample': 0.8, 'seed': 1, 'eval_metric': 'map@1'}

        self.num_round = 2500
        self.model_path = model_path
        self.data_util = data_util

    def set_data_name(self, data_name):
        """

        :param data_name:
        :return:
        """
        self.data_name = data_name

    def filter_data(self, train_path):
        """
        Filtering groups that do not contain correct candidates
        :param train_path:
        :return:
        """
        group_id_dict = {}
        with open(train_path, "r", encoding="utf-8") as train_file:
            for item in train_file:
                item = item.strip()

                try:
                    group_str, label_str = item.split("\t")[:2]
                    group_id = int(group_str)
                    label = int(label_str)

                    if group_id in group_id_dict:
                        group_id_dict[group_id].add(label)
                    else:
                        label_set = set()
                        label_set.add(label)
                        group_id_dict[group_id] = label_set
                except:
                    print(item)

        group_id_set = set()
        for group_id, label_set in group_id_dict.items():
            if 1 in label_set:
                group_id_set.add(group_id)

        print("all group num: {0}".format(len(group_id_dict)))
        print("after filter group num: {0}".format(len(group_id_set)))

        return group_id_set

    def read_data(self, file_path, isFilter=False):
        """
        read data for ranking
        :param file_path:
        :param isFilter: is filtering groups that do not contain correct candidates
        :return:
        """
        y_list = []
        x_list = []
        group_len_dict = {}
        group_id_list = []
        group_id_set = set()

        save_group_set = set()
        if isFilter:
            save_group_set = self.filter_data(file_path)

        with open(file_path, "r", encoding="utf-8") as data_file:
            for item in data_file:
                item = item.strip()
                group_id_str, label_id_str, features_str, mention_str, entity_str = item.split("\t")
                group_id = int(group_id_str)

                if isFilter and (group_id not in save_group_set):
                    continue

                if group_id not in group_id_set:
                    group_id_list.append(group_id)
                    group_id_set.add(group_id)

                label = int(label_id_str)
                y_list.append(label)

                features_obj = json.loads(features_str)

                # if self.data_name == "ace2004":
                #     x_list.append([float(val) for name, val in features_obj.items()])

                x_list.append([float(val) for name, val in features_obj.items()])
                if group_id in group_len_dict:
                    group_len_dict[group_id] += 1
                else:
                    group_len_dict[group_id] = 1

        tmp_turple = sorted(group_len_dict.items(), key=lambda x:x[0])
        group_list = [ele[1] for ele in tmp_turple]

        print("length: {0}, {1}, {2}, {3}".format(len(y_list), len(x_list), sum(group_list), len(group_id_list)))

        return np.array(y_list), np.array(x_list), np.array(group_list), np.array(group_id_list)

    def train_model(self, train_path, dev_path):
        """
        train xgboost model
        :param file_path:
        :return:
        """
        # load train data
        y_train_array, x_train_array, group_train_array, group_id_array = self.read_data(train_path, isFilter=True)

        dtrain = xgb.DMatrix(x_train_array, label=y_train_array)
        dtrain.set_group(group_train_array)

        # load dev data
        y_dev_array, x_dev_array, group_dev_array, group_id_array = self.read_data(dev_path, isFilter=False)

        dDev = xgb.DMatrix(x_dev_array, label=y_dev_array)
        dDev.set_group(group_dev_array)

        evallist = [(dtrain, "train"), (dDev, "dev")]

        self.model = xgb.train(self.param, dtrain, num_boost_round=self.num_round, evals=evallist)
        self.model.save_model(self.model_path)
        # self.model.dump_model(self.model_path + '.dump')

        return self.model

    def load_rank_model(self, model_path=None):
        """
        加载保存的模型
        :param model_path:
        :return:
        """
        model_path = model_path if model_path else self.model_path
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        return self.model

    def compute_precision(self, y_list, pred_list, group_len_list, group_id_list):
        """
        calculate group precision
        :param y_list:
        :param pred_list:
        :param group_len_list:
        :param group_id_list:
        :return:
        """
        not_map1_group_dict = {}

        mention_num = len(group_len_list)
        correct = 0
        i = 0
        j = group_len_list[0]
        group_index = 0
        group_id = group_id_list[group_index]
        while 1:
            group_index += 1
            if group_index >= mention_num:
                break
            candidate_label_list = y_list[i:j]
            candidate_pred_list = pred_list[i:j]
            y_index_list = [c_index for c_index, c_label in enumerate(candidate_label_list) if c_label == 1]
            preds_index = candidate_pred_list.index(max(candidate_pred_list))
            if preds_index in set(y_index_list):
                correct += 1
            else:
                not_map1_group_dict[group_id] = preds_index

            i = j
            j = i + group_len_list[group_index]
            group_id = group_id_list[group_index]

        precison = float(correct) / mention_num

        return precison, not_map1_group_dict

    def not_map1_mention(self, file_path, not_map1_mention_path, not_map1_group_dict):
        """

        :param file_path:
        :param not_map1_mention_path:
        :param not_map1_group_dict:
        :return:
        """
        with open(file_path, "r", encoding="utf-8") as data_file:
            with open(not_map1_mention_path, "w", encoding="utf-8") as not_map1_file:
                for item in data_file:
                    item = item.strip()

                    group_id_str, label_id_str, features_str, mention_str, entity_str = item.split("\t")
                    mention_obj = json.loads(mention_str)
                    entity_obj = json.loads(entity_str)

                    mention_form = mention_obj["mention_form"]
                    entity_name = entity_obj["source_name"]

                    group_id = int(group_id_str)

                    if group_id in not_map1_group_dict:
                        not_map1_file.write("\t".join([group_id_str, label_id_str, str(not_map1_group_dict[group_id]), features_str, mention_form, entity_name]) + "\n")


    def predict(self, file_path, isFilter=True):
        """
        predict model
        :param file_path:
        :param isFilter:
        :return:
        """
        y_array, x_array, group_array, group_id_array = self.read_data(file_path, isFilter=isFilter)
        dtest = xgb.DMatrix(x_array, label=y_array)
        dtest.set_group(group_array)
        preds = self.model.predict(dtest)

        # calculate group precision
        precision, not_map1_group_dict = self.compute_precision(list(y_array), list(preds), list(group_array), list(group_id_array))

        # error log
        not_map1_mention_path = file_path + "_not_map1"
        self.not_map1_mention(file_path, not_map1_mention_path, not_map1_group_dict)

        print("group acc:{0}, recall num: {1}".format(precision, len(group_id_array) * precision))

    def cut_candidate(self, file_path, cut_candidate_path, max_num, is_train=False):
        """
        Filtering redundant candidate entities, and add xgboost's rank postion to entity object
        :param file_path:
        :param max_num:  number of candidate entities retained
        :param cut_candidate_path:
        :param is_train:
        :return:
        """
        y_array, x_array, group_array, group_id_array = self.read_data(file_path, isFilter=False)
        dtest = xgb.DMatrix(x_array, label=y_array)
        preds = self.model.predict(dtest)

        # saved list of candidate entities index
        save_index_list = []
        # saved list of candidate entities predict value
        save_pred_list = []
        # correct list of candidate entities index
        positive_index_list = []

        num = len(group_array)
        i = 0
        j = group_array[0]
        group_index = 0

        while 1:
            candidate_y_list = list(y_array[i:j])
            candidate_pred_list = list(preds[i:j])

            # correct candidate entity index
            y_index = candidate_y_list.index(max(candidate_y_list))
            if max(candidate_y_list) == 1:
                positive_index_list.append(i+y_index)

            # ranking candidate entities based on prediction results
            tmp_dict = {}
            for index, val in enumerate(candidate_pred_list):
                tmp_dict[index] = val
            rank_index_list = [item[0] for item in sorted(tmp_dict.items(), key=lambda x: x[1], reverse=True)]
            rank_pred_list = [item[1] for item in sorted(tmp_dict.items(), key=lambda x: x[1], reverse=True)]

            candidate_save_list = rank_index_list[:max_num]
            cut_pred_list = rank_pred_list[:max_num]

            # for train, add correct entity to candidate set
            if is_train and (y_index not in set(candidate_save_list)):
                # # error analyse
                # self.error_analyse(i + y_index, [i+ele for ele in candidate_save_list], x_array)
                candidate_save_list[-1] = y_index
                cut_pred_list[-1] = candidate_pred_list[y_index]

            save_index_list.append([i + index for index in candidate_save_list])
            save_pred_list.append(cut_pred_list)

            group_index += 1
            if group_index >= num:
                break

            i = j
            group_size = group_array[group_index]
            j = i + group_size

        # calculate the recall rate of candidate entities
        count = 0
        save_index_set = set()
        for index_list in save_index_list:
            for index in index_list:
                save_index_set.add(index)

        for positive_index in positive_index_list:
            if positive_index in save_index_set:
                count += 1
        print(count, len(set(list(group_id_array))), "recall is: {0}".format(float(count) / len(set(list(group_id_array)))))

        # filtered candidate entity files
        entity_all_list = []
        group_id_set = set(list(group_id_array))
        with open(file_path, "r", encoding="utf-8") as source_feature_file:
            with open(cut_candidate_path, "w", encoding="utf-8") as cut_candidate_file:
                for item in source_feature_file:
                    item = item.strip()
                    group_id = int(item.split("\t")[0])

                    if group_id not in group_id_set:
                        continue

                    entity_all_list.append(item)

                for index_list, pred_list in zip(save_index_list, save_pred_list):
                    for position, save_index in enumerate(index_list):
                        item = entity_all_list[save_index]

                        group_str, label_str, fea_str, mention_str, entity_str = item.split("\t")
                        entity_obj = json.loads(entity_str)
                        entity_obj["xgboost_rank_position"] = position
                        entity_obj["xgboost_pred"] = float(pred_list[position])

                        entity_str = json.dumps(entity_obj)

                        cut_candidate_file.write("\t".join([group_str, label_str, fea_str, mention_str, entity_str]) + "\n")

    def error_analyse(self, positive_index, index_list, x_array):
        """
        analyse golden entity which is not recalled by xgboost model
        :param positive_index:
        :param index_list:
        :param x_array:
        :return:
        """
        positive_candidate = x_array[positive_index]
        candidate_list = [x_array[index] for index in index_list]

        # print(positive_candidate)
        # print(candidate_list)
        # print("****************************" + "\n")

    def cal_recall(self, cut_candidate_path):
        """

        :param cut_candidate_path:
        :return:
        """
        group_id_dict = {}
        with open(cut_candidate_path, "r", encoding="utf-8") as cut_candidate_file:
            for item in cut_candidate_file:
                item = item.strip()

                group_str, label_str = item.split("\t")[:2]
                group_id = int(group_str)
                label = int(label_str)

                if group_id in group_id_dict:
                    group_id_dict[group_id].add(label)
                else:
                    label_set = set()
                    label_set.add(label)
                    group_id_dict[group_id] = label_set

        count = 0
        for group_id, label_set in group_id_dict.items():
            if len(label_set) > 1 or (len(label_set) == 1 and 1 in label_set):
                count += 1

        print(count, len(group_id_dict), "cut candidate recall is: {0}".format(count / len(group_id_dict)))

    def cul_fea_weight(self):
        """
        计算每个feature的权重信息
        :return:
        """
        model = self.load_rank_model(self.model_path)
        importance = model.get_fscore()

        val_list = []
        for fea, val in importance.items():
            val_list.append(val)

        fea_weight = [round(float(item)/sum(val_list), 3) for item in val_list]

        print(importance)

    def not_recall_analyse(self, rank_format_path, cut_rank_format_path):
        """
        analyse golden entity which is not recalled by xgboost model
        :param rank_format_path:
        :param cut_rank_format_path:
        :return:
        """
        all_group_dict = {}
        with open(cut_rank_format_path, "r", encoding="utf-8") as cut_rank_format_file:
            for item in cut_rank_format_file:
                item = item.strip()

                group_id_str, label_id_str, features_str, mention_str, entity_str = item.split("\t")
                group_id = int(group_id_str)
                label = int(label_id_str)

                if group_id not in all_group_dict:
                    all_group_dict[group_id] = [label]
                else:
                    all_group_dict[group_id].append(label)

        not_recall_group = set()
        for group_id, label_list in all_group_dict.items():
            if 1 not in label_list:
                print(group_id)
                not_recall_group.add(group_id)

        not_recall_path = cut_rank_format_path + "_not_recall"
        with open(rank_format_path, "r", encoding="utf-8") as rank_format_file:
            with open(not_recall_path, "w", encoding="utf-8") as not_recall_file:
                for item in rank_format_file:
                    item = item.strip()

                    group_id_str, label_id_str, features_str, mention_str, entity_str = item.split("\t")

                    group_id = int(group_id_str)

                    if group_id in not_recall_group:
                        not_recall_file.write(item + "\n")

    def controller_local_train(self):
        source_dir = "/data/fangzheng/rlel/"
        # train_path = source_dir + "aida_train/candidate/aida_train_rank_format"
        train_path = source_dir + "wiki_clueweb/candidate/wiki_clueweb_rank_format"
        dev_path = source_dir + "aida_testB/candidate/aida_testB_rank_format"

        self.set_data_name("wiki_clueweb")
        # self.train_model(train_path, dev_path)

        self.load_rank_model()
        self.predict(dev_path, isFilter=False)

        self.set_data_name("wiki_clueweb")
        cut_train_path = source_dir + "wiki_clueweb/candidate/wiki_clueweb_cut_rank_format"
        self.cut_candidate(train_path, cut_train_path, 5, is_train=True)

        # self.predict(cut_train_path, isFilter=True)

    def controller_local_test(self):
        source_dir = "/data/fangzheng/rlel/"
        # data_list = ["aida_testB", "aida_testA"]
        data_list = ["ace2004", "msnbc", "aquaint", "aida_testA", "aida_testB"]
        for data_name in data_list:
            test_path = source_dir + data_name + "/candidate/" + data_name + "_rank_format"
            cut_test_path = source_dir + data_name + "/candidate/" + data_name + "_cut_rank_format"
            print(data_name)
            self.set_data_name(data_name)

            self.load_rank_model()
            self.predict(test_path, isFilter=False)

            # self.cut_candidate(test_path, cut_test_path, 5, is_train=False)
            # print("mention num: {0}".format(self.data_util.get_mention_num(cut_test_path)))
            # self.predict(cut_test_path, isFilter=True)
            # self.not_recall_analyse(test_path, cut_test_path)

    def controller_global_train(self):
        source_dir = "/data/fangzheng/rlel/"
        # train_path = source_dir + "/aida_train/candidate/aida_train_cut_rank_format"
        dev_path = source_dir + "aida_testB/candidate/aida_testB_cut_rank_format"

        train_path = source_dir + "wiki_clueweb/candidate/wiki_clueweb_cut_rank_format"

        self.set_data_name("wiki_clueweb")
        self.train_model(train_path, dev_path)

        self.load_rank_model()
        self.predict(dev_path, isFilter=False)

        # self.set_data_name("aida_train")
        # cut_train_path = source_dir + "wiki_clueweb/candidate/wiki_clueweb_cut_rank_format"
        # self.cut_candidate(train_path, cut_train_path, 5, is_train=True)

    def controller_global_test(self):
        source_dir = "/data/fangzheng/rlel/"
        data_list = ["aida_testA", "aida_testB"]
        # data_list = ["ace2004", "msnbc", "aquaint", "clueweb", "wiki", "aida_testA", "aida_testB"]
        for data_name in data_list:
            test_path = source_dir + data_name + "/candidate/" + data_name + "_cut_rank_format"
            cut_test_path = source_dir + data_name + "/candidate/" + data_name + "_global_rank_format"
            print(data_name)
            self.set_data_name(data_name)

            # if data_name == "wiki" or data_name == "clueweb":
            #     test_path = source_dir + data_name + "/candidate/" + data_name + "_all_rank_format"

            self.load_rank_model()
            self.predict(test_path, isFilter=False)

            # self.cut_candidate(test_path, cut_test_path, 5, is_train=False)
            # print("mention num: {0}".format(self.data_util.get_mention_num(cut_test_path)))

if __name__ == "__main__":
    source_dir = "/data/fangzheng/rlel/"
    model_path = source_dir + "model/xgboost/aida_train_local"
    # model_path = source_dir + "/model/xgboost/aida_train_global"
    # model_path = source_dir + "model/xgboost/wiki_clueweb_local"
    # model_path = source_dir + "model/xgboost/wiki_clueweb_global"

    data_util = data_util.DataUtil()

    xgboost_rank = XgboostRank(model_path, data_util)

    xgboost_rank.controller_local_test()

