# encoding:utf-8

import re
import sys
import time
import json
from datetime import timedelta

import numpy as np
from sklearn import preprocessing

import config_util

class DataUtil(object):
    """
    process data
    """

    def __init__(self):
        self.word_vocab_path = config_util.word_vocab_path
        self.all_word_embedding_path = config_util.word_embed_path

        self.entity_vocab_path = config_util.entity_vocab_path
        self.all_entity_embedding_path = config_util.entity_embed_path

    def get_time_dif(self, start_time):
        """
        get run time
        :param start_time: 起始时间
        :return:
        """
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

    def load_word_embed(self):
        """
        load all word embedding
        :return:
        """
        print("Loading word embedding...")
        start_time = time.time()

        word_dict = {}
        word_embed = np.load(self.all_word_embedding_path)

        with open(self.word_vocab_path, "r", encoding="utf-8") as word_vocab_file:
            count = 0
            for item in word_vocab_file:
                item = item.strip()
                word = item.split("\t")[0]
                word = word.lower()
                word_dict[word] = word_embed[count]

                count += 1

        run_time = self.get_time_dif(start_time)
        print("Time usage:{0}, Memory usage: {1} GB".format(run_time, int(sys.getsizeof(word_dict) / (1024 * 1024))))

        return word_dict

    def load_entity_embed(self):
        """
        load all entity embedding
        :return:
        """
        print("Loading entity embedding...")
        start_time = time.time()

        entity_embed = np.load(self.all_entity_embedding_path)

        entity_dict = {}
        with open(self.entity_vocab_path, "r", encoding="utf-8") as entity_url_file:
            count = 0
            for item in entity_url_file:
                item = item.strip()
                url = item.split("\t")[0]
                url = url.replace("en.wikipedia.org/wiki/", "")
                url = url.lower()
                entity_dict[url] = entity_embed[count]

                count += 1

        run_time = self.get_time_dif(start_time)
        print("Time usage:{0}, Memory usage: {1} GB".format(run_time, int(sys.getsizeof(entity_dict) / (1024 * 1024))))

        return entity_dict

    def remove_special_char(self, text):
        """
        remove special char from text
        :param text:
        :return:
        """
        special_char = u"[\n.,?!;:$*/'\\#\"\(\)\[\]\{\}\<\>]"
        text = re.sub(special_char, "", text)

        text = re.sub(u"-", " ", text)

        return text

    def remove_stop_word(self, text):
        """
        remove stop word from text
        :param text:
        :return:
        """
        stop_word_set = self.load_stop_words()

        new_text = [word for word in text.split(" ") if word.lower() not in stop_word_set and word != ""]

        return " ".join(new_text)

    def load_stop_words(self):
        """
        load english stop words
        :param stop_word_path:
        :return:
        """
        stop_word_set = set()
        with open(config_util.stop_word_path, "r", encoding="utf-8") as stop_word_file:
            for item in stop_word_file:
                item = item.strip().lower()
                stop_word_set.add(item)

        return stop_word_set

    def is_contain_keyword(self, name):
        """

        :param name:
        :return:
        """
        keyword_flag = False
        new_name = name.lower().replace(".", "")
        keyword_list = ["fc", "afc", "football", "soccer", "cricket", "rugby", "nba", "nhl"]
        not_keyword_list = ["system"]

        for word in keyword_list:
            if new_name.__contains__(word):
                keyword_flag = True
                break

        for word in not_keyword_list:
            if new_name.__contains__(word):
                keyword_flag = False
                break

        return keyword_flag

    def cos_distance(self, vector1, vector2):
        """
        余弦距离
        :param vector1:
        :param vector2:
        :return:
        """
        cos = 0.0
        vector1_norm = np.linalg.norm(vector1)
        vector2_norm = np.linalg.norm(vector2)
        if len(vector1) == len(vector2) and len(vector1) > 0 \
                and vector1_norm != 0 and vector2_norm != 0:
            cos = np.dot(vector1, vector2) / (vector1_norm * vector2_norm)

        return cos

    def cal_candidate_recall(self, data_path):
        """

        :param candidate_format_path:
        :param candidate_num:
        :return:
        """
        group_id_dict = {}
        group_mention_dict = {}
        with open(data_path, "r", encoding="utf-8") as fea_file:
            for item in fea_file:
                item = item.strip()

                group_str, label_str, fea_str, mention_str, entity_str = item.split("\t")
                group_id = int(group_str)
                label = int(label_str)
                mention = json.loads(mention_str)
                group_mention_dict[group_id] = mention

                if group_id in group_id_dict:
                    group_id_dict[group_id].append(label)
                else:
                    group_id_dict[group_id] = [label]

        recall_count = 0
        for group_id, label_list in group_id_dict.items():
            if 1 in set(label_list):
                recall_count += 1
            else:
                mention = group_mention_dict[group_id]
                print("not recall: {0}(mention form ), {1}(target_name ), {2}(mention_file)".format(mention["mention_form"], mention["target_name"], mention["mention_file"]))

        print("all count:{0}, recall count:{1}, recall:{2}".format(len(group_id_dict), recall_count, float(recall_count)/len(group_id_dict)))


    def get_mention_num(self, cut_candidate_path):
        """
        get mention num in cut candidate file
        :param cut_candidate_path:
        :return:
        """
        group_id_set = set()
        with open(cut_candidate_path, "r", encoding="utf-8") as cut_candidate_file:
            for item in cut_candidate_file:
                group_str, label_str, fea_str, mention_str, entity_str = item.split("\t")

                group_id = int(group_str)
                if group_id not in group_id_set:
                    group_id_set.add(group_id)

        return len(group_id_set)

    def get_group_list(self, data_path):
        """

        :param data_path:
        :return:
        """
        group_item_dict = {}
        with open(data_path, "r", encoding="utf-8") as data_file:
            for item in data_file:
                item = item.strip()
                group_str, label_str, fea_str, mention_str, entity_str = item.split("\t")

                group_id = int(group_str)
                if group_id not in group_item_dict:
                    group_item_dict[group_id] = [item]
                else:
                    group_item_dict[group_id].append(item)

        return group_item_dict

    def get_file_group(self, data_path):
        """

        :param data_path:
        :return:
        """
        file_group_dict = {}
        with open(data_path, "r", encoding="utf-8") as data_file:
            for item in data_file:
                item = item.strip()
                group_str, label_str, fea_str, mention_str, entity_str = item.split("\t")

                mention_obj = json.loads(mention_str)
                mention_file = mention_obj["mention_file"]
                group_id = int(group_str)

                if mention_file not in file_group_dict:
                    file_group_dict[mention_file] = [group_id]
                elif group_id not in file_group_dict[mention_file]:
                    file_group_dict[mention_file].append(group_id)

        return file_group_dict

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

if __name__ == "__main__":
    data_util = DataUtil()