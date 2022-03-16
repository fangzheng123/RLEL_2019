# encoding: utf-8

import re
import json
from collections import Counter
from data_util import DataUtil

class GlobalFeature(object):

    def __init__(self, data_util):
        self.data_util = data_util

    def set_data_name(self, data_name):
        """

        :param data_name:
        :return:
        """
        self.data_name = data_name

    def get_doc_mention_obj(self, candidate_path):
        """
        get candidates of mentions which are in the same document
        :param candidate_path:
        :return:
        """
        doc_mention_dict = {}
        with open(candidate_path, "r", encoding="utf-8") as candidate_file:
            for item in candidate_file:
                item = item.strip()

                mention_obj = json.loads(item)
                mention_file = mention_obj["mention_file"]

                if mention_file not in doc_mention_dict:
                    doc_mention_dict[mention_file] = [mention_obj]
                else:
                    doc_mention_dict[mention_file].append(mention_obj)

        return doc_mention_dict

    def get_not_recall(self, not_recall_path):
        """

        :return:
        """
        not_recall_group_set = set()
        with open(not_recall_path, "r", encoding="utf-8") as not_recall_file:
            for item in not_recall_file:
                item = item.strip()
                group_id_str, label_id_str, features_str, mention_str, entity_str = item.split("\t")
                group_id = int(group_id_str)

                not_recall_group_set.add(group_id)

        return not_recall_group_set

    def global_recall_candidate(self, candidate_path, rank_format_path, not_recall_path, global_rank_format_path):
        """
        utilize global sim to recall candidate
        :param candidate_path:
        :param rank_format_path:
        :param not_recall_path:
        :param global_rank_format_path:
        :return:
        """
        doc_mention_dict = self.get_doc_mention_obj(candidate_path)

        print("start getting global candidate name")
        global_candidate_dict = {}
        with open(candidate_path, "r", encoding="utf-8") as candidate_file:
            for item in candidate_file:
                item = item.strip()

                mention_obj = json.loads(item)
                mention_file = mention_obj["mention_file"]
                current_mention_index = mention_obj["mention_index"]
                current_mention_form = mention_obj["mention_form"]
                mention_context = mention_obj["mention_left_context"] + " " + mention_obj["mention_right_context"]
                current_candidate_list = [ele["name"] for ele in mention_obj["candidate"]]

                doc_mention_list = doc_mention_dict[mention_file]

                for index, current_name in enumerate(current_candidate_list):
                    current_name_update = current_name.replace(",", "").replace("_", " ").lower()
                    current_name_update = self.data_util.remove_stop_word(current_name_update)

                    # mention is number
                    current_mention_update = current_mention_form.replace("\u2013", "")
                    if current_mention_update.isdigit():
                        mention_file_update = re.sub(u"[()]", "", mention_file).lower()

                        common_words = set(current_name_update.split(" ")) & set(mention_file_update.split("_"))
                        if current_mention_form in common_words:
                            continue

                        if len(common_words) > 1:
                            # print(" ## ".join([current_name_update, current_mention_form, mention_file_update]))

                            if current_mention_index not in global_candidate_dict:
                                global_candidate_dict[current_mention_index] = [current_name]
                            else:
                                global_candidate_dict[current_mention_index].append(current_name)

                        # has same word with mention context
                        context_common_words = set(current_name_update.split(" ")) & set(
                            mention_context.lower().split(" "))
                        context_common_words = [word for word in context_common_words if not word.isdigit()]
                        if len(context_common_words) > 1 and (len(current_mention_update) > 4):
                            # print(" ## ".join([current_name_update, current_mention_form, mention_file_update]))

                            if current_mention_index not in global_candidate_dict:
                                global_candidate_dict[current_mention_index] = [current_name]
                            else:
                                global_candidate_dict[current_mention_index].append(current_name)

                    for other_mention in doc_mention_list:
                        other_mention_index = other_mention["mention_index"]
                        other_mention_form = other_mention["mention_form"]
                        other_mention_candidate = other_mention["candidate"]

                        # for mention which has only 1 candidate
                        if other_mention_index != current_mention_index and \
                            other_mention_form.lower() != current_mention_form.lower() \
                            and len(other_mention_candidate) == 1:
                            other_candidate_name = other_mention_candidate[0]["name"]

                            other_name_update = other_candidate_name.replace(",", "").replace("_", " ").lower()
                            current_mention_form = current_mention_form.lower()

                            other_common_flag = True
                            common_words = set(current_name_update.split(" ")) & set(other_name_update.split(" "))
                            for word in common_words:
                                if word in current_mention_form.split(" "):
                                    other_common_flag = False
                                    break

                            if not other_common_flag:
                                continue

                            if len(common_words) > 0 and (
                                    (len(set(current_name_update.split(" ")) & set(current_mention_form.split(" "))) > 0)
                                    or (current_mention_update.isdigit() and len(current_mention_update) == 2)):

                                if current_mention_index not in global_candidate_dict:
                                    global_candidate_dict[current_mention_index] = [current_name]
                                else:
                                    global_candidate_dict[current_mention_index].append(current_name)

                        # has same word with other mention
                        if other_mention_index != current_mention_index \
                                and other_mention_form.lower() != current_mention_form.lower() \
                                and (other_mention_index < current_mention_index):

                            other_mention_form = other_mention_form.lower()
                            current_mention_form = current_mention_form.lower()

                            other_common_flag = True
                            common_words = set(current_name_update.split(" ")) & set(other_mention_form.split(" "))
                            for word in common_words:
                                if word in current_mention_form.split(" "):
                                    other_common_flag = False
                                    break

                            if current_name.lower().replace("_", " ") == other_mention_form:
                                other_common_flag = True

                            if not other_common_flag:
                                continue

                            if (len(set(current_name_update.split(" ")) & set(other_mention_form.split(" "))) > 0
                                    and len(set(current_name_update.split(" "))
                                            & set(current_mention_form.split(" "))) > 0
                                and len(current_name.split("_")) == len(current_mention_form.split(" "))+1) \
                                    or (current_name.lower().replace("_", " ") == other_mention_form):
                                # print(" ## ".join([current_name, other_mention_form, current_mention_form]))

                                if current_mention_index not in global_candidate_dict:
                                    global_candidate_dict[current_mention_index] = [current_name]
                                else:
                                    global_candidate_dict[current_mention_index].append(current_name)

        # add recall group
        recall_id_set = set()
        not_recall_set = self.get_not_recall(not_recall_path)

        print("get global candidate data")
        global_candidate_list = []
        count = 0
        with open(rank_format_path, "r", encoding="utf-8") as rank_file:
            for item in rank_file:
                item = item.strip()

                mention_index_str, label_str, fea_str, mention_str, entity_str = item.split("\t")

                mention_index = int(mention_index_str)

                if mention_index in global_candidate_dict:
                    entity_json = json.loads(entity_str)
                    entity_name = entity_json["source_name"]

                    if int(label_str) == 1 and entity_name in set(global_candidate_dict[mention_index]):
                        count += int(label_str)

                        if mention_index in not_recall_set:
                            recall_id_set.add(mention_index)

                    if entity_name in set(global_candidate_dict[mention_index]):
                        global_candidate_list.append(item)

        for id in sorted(list(not_recall_set)):
            if id not in recall_id_set:
                print(id)
        print("add recall num:{0}, all not recall num:{1}".format(len(recall_id_set), len(not_recall_set)))

        print("right label:{0}, global candidate num: {1}".format(count, len(global_candidate_list)))

        with open(global_rank_format_path, "w", encoding="utf-8") as global_rank_file:
            for item in global_candidate_list:
                global_rank_file.write(item + "\n")
                global_rank_file.flush()

    def add_global_candidate(self, global_rank_format_path, cut_rank_format_path, all_rank_format_path):
        """

        :param global_rank_format_path:
        :param cut_rank_format_path:
        :return:
        """

        global_rank_dict = {}
        with open(global_rank_format_path, "r", encoding="utf-8") as global_rank_file:
            for item in global_rank_file:
                item = item.strip()
                mention_index_str, label_str, fea_str, mention_str, entity_str = item.split("\t")
                menntion_index = int(mention_index_str)

                if menntion_index not in global_rank_dict:
                    global_rank_dict[menntion_index] = [item]
                else:
                    global_rank_dict[menntion_index].append(item)

        cut_rank_dict = {}
        with open(cut_rank_format_path, "r", encoding="utf-8") as cut_rank_file:
            for item in cut_rank_file:
                item = item.strip()
                mention_index_str, label_str, fea_str, mention_str, entity_str = item.split("\t")

                mention_index = int(mention_index_str)

                if mention_index not in cut_rank_dict:
                    cut_rank_dict[mention_index] = [item]
                else:
                    cut_rank_dict[mention_index].append(item)

        count = 0
        new_item_list = []
        for mention_index, item_list in cut_rank_dict.items():
            new_item_list.extend(item_list)

            if mention_index in global_rank_dict:
                global_item_list = global_rank_dict[mention_index]

                cut_ele_dict = {}
                for ele in item_list:
                    cut_ele_dict[json.loads(ele.split("\t")[-1])["source_name"]] = ele

                for ele in global_item_list:
                    global_name = json.loads(ele.split("\t")[-1])["source_name"]
                    if global_name not in cut_ele_dict:
                        new_item_list.append(ele)
                        count += 1

        print("add count: {0}".format(count))

        with open(all_rank_format_path, "w", encoding="utf-8") as all_rank_file:
            for item in new_item_list:
                all_rank_file.write(item + "\n")

    def get_mention_candidate(self, all_rank_format_path):
        """

        :param all_rank_format_path:
        :return:
        """
        doc_mention_dict = {}
        mention_detail_dict = {}
        mention_candidate_dict = {}
        with open(all_rank_format_path, "r", encoding="utf-8") as all_rank_format_file:
            for item in all_rank_format_file:
                item = item.strip()

                mention_index_str, label_str, fea_str, mention_str, entity_str = item.split("\t")
                entity = json.loads(entity_str)
                mention_obj = json.loads(mention_str)
                mention_file = mention_obj["mention_file"]

                mention_index = int(mention_index_str)

                mention_detail_dict[mention_index] = mention_obj

                if mention_index not in mention_candidate_dict:
                    mention_candidate_dict[mention_index] = [entity]
                else:
                    mention_candidate_dict[mention_index].append(entity)

                if mention_file not in doc_mention_dict:
                    doc_mention_dict[mention_file] = [mention_index]
                else:
                    doc_mention_dict[mention_file].append(mention_index)

        return doc_mention_dict, mention_detail_dict, mention_candidate_dict

    def process_candidate_name(self, name):
        """

        :param name:
        :return:
        """
        name = name.lower().replace("(", "").replace(")", "").replace(".", "").replace(",", "")
        return name

    def process_candidate_category(self, category):
        """

        :param category:
        :return:
        """
        new_category = []
        for word in category:
            new_word = word.replace("|", " ").strip().lower()
            new_category.append(new_word)

        return new_category

    def cal_global_fea(self, mention_obj, entity, doc_mention_dict, mention_detail_dict, mention_candidate_dict):
        """
        calculate global feature between candidates
        :param mention_obj:
        :param entity:
        :param doc_mention_dict:
        :param mention_detail_dict:
        :param mention_candidate_dict:
        :return:
        """
        # 这些规则也可以作为本地排序前, 过滤候选的规则
        # 是否和只有一个候选实体的候选有相同词

        same_candidate_word_num = 0
        same_mention_word_num = 0
        has_frequency_word = 0
        same_word_yago_top = 0
        second_rank_yago = 0
        has_keyword = 0

        entity_name = entity["source_name"]
        current_mention_index = int(mention_obj["mention_index"])
        mention_file = mention_obj["mention_file"]
        current_mention_form = mention_obj["mention_form"]
        current_candidate_index = 0
        for index, ele in enumerate(mention_candidate_dict[current_mention_index]):
            if entity_name == ele["source_name"]:
                current_candidate_index = index

        doc_mention_list = doc_mention_dict[mention_file]
        adj_candidate_list = []
        adj_mention_list = []
        for other_mention_index in doc_mention_list:
            other_mention = mention_detail_dict[other_mention_index]

            if other_mention_index != current_mention_index \
                    and other_mention["mention_form"].lower() != current_mention_form.lower():

                if abs(other_mention_index - current_mention_index) < 5:
                    other_candidate_list = [self.process_candidate_name(ele["source_name"].replace(",", "").lower())
                                            for ele in mention_candidate_dict[other_mention_index]]

                    if len(mention_candidate_dict[other_mention_index]) == 1:
                        for i in range(3):
                            adj_candidate_list.extend(other_candidate_list)

                    else:
                        adj_candidate_list.extend(other_candidate_list)

                adj_mention_list.append(other_mention["mention_form"].lower())

        other_candidate_word_list = []
        other_candidate2_word_list = []
        other_candidate3_word_list = []
        for name in adj_candidate_list:
            other_candidate_word_list.extend(name.split("_"))
            other_candidate2_word_list.extend(name.split("_"))
            other_candidate3_word_list.extend(name.split("_"))

        other_mention_word_list = []
        for name in adj_mention_list:
            other_mention_word_list.extend(name.split(" "))

        other_candidate_word_counter = Counter(other_candidate_word_list)
        other_mention_word_counter = Counter(other_mention_word_list)

        sort_candidate_word_list = [pair[0] for pair in sorted(other_candidate_word_counter.items(), key=lambda x: x[1], reverse=True)]

        # print("#####".join(sort_candidate_word_list[:3]))

        if ("mention_context" in mention_obj and mention_obj["mention_context"] != "") \
                or ("mention_context" not in mention_obj):
            clean_entity_name = self.data_util.remove_stop_word(self.process_candidate_name(entity_name.lower().replace("_", " ")))
            for word in clean_entity_name.split(" "):
                # the same word num between current candidate and other mention's candidate
                if word in other_candidate_word_counter:
                    same_candidate_word_num += other_candidate_word_counter[word]

                # the same word num between current candidate and other mention's surface form
                if word in other_mention_word_counter:
                    same_mention_word_num += other_mention_word_counter[word]

                # has high frequency candidate word
                if word in sort_candidate_word_list[:3]:
                    has_frequency_word += 1

        # yago feature
        yago_list = [ele["yago_score"] for ele in mention_candidate_dict[current_mention_index]]
        if (entity["yago_score"] == max(yago_list) or entity["yago_score"] == max(yago_list[:2])) \
                and same_candidate_word_num > 8:
            same_word_yago_top = 1
        if entity["yago_score"] == max(yago_list[:2]) and entity["yago_score"] > 10*yago_list[0]:
            second_rank_yago = 1

        if self.data_name.__contains__("aida"):
            keyword = mention_obj["mention_context"].split(" ")[0].lower().replace("soccer", "football")
            if current_candidate_index < 2 and entity_name.__contains__(keyword):
                has_keyword = 1

        return same_candidate_word_num, same_mention_word_num, has_frequency_word, same_word_yago_top, second_rank_yago, has_keyword

    def add_global_fea(self, all_rank_format_path):
        """
        add global feature
        :param all_rank_format_path:
        :return:
        """
        doc_mention_dict, mention_detail_dict, mention_candidate_dict = self.get_mention_candidate(all_rank_format_path)

        global_item_list = []
        no_global_set = set()
        group_keyword_set = set()
        with open(all_rank_format_path, "r", encoding="utf-8") as all_rank_format_file:
            for item in all_rank_format_file:
                item = item.strip()

                mention_index_str, label_str, fea_str, mention_str, entity_str = item.split("\t")
                mention_obj = json.loads(mention_str)
                entity = json.loads(entity_str)
                mention_index = int(mention_index_str)

                same_candidate_word_num, same_mention_word_num, has_frequency_word, \
                same_word_yago_top, second_rank_yago, has_keyword = self.cal_global_fea(mention_obj, entity, doc_mention_dict, mention_detail_dict, mention_candidate_dict)

                fea_obj = json.loads(fea_str)

                if "has_keyword" in fea_obj:
                    del fea_obj["has_keyword"]

                fea_obj["same_candidate_word_num"] = same_candidate_word_num
                fea_obj["same_mention_word_num"] = same_mention_word_num
                fea_obj["has_frequency_word"] = has_frequency_word
                fea_obj["same_word_yago_top"] = same_word_yago_top
                fea_obj["second_rank_yago"] = second_rank_yago
                fea_obj["has_keyword"] = has_keyword

                global_item_list.append("\t".join([mention_index_str, label_str, json.dumps(fea_obj), mention_str, entity_str]))

        with open(all_rank_format_path, "w", encoding="utf-8") as all_rank_format_file:
            for item in global_item_list:
                all_rank_format_file.write(item + "\n")
                all_rank_format_file.flush()

    def add_seq_fea(self, rank_mention_path, seq_len):
        """
        add seq feature
        :param rank_mention_path:
        :param seq_len:
        :return:
        """
        group_item_dict = self.data_util.get_group_list(rank_mention_path)
        file_group_dict = self.data_util.get_file_group(rank_mention_path)

        group_fea_dict = {}
        for mention_file, group_id_list in file_group_dict.items():
            batch_num = int((len(group_id_list) - 1) / seq_len) + 1
            for i in range(batch_num):
                start_id = i * seq_len
                end_id = min((i + 1) * seq_len, len(group_id_list))
                seq_group_list = group_id_list[start_id:end_id]

                name_keyword_list = []
                category_keyword_list = []

                # the first mention in sequence
                seq_first_group_id = seq_group_list[0]
                item_list = group_item_dict[seq_first_group_id]
                group_id_str, label_str, fea_str, mention_str, entity_str = item_list[0].split("\t")
                entity_obj = json.loads(entity_str)
                entity_name = self.process_candidate_name(entity_obj["source_name"].replace(",", "").lower())
                name_keyword_list.extend(entity_name.split("_"))
                if "category" in entity_obj:
                    entity_category = self.process_candidate_category(entity_obj["category"])
                    category_keyword_list.extend(" ".join(entity_category).split(" "))

                for index, group_id in enumerate(seq_group_list[1:]):
                    item_list = group_item_dict[group_id]

                    entity_name_counter = Counter(name_keyword_list)
                    entity_category_counter = Counter(category_keyword_list)

                    for item in item_list:
                        same_name_word_num = 0
                        same_category_word_num = 0

                        group_id_str, label_str, fea_str, mention_str, entity_str = item.split("\t")
                        entity_obj = json.loads(entity_str)
                        entity_name = self.process_candidate_name(entity_obj["source_name"].replace(",", "").lower())
                        same_name_word_num = sum([entity_name_counter[word] for word in entity_name.split("_") if word in entity_name_counter])

                        if "category" in entity_obj:
                            entity_category = self.process_candidate_category(entity_obj["category"])
                            category_word_list = " ".join(entity_category).split(" ")
                            same_category_word_num = sum([entity_category_counter[word] for word in category_word_list if
                                 word in entity_category_counter and word != ""])

                        fea_obj = json.loads(fea_str)
                        fea_obj["seq_name_word_num"] = same_name_word_num
                        fea_obj["seq_category_word_num"] = same_category_word_num

                        if group_id not in group_fea_dict:
                            group_fea_dict[group_id] = [fea_obj]
                        else:
                            group_fea_dict[group_id].append(fea_obj)

                for item in group_item_dict[seq_first_group_id]:
                    fea_str = item.strip().split("\t")[2]
                    fea_obj = json.loads(fea_str)
                    fea_obj["seq_name_word_num"] = 0
                    fea_obj["seq_category_word_num"] = 0

                    if seq_first_group_id not in group_fea_dict:
                        if len(seq_group_list) > 1:
                            # the first mention is replaced by the second
                            fea_obj["seq_name_word_num"] = group_fea_dict[seq_group_list[1]][0]["seq_name_word_num"]
                            fea_obj["seq_category_word_num"] = group_fea_dict[seq_group_list[1]][0]["seq_category_word_num"]
                            group_fea_dict[seq_first_group_id] = [fea_obj]
                        else:
                            group_fea_dict[seq_first_group_id] = [fea_obj]
                    else:
                        group_fea_dict[seq_first_group_id].append(fea_obj)

        seq_fea_path = rank_mention_path + "_seq_fea"
        with open(seq_fea_path, "w", encoding="utf-8") as seq_fea_file:
            for mention_file, group_id_list in file_group_dict.items():
                for group_id in group_id_list:
                    item_list = group_item_dict[group_id]
                    add_fea_list = group_fea_dict[group_id]

                    for index, item in enumerate(item_list):
                        item = item.strip()
                        group_id_str, label_str, fea_str, mention_str, entity_str = item.split("\t")

                        seq_fea_file.write("\t".join([group_id_str, label_str, json.dumps(add_fea_list[index]), mention_str, entity_str]) + "\n")
                        seq_fea_file.flush()

    def test(self, rank_path):
        """

        :param rank_path:
        :return:
        """
        group_item_dict = {}
        keyword_set = set()
        with open(rank_path, "r", encoding="utf-8") as all_rank_format_file:
            for item in all_rank_format_file:
                item = item.strip()

                mention_index_str, label_str, fea_str, mention_str, entity_str = item.split("\t")
                mention_index = int(mention_index_str)
                fea_obj = json.loads(fea_str)

                if fea_obj["has_keyword"] == 1:
                    keyword_set.add(mention_index)

                if mention_index not in group_item_dict:
                    group_item_dict[mention_index] = [item]
                else:
                    group_item_dict[mention_index].append(item)

        with open(rank_path, "w", encoding="utf-8") as all_rank_format_file:
            for group_id, item_list in group_item_dict.items():
                for item in item_list:
                    if group_id in keyword_set:
                        item = item.strip()
                        mention_index_str, label_str, fea_str, mention_str, entity_str = item.split("\t")

                        fea_obj = json.loads(fea_str)
                        fea_obj["pageview"] = 0
                        fea_obj["name_ratio"] = 0

                        all_rank_format_file.write("\t".join([mention_index_str, label_str, json.dumps(fea_obj), mention_str, entity_str]) + "\n")
                    else:
                        all_rank_format_file.write(item + "\n")

if __name__ == "__main__":
    data_util = DataUtil()
    global_fea = GlobalFeature(data_util)

    source_dir = "/data/fangzheng/rlel/"
    data_list = ["aida_testB"]
    # data_list = ["ace2004", "msnbc", "aquaint", "clueweb", "wiki", "aida_train", "aida_testA", "aida_testB"]
    for data_name in data_list:
        print(data_name)

        global_fea.set_data_name(data_name)
        candidate_path = source_dir + data_name + "/source/" + data_name + "_candidate"
        rank_format_path = source_dir + data_name + "/candidate/" + data_name + "_rank_format"
        global_rank_format_path = source_dir + data_name + "/candidate/" + data_name + "_rank_global_format"
        cut_rank_format_path = source_dir + data_name + "/candidate/" + data_name + "_cut_rank_format"
        not_recall_path = cut_rank_format_path + "_not_recall"
        all_rank_path = source_dir + data_name + "/candidate/" + data_name + "_all_rank_format"
        test_local_mention_rank_path = source_dir + data_name + "/local/" + data_name + "_local_rank_mention"

        if data_name == "wiki" or data_name == "clueweb":
            global_fea.global_recall_candidate(candidate_path, rank_format_path, not_recall_path, global_rank_format_path)
            global_fea.add_global_candidate(global_rank_format_path, cut_rank_format_path, all_rank_path)
            global_fea.add_global_fea(all_rank_path)
        else:
            global_fea.add_global_fea(cut_rank_format_path)

        global_fea.add_seq_fea(test_local_mention_rank_path, 6)

        # data_util.cal_candidate_recall(all_rank_path)
