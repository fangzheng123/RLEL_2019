# encoding: utf-8

import json

def pred_acc(source_candidate_path, policy_pred_path):
    """

    :param source_candidate_path:
    :param policy_pred_path:
    :return:
    """
    pred_right_num = 0
    pred_result_dict = {}
    with open(policy_pred_path, "r", encoding="utf-8") as policy_pred_file:
        for item in policy_pred_file:
            item = item.strip()
            mention_index_str, label_str, fea_str, mention_str, entity_str = item.split("\t")
            mention_index = int(mention_index_str)
            entity_obj = json.loads(entity_str)
            entity_name = entity_obj["source_name"]

            pred_result_dict[mention_index] = entity_name
            if int(label_str) == 1:
                pred_right_num += 1

    with open(source_candidate_path, "r", encoding="utf-8") as source_candidate_file:
        for item in source_candidate_file:
            item = item.strip()
            mention_obj = json.loads(item)
            mention_index = mention_obj["mention_index"]

            # predict the first candidate
            if mention_index not in pred_result_dict:
                if "candidate" in mention_obj and len(mention_obj["candidate"]) > 0:
                    pred_result_dict[mention_index] = mention_obj["candidate"][0]["name"]
                else:
                    pred_result_dict[mention_index] = ""

                if pred_result_dict[mention_index] == mention_obj["target_name"]:
                    pred_right_num += 1

    print("predict right num:{0}, all mention num:{1}, acc:{2}".format(pred_right_num,
                                                                       len(pred_result_dict),
                                                                       pred_right_num / len(pred_result_dict)))

if __name__ == "__main__":
    data_list = ["aida_testB"]
    for data_name in data_list:
        print(data_name)
        source_candidate_path = "/home1/fangzheng/data/rlel_data/" + data_name + "/source/" + data_name + "_candidate"
        if data_name.__contains__("aida"):
            source_candidate_path = source_candidate_path + "_new_context"

        policy_pred_path = "/home1/fangzheng/data/rlel_data/" + data_name + "/policy/" + data_name + "_policy_pred"

        pred_acc(source_candidate_path, policy_pred_path)