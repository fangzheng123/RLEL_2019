# encoding: utf-8


class BaseConfig(object):

    # candidate num for each mention
    candidate_num = 5

    # the mention number for each sequence
    sequence_len = 5

    # class num
    class_num = 2

    # fea num
    fea_dim = 19

    # the dimension of word embedding and entity embedding
    word_embedd_dim = 300

    # output loss every n batch
    print_per_batch = 5

    # train
    train_path = "/home1/fangzheng/data/rlel_data/aida_train/candidate/aida_train_cut_rank_format"
    # local pred path
    train_local_pred_path = "/home1/fangzheng/data/rlel_data/aida_train/local/aida_train_local_predict"
    # rank candidate path
    train_local_candidate_rank_path = "/home1/fangzheng/data/rlel_data/aida_train/local/aida_train_local_rank_candidate"
    # rank mention path
    train_local_mention_rank_path = "/home1/fangzheng/data/rlel_data/aida_train/local/aida_train_local_rank_mention"
    # local representations of training data
    local_rep_train_path = "/home1/fangzheng/data/rlel_data/aida_train/local/aida_train_rank_mention_local_rep.npy"

    # validate
    val_name = "aida_testB"
    val_path = "/home1/fangzheng/data/rlel_data/" + val_name + "/candidate/" + val_name + "_global_rank_format"

    # test
    test_name = "aida_testB"
    test_path = "/home1/fangzheng/data/rlel_data/" + test_name + "/candidate/" + test_name + "_global_rank_format"
    # pred path
    test_local_pred_path = "/home1/fangzheng/data/rlel_data/" + test_name + "/local/" + test_name + "_local_pred"
    # rank candidate path
    test_local_candidate_rank_path = "/home1/fangzheng/data/rlel_data/" + test_name + "/local/" + test_name + "_local_rank_candidate"
    # rank mention path
    test_local_mention_rank_path = "/home1/fangzheng/data/rlel_data/" + test_name + "/local/" + test_name + "_local_rank_mention"
    # local representations of test data
    local_rep_test_path = "/home1/fangzheng/data/rlel_data/" + test_name + "/local/" + test_name + "_rank_mention_local_rep.npy"
    # policy reward path
    test_policy_reward_path = "/home1/fangzheng/data/rlel_data/" + test_name + "/policy/" + test_name + "_policy_reward.npy"
    # policy
    test_policy_pred_path = "/home1/fangzheng/data/rlel_data/" + test_name + "/policy/" + test_name + "_policy_pred"


class LocalConfig(BaseConfig):
    """
    本地模型配置类
    """
    def __init__(self):

        pass

    # batch大小
    local_batch_size = 64

    # the length of mention context and entity desc
    seq_length = 40

    # rnn隐藏层大小
    rnn_hidden_size = 512

    # rnn隐藏层层数
    rnn_num_layers = 2

    # 是否使用预训练词向量
    is_pre_train_embed = True

    # mention表示和entity表示的计算方式
    metric = "mlp"

    # 全连接层的层数
    mlp_metric_layer = 2

    # loss函数
    loss = "hinge"

    # 正负例的margin
    local_margin = 0.2

    # dropout保留比例
    dropout_keep_prob = 0.8

    # learning rate in local model
    local_learning_rate = 1e-3

    # epoch num
    local_num_epochs = 60

    # 提前结束训练间隔的轮数
    local_require_improvement = 3000

    # local model dir
    save_local_dir = "/home1/fangzheng/data/rlel_data/model/local_lstm/"



class GlobalConfig(BaseConfig):

    def __init__(self):
        pass

    local_config = LocalConfig()

    # the dimension of local embedding
    local_rep_dim = local_config.rnn_hidden_size

    # rnn隐藏层大小
    rnn_hidden_size = 756

    mlp_metric_layer = 3

    # rnn隐藏层层数
    rnn_num_layers = 2

    dropout_keep_prob = 0.8

    global_learning_rate = 1e-3

    # epoch num
    global_num_epochs = 150

    # 提前结束训练间隔的轮数
    local_require_improvement = 2000

    # batch size
    global_batch_size = 32

    # local model dir
    save_global_dir = "/home1/fangzheng/data/rlel_data/model/global_lstm/"


class PolicyConfig(BaseConfig):

    def __init__(self):
        pass

    local_config = LocalConfig()
    global_config = GlobalConfig()

    # the dimension of local embedding
    local_rep_dim = local_config.rnn_hidden_size

    # the dimension of global embedding
    global_rep_dim = global_config.rnn_hidden_size

    policy_mlp_layer = 2

    policy_batch_size = 32

    policy_learning_rate = 1e-3

    policy_num_epochs = 10

    policy_require_improvement = 1500

    # policy model dir
    save_policy_dir = "/home1/fangzheng/data/rlel_data/model/policy/"

