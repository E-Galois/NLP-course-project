import os
import torch


class DefaultSettings:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = "./data"
    model_path = "./models/chinese-roberta-wwm-ext"
    train_data = os.path.join(data_path, "train_data_public.csv")  # 训练数据集
    # valid_data = os.path.join(data_path, "valid_data_public.csv")  # 验证数据集
    valid_data = train_data
    valid_sel_frac = 0.1
    test_data = os.path.join(data_path, "test_public.csv")  # 验证数据集
    vocab_file = os.path.join(model_path, "vocab.txt")  # 词汇表
    config_file = os.path.join(model_path, "bert_config.json")  # config
    model_file = os.path.join(model_path, "pytorch_model.bin")  # model

    hidden_dropout_prob = 0.3
    learning_rate = 1e-5
    weight_decay = 1e-2
    epochs = 5
    batch_size = 4
