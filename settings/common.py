import os
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_path = "./data"
model_path = "./models/model_chinese_bert_wwm_ext"
train_data = os.path.join(data_path, "train_data_public.csv")  # 训练数据集
# valid_data = os.path.join(data_path, "valid_data_public.csv")  # 验证数据集
test_data = os.path.join(data_path, "test_public.csv")  # 验证数据集
vocab_file = os.path.join(model_path, "vocab.txt")  # 词汇表
config_file = os.path.join(model_path, "bert_config.json")  # config
model_file = os.path.join(model_path, "pytorch_model.bin")  # model
