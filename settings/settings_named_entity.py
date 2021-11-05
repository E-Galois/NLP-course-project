import torch
import os

# 超参数
hidden_dropout_prob = 0.3
num_labels = 9
learning_rate = 1e-5
weight_decay = 1e-2
epochs = 2
batch_size = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 文件路径
data_path = "./data"
model_path = "./models/model_chinese_bert_wwm_ext"
train_data = os.path.join(data_path, "train_data_public.csv")  # 训练数据集
# valid_data = os.path.join(data_path, "valid_data_public.csv")  # 验证数据集
test_data = os.path.join(data_path, "test_public.csv")  # 验证数据集
vocab_file = os.path.join(model_path, "vocab.txt")  # 词汇表
config_file = os.path.join(model_path, "bert_config.json")  # config
model_file = os.path.join(model_path, "pytorch_model.bin")  # model_chinese_bert_wwm_ext
entity_vocab_file = os.path.join(data_path, "entity_vocab.txt")
