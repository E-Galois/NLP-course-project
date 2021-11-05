from transformers import BertConfig, BertTokenizer, AdamW
import torch.nn as nn
# from settings.settings_sentiment import *


def load_model(
    bert_cls,
    vocab_file,
    config_file,
    num_labels,
    hidden_dropout_prob,
    model_file,
    device,
    weight_decay,
    learning_rate,
):
    # 加载模型
    #
    tokenizer = BertTokenizer.from_pretrained(vocab_file, strip_accents=False)
    config = BertConfig.from_pretrained(config_file, num_labels=num_labels, hidden_dropout_prob=hidden_dropout_prob)
    model = bert_cls.from_pretrained(model_file, config=config)
    model.to(device)

    # 定义优化器和损失函数
    # Prepare optimizer and schedule (linear warmup and decay)
    # 设置 bias 和 LayerNorm.weight 不使用 weight_decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # optimizer = AdamW(model_chinese_bert_wwm_ext.parameters(), lr=learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    return tokenizer, model, optimizer, criterion