from transformers import AdamW
import torch.nn as nn
# from settings.settings_sentiment import *


def load_model(
    model_cls,
    cfg_cls,
    tokenizer_cls,
    vocab_file=None,
    config_file=None,
    num_labels=None,
    hidden_dropout_prob=None,
    model_file=None,
    device=None,
    weight_decay=None,
    learning_rate=None,
    load_from_dir=False,
    load_dir=None
):
    # 加载模型
    if load_from_dir:
        tokenizer = tokenizer_cls.from_pretrained(load_dir, strip_accents=False)
        config = cfg_cls.from_pretrained(load_dir, num_labels=num_labels, hidden_dropout_prob=hidden_dropout_prob)
        model = model_cls.from_pretrained(load_dir, config=config)
        model.to(device)
    else:
        tokenizer = tokenizer_cls.from_pretrained(vocab_file, strip_accents=False)
        config = cfg_cls.from_pretrained(config_file, num_labels=num_labels, hidden_dropout_prob=hidden_dropout_prob)
        model = model_cls.from_pretrained(model_file, config=config)
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