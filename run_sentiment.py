import os
from tqdm import tqdm
import json
import time
import torch
from settings.settings_sentiment import SentimentSettings
from data_process import get_dataloaders, SentimentDataset
from model_process import load_model
# from transformers import BertForSequenceClassification
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import get_linear_schedule_with_warmup


# 定义训练的函数
def train(tokenizer, model, dataloader, optimizer, criterion=None, device='cuda', scheduler=None):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for i, batch in tqdm(enumerate(dataloader)):
        # 标签形状为 (batch_size, 1)
        label = batch["label"]
        text = batch["text"]

        # tokenized_text 包括 input_ids， token_type_ids， attention_mask
        tokenized_text = tokenizer(text, max_length=100, add_special_tokens=True, truncation=True, padding=True, return_tensors="pt")
        tokenized_text = tokenized_text.to(device)
        label = label.to(device)

        # 梯度清零
        optimizer.zero_grad()

        #output: (loss), logits, (hidden_states), (attentions)
        output = model(**tokenized_text, labels=label)

        # y_pred_prob = logits : [batch_size, num_labels]
        y_pred_prob = output['logits']
        y_pred_label = y_pred_prob.argmax(dim=1)

        # 计算loss
        # 这个 loss 和 output[0] 是一样的
        # loss = criterion(y_pred_prob.view(-1, 3), label.view(-1))
        loss = output['loss']

        # 计算acc
        acc = ((y_pred_label == label.view(-1)).sum()).item()

        # 反向传播
        loss.backward()
        optimizer.step()

        # epoch 中的 loss 和 acc 累加
        # loss 每次是一个 batch 的平均 loss
        epoch_loss += loss.item()
        # acc 是一个 batch 的 acc 总和
        epoch_acc += acc
        if i % 200 == 0:
            print("current loss:", epoch_loss / (i+1), "\t", "current acc:", epoch_acc / ((i+1)*len(label)))

    # len(dataloader) 表示有多少个 batch，len(dataloader.dataset.dataset) 表示样本数量
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader.dataset.dataset)


def evaluate(tokenizer, model, iterator, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for _, batch in enumerate(iterator):
            label = batch["label"]
            text = batch["text"]
            tokenized_text = tokenizer(text, max_length=100, add_special_tokens=True, truncation=True, padding=True,
                                       return_tensors="pt")
            tokenized_text = tokenized_text.to(device)
            label = label.to(device)

            output = model(**tokenized_text, labels=label)
            y_pred_label = output['logits'].argmax(dim=1)
            loss = output['loss']
            acc = ((y_pred_label == label.view(-1)).sum()).item()
            # epoch 中的 loss 和 acc 累加
            # loss 每次是一个 batch 的平均 loss
            epoch_loss += loss.item()
            # acc 是一个 batch 的 acc 总和
            epoch_acc += acc

    # len(dataloader) 表示有多少个 batch，len(dataloader.dataset.dataset) 表示样本数量
    return epoch_loss / len(iterator), epoch_acc / len(iterator.dataset.dataset)


def test(tokenizer, model, iterator, device):
    model.eval()
    ids = []
    classes = []
    with torch.no_grad():
        for _, batch in enumerate(iterator):
            _id = batch["id"].numpy().tolist()
            text = batch["text"]
            tokenized_text = tokenizer(text, max_length=100, add_special_tokens=True, truncation=True, padding=True,
                                       return_tensors="pt")
            tokenized_text = tokenized_text.to(device)

            output = model(**tokenized_text, labels=None)
            y_pred_label = output['logits'].argmax(dim=1).cpu().numpy().tolist()

            ids.extend(_id)
            classes.extend(y_pred_label)
    return dict(zip(ids, classes))


def train_and_test(CFG, global_timestamp, search_node_name):
    sentiment_train_loader, test_loader, valid_loader = get_dataloaders(
        SentimentDataset,
        CFG.train_data,
        CFG.valid_data,
        CFG.test_data,
        batch_size=CFG.batch_size,
        valid_sel_frac=0.1
    )
    tokenizer, model, optimizer, criterion = load_model(
        BertForSequenceClassification,
        BertConfig,
        BertTokenizer,
        CFG.vocab_file,
        CFG.config_file,
        CFG.num_labels,
        CFG.hidden_dropout_prob,
        CFG.model_file,
        CFG.device,
        CFG.weight_decay,
        CFG.learning_rate,
        # load_from_dir=True,
        # load_dir=CFG.model_path
    )
    warm_up_ratio = 0.1  # 定义要预热的step
    total_steps = len(sentiment_train_loader) * CFG.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_ratio * total_steps,
                                                num_training_steps=total_steps)
    # 开始训练和验证
    for i in range(CFG.epochs):
        train_loss, train_acc = train(tokenizer, model, sentiment_train_loader, optimizer, device=CFG.device,
                                      scheduler=scheduler)
        print("train loss: ", train_loss, "\t", "train acc:", train_acc)

        valid_loss, valid_acc = evaluate(tokenizer, model, valid_loader, CFG.device)
        print("valid loss: ", valid_loss, "\t", "valid acc:", valid_acc)

        result = test(tokenizer, model, test_loader, CFG.device)

        epoch_node_name = f'epoch-{i}_vacc-{valid_acc}'
        result_dir = f'./results/sentiment/{global_timestamp}/{search_node_name}/{epoch_node_name}'
        os.mkdir(result_dir)
        with open(os.path.join(result_dir, f'/senti_res.json'), 'w') as f:
            json.dump(result, f)


if __name__ == '__main__':
    global_timestamp = f'{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}'
    os.mkdir(f'./results/sentiment/{global_timestamp}')
    CFG = SentimentSettings()
    for batch_size in [1, 2, 3, 4, 6, 8]:
        for lr in [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4]:
            CFG.batch_size = batch_size
            CFG.learning_rate = lr
            search_node_name = f'bs-{CFG.batch_size}_lr-{CFG.learning_rate}'
            os.mkdir(f'./results/sentiment/{global_timestamp}/{search_node_name}')
            train_and_test(CFG, global_timestamp, search_node_name)
