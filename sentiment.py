from tqdm import tqdm
import json
import time
from settings.settings_sentiment import *
from data_process import get_dataloaders, SentimentDataset
from model_process import load_model
from transformers import BertForSequenceClassification


# 定义训练的函数
def train(tokenizer, model, dataloader, optimizer, criterion, device):
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
        loss = criterion(y_pred_prob.view(-1, 3), label.view(-1))

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
    epoch_loss = 0
    epoch_acc = 0
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


def train_and_test():
    sentiment_train_loader, test_loader = get_dataloaders(SentimentDataset, train_data, None, test_data, batch_size)
    tokenizer, model, optimizer, criterion = load_model(
        BertForSequenceClassification,
        vocab_file,
        config_file,
        num_labels,
        hidden_dropout_prob,
        model_file,
        device,
        weight_decay,
        learning_rate
    )
    # 开始训练和验证
    for i in range(epochs):
        train_loss, train_acc = train(tokenizer, model, sentiment_train_loader, optimizer, criterion, device)
        print("train loss: ", train_loss, "\t", "train acc:", train_acc)
        '''
        valid_loss, valid_acc = evaluate(tokenizer, model_chinese_bert_wwm_ext, sentiment_valid_loader, device)
        print("valid loss: ", valid_loss, "\t", "valid acc:", valid_acc)
        '''

    result = test(tokenizer, model, test_loader, device)
    with open(f'senti_res-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}.json', 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    train_and_test()
