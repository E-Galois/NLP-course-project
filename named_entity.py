from tqdm import tqdm
import json
import time
from settings.settings_named_entity import *
from data_process import get_dataloaders, NamedEntityDataset, clean_split
from model_process import load_model
from transformers import BertTokenizer, BertForTokenClassification


def get_entity_tokenizer():
    entity_tokenizer = BertTokenizer.from_pretrained(entity_vocab_file)
    return entity_tokenizer


# 定义训练的函数
def train(entity_tokenizer, tokenizer, model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    epoch_total_positions = 0
    for i, batch in tqdm(enumerate(dataloader)):
        # 标签形状为 (batch_size, 1)
        text = batch["text"]
        label = batch["label"]
        text = [clean_split(one) for one in text]
        label = [one.replace('-', '').replace('_', '') for one in label]
        # label = [entity_tokenizer.convert_tokens_to_ids(one.split(' ')) for one in label]

        # tokenized_text 包括 input_ids， token_type_ids， attention_mask
        tokenized_text = tokenizer(text, max_length=100, add_special_tokens=False, truncation=True, padding=True, return_tensors="pt", is_split_into_words=True)
        tokenized_text = tokenized_text.to(device)
        tokenized_label = entity_tokenizer(label, max_length=100, add_special_tokens=False, truncation=True, padding=True, return_tensors="pt")
        tokenized_label = tokenized_label['input_ids'].to(device)

        # 梯度清零
        optimizer.zero_grad()

        #output: (loss), logits, (hidden_states), (attentions)
        output = model(**tokenized_text, labels=tokenized_label)

        # y_pred_prob = logits : [batch_size, num_labels]
        y_pred_prob = output['logits']
        y_pred_label = y_pred_prob.argmax(dim=2)

        # 计算loss
        # 这个 loss 和 output[0] 是一样的
        # loss = criterion(y_pred_prob.view(-1, 3), tokenized_label.view(-1))
        loss = output['loss']

        # 计算acc
        mapped_positions = (y_pred_label == tokenized_label) * tokenized_text['attention_mask']
        acc = mapped_positions.sum().item()

        # 反向传播
        loss.backward()
        optimizer.step()

        # epoch 中的 loss 和 acc 累加
        # loss 每次是一个 batch 的平均 loss
        epoch_loss += loss.item()
        # acc 是一个 batch 的 acc 总和
        epoch_acc += acc
        epoch_total_positions += tokenized_text['attention_mask'].sum().item()
        if i % 200 == 0:
            print("current loss:", epoch_loss / (i+1), "\t", "current acc:", epoch_acc / epoch_total_positions)
    return epoch_loss / len(dataloader), epoch_acc / epoch_total_positions


def evaluate(entity_tokenizer, tokenizer, model, iterator, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    epoch_total_positions = 0
    with torch.no_grad():
        for _, batch in tqdm(enumerate(iterator)):
            text = batch["text"]
            label = batch["label"]
            text = [clean_split(one) for one in text]
            label = [one.replace('-', '').replace('_', '') for one in label]
            # label = [entity_tokenizer.convert_tokens_to_ids(one.split(' ')) for one in label]

            # tokenized_text 包括 input_ids， token_type_ids， attention_mask
            tokenized_text = tokenizer(text, max_length=100, add_special_tokens=False, truncation=True, padding=True, return_tensors="pt", is_split_into_words=True)
            tokenized_text = tokenized_text.to(device)
            tokenized_label = entity_tokenizer(label, max_length=100, add_special_tokens=False, truncation=True,
                                               padding=True, return_tensors="pt")
            tokenized_label = tokenized_label['input_ids'].to(device)

            output = model(**tokenized_text, labels=tokenized_label)
            y_pred_label = output['logits'].argmax(dim=1)
            loss = output['loss']
            # 计算acc
            mapped_positions = (y_pred_label == tokenized_label) * tokenized_text['attention_mask']
            acc = mapped_positions.sum().item()
            # epoch 中的 loss 和 acc 累加
            # loss 每次是一个 batch 的平均 loss
            epoch_loss += loss.item()
            # acc 是一个 batch 的 acc 总和
            epoch_acc += acc
            epoch_total_positions += tokenized_text['attention_mask'].sum().item()
    return epoch_loss / len(iterator), epoch_acc / epoch_total_positions


def test(entity_tokenizer, tokenizer, model, iterator, device):
    def convert_ids_to_annos(entity_tokenizer, y_pred_label, attention_mask, pad=9):
        def convert_simplified_to_formal(raw_anno_string):
            return raw_anno_string\
                .upper()\
                .replace('BANK', '-BANK')\
                .replace('PRODUCT', '-PRODUCT')\
                .replace('COMMENTS', '-COMMENTS_')
        output = []
        b, c = y_pred_label.shape
        y_pred_label = y_pred_label * attention_mask + pad * (1 - attention_mask)
        y_pred_label = y_pred_label.cpu().numpy()
        for i in range(b):
            one_pred = y_pred_label[i].tolist()
            tokens = entity_tokenizer.convert_ids_to_tokens(one_pred, skip_special_tokens=True)
            raw_anno_string = entity_tokenizer.convert_tokens_to_string(tokens)
            anno_string = convert_simplified_to_formal(raw_anno_string)
            output.append(anno_string)
        return output

    model.eval()
    ids = []
    classes = []
    with torch.no_grad():
        for _, batch in tqdm(enumerate(iterator)):
            _id = batch["id"].numpy().tolist()
            text = batch["text"]
            text = [clean_split(one) for one in text]
            tokenized_text = tokenizer(text, max_length=100, add_special_tokens=False, truncation=True, padding=True, return_tensors="pt", is_split_into_words=True)
            tokenized_text = tokenized_text.to(device)

            output = model(**tokenized_text, labels=None)
            y_pred_label = output['logits'].argmax(dim=2)
            y_pred_label = convert_ids_to_annos(entity_tokenizer, y_pred_label, tokenized_text['attention_mask'])

            ids.extend(_id)
            classes.extend(y_pred_label)
    return dict(zip(ids, classes))


def train_and_test():
    entity_tokenizer = get_entity_tokenizer()
    entity_train_loader, test_loader = get_dataloaders(NamedEntityDataset, train_data, None, test_data, batch_size)
    tokenizer, model, optimizer, criterion = load_model(
        BertForTokenClassification,
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
        train_loss, train_acc = train(entity_tokenizer, tokenizer, model, entity_train_loader, optimizer, criterion,
                                      device)
        print("train loss: ", train_loss, "\t", "train acc:", train_acc)
        '''
        valid_loss, valid_acc = evaluate(entity_tokenizer, tokenizer, model_chinese_bert_wwm_ext, sentiment_valid_loader, device)
        print("valid loss: ", valid_loss, "\t", "valid acc:", valid_acc)
        '''

    result = test(entity_tokenizer, tokenizer, model, test_loader, device)
    with open(f'ner_res-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}.json', 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    train_and_test()
