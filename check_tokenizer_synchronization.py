from tqdm import tqdm
from settings.settings_named_entity import NERSettings as CFG
from data_process import get_dataloaders, NamedEntityDataset, clean_split
from model_process import load_model
from transformers import BertForTokenClassification, BertConfig, BertTokenizer
from run_named_entity import get_entity_tokenizer
import pandas as pd


entity_tokenizer = get_entity_tokenizer()
loader, _, _ = get_dataloaders(
    NamedEntityDataset,
    CFG.train_data,
    CFG.valid_data,
    CFG.train_data,
    1,
    train_shuffle=False
)

tokenizer, model, optimizer, criterion = load_model(
    RobertaForTokenClassification,
    RobertaConfig,
    RobertaTokenizer,
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

inconformity = {'text': [], 'label': [], 'tlen': [], 'llen': []}
for i, batch in tqdm(enumerate(loader)):
    ori_text = batch["text"]
    ori_label = batch["label"]
    # text = [' '.join(one.replace(' ', '')) for one in ori_text]
    text = [clean_split(one) for one in ori_text]
    label = [one.replace('-', '').replace('_', '') for one in ori_label]
    # label = [entity_tokenizer.convert_tokens_to_ids(one.split(' ')) for one in label]

    # tokenized_text 包括 input_ids， token_type_ids， attention_mask
    tokenized_text = tokenizer(text, max_length=100, add_special_tokens=False, truncation=True, padding=True,
                               return_tensors="pt", is_split_into_words=True)
    tokenized_text = tokenized_text.to(CFG.device)
    tokenized_label = entity_tokenizer(label, max_length=100, add_special_tokens=False, truncation=True, padding=True,
                                       return_tensors="pt")
    tokenized_label = tokenized_label['input_ids'].to(CFG.device)
    if tokenized_text['input_ids'].shape[1] != tokenized_label.shape[1]:
        print(ori_text)
        print(text)
        print(tokenized_text)
        inconformity['text'].append(ori_text[0])
        inconformity['label'].append(ori_label[0])
        inconformity['tlen'].append(tokenized_text['input_ids'].shape[1])
        inconformity['llen'].append(tokenized_label.shape[1])
pd.DataFrame(inconformity).to_csv('./data/inconformity.csv', sep=',', index=False)
