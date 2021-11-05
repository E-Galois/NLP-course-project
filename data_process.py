from torch.utils.data import Dataset, DataLoader
import pandas as pd
import unicodedata


class SentimentDataset(Dataset):
    def __init__(self, path_to_file, nrows=None):
        if nrows is not None:
            self.dataset = pd.read_csv(path_to_file, nrows=nrows)
        else:
            self.dataset = pd.read_csv(path_to_file)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 根据 idx 分别找到 text 和 label
        text = self.dataset.iloc[idx]["text"]
        label = self.dataset.loc[idx]["class"]
        sample = {"text": text, "label": label}
        # 返回一个 dict
        return sample


class NamedEntityDataset(Dataset):
    def __init__(self, path_to_file, nrows=None):
        if nrows is not None:
            self.dataset = pd.read_csv(path_to_file, nrows=nrows)
        else:
            self.dataset = pd.read_csv(path_to_file)
        # self.dataset['BIO_anno'] = self.dataset['BIO_anno'].apply(lambda x: x.split(' '))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 根据 idx 分别找到 text 和 label
        text = self.dataset.iloc[idx]["text"]
        label = self.dataset.loc[idx]["BIO_anno"]
        sample = {"text": text, "label": label}
        # 返回一个 dict
        return sample


class IndexedTestDataset(Dataset):
    def __init__(self, path_to_file, nrows=None):
        if nrows is not None:
            self.dataset = pd.read_csv(path_to_file, nrows=nrows)
        else:
            self.dataset = pd.read_csv(path_to_file)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 根据 idx 分别找到 text 和 label
        _id = self.dataset.iloc[idx]["id"]
        text = self.dataset.iloc[idx]["text"]
        sample = {"id": _id, "text": text}
        # 返回一个 dict
        return sample


def _is_control(char):
    """Checks whether `char` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_whitespace(char):
    """Checks whether `char` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def clean_split(sentence):
    """Performs invalid character removal and whitespace cleanup on text."""
    ret = []
    for char in sentence:
        cp = ord(char)
        if cp == 0 or cp == 0xFFFD or _is_control(char):
            ret.append("[UNK]")
        if _is_whitespace(char):
            ret.append("[UNK]")
        if unicodedata.category(char) == "Mn":
            ret.append("[UNK]")
        else:
            ret.append(char)
    return ret


def get_dataloaders(train_cls, train_path, valid_path, test_path, batch_size, train_shuffle=True):
    # 加载训练集
    train_set = train_cls(train_path)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    '''
    # 加载验证集
    sentiment_valid_set = SentimentDataset(valid_path)
    sentiment_valid_loader = DataLoader(sentiment_valid_set, batch_size=batch_size, shuffle=False, num_workers=0)
    '''
    # 加载测试集
    test_set = IndexedTestDataset(test_path)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader
