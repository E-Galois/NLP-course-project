from torch.utils.data import Dataset, DataLoader
import pandas as pd
from settings import *


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


class IndexedSentimentDataset(SentimentDataset):
    def __init__(self, path_to_file, nrows=None):
        super(IndexedSentimentDataset, self).__init__(path_to_file, nrows=nrows)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 根据 idx 分别找到 text 和 label
        _id = self.dataset.iloc[idx]["id"]
        text = self.dataset.iloc[idx]["text"]
        sample = {"id": _id, "text": text}
        # 返回一个 dict
        return sample


# 加载训练集
sentiment_train_set = SentimentDataset(train_data)
sentiment_train_loader = DataLoader(sentiment_train_set, batch_size=batch_size, shuffle=True, num_workers=0)
'''
# 加载验证集
sentiment_valid_set = SentimentDataset(valid_data)
sentiment_valid_loader = DataLoader(sentiment_valid_set, batch_size=batch_size, shuffle=False, num_workers=0)
'''
# 加载测试集
sentiment_test_set = IndexedSentimentDataset(test_data)
sentiment_test_loader = DataLoader(sentiment_test_set, batch_size=batch_size, shuffle=False, num_workers=0)
