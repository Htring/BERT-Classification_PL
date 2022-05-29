#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: dataloader.py
@time:2022/03/13
@description:
"""
import json
import os
from typing import Optional, Any
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import BertTokenizer
import pandas as pd


class BertDataSet(Dataset):
    __doc__ = """ bert格式dataset  """

    def __init__(self, content_label_list, tokenizer: BertTokenizer, tag2index_dict, max_length):
        self.token2index: dict = tag2index_dict
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.data_list = self._build_data_list(content_label_list)

    def _build_data_list(self, content_label_list):
        data_list = []
        for content, label in zip(*content_label_list):
            res = self.tokenizer.encode_plus(content,
                                             pad_to_max_length=True,
                                             max_length=self.max_length)
            data_unit = (res['input_ids'],
                         res["attention_mask"],
                         self.token2index.get(label))
            data_list.append(data_unit)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        return self.data_list[item]


class TextDataModule(pl.LightningDataModule):

    def __init__(self, data_dir="data", batch_size=128, max_length=200):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self.token2index, self.tag2index, self.index2token, self.index2tag = {}, {}, {}, {}
        self.tokenizer = BertTokenizer.from_pretrained("chinese-roberta-wwm-ext")
        self.train_set, self.dev_set, self.test_set = None, None, None
        self.setup()

    def _load_data(self, file_path):
        data_frame = pd.read_csv(file_path, sep="\t")
        contents = data_frame["text"].tolist()
        labels = data_frame["label"].tolist()
        return contents, labels

    def setup(self, stage: Optional[str] = None) -> None:
        train_contents_labels = self._load_data(os.path.join(self.data_dir, "train.tsv"))
        dev_contents_labels = self._load_data(os.path.join(self.data_dir, "dev.tsv"))
        test_contents_labels = self._load_data(os.path.join(self.data_dir, "test.tsv"))
        tag_set = set()
        tag_set.update(train_contents_labels[1])
        tag_list = list(tag_set)
        tag_list.sort()
        self.tag2index = {tag: index for index, tag in enumerate(tag_list)}
        self.index2tag = {index: tag for tag, index in self.tag2index.items()}

        self.train_set = BertDataSet(train_contents_labels, self.tokenizer, self.tag2index, self.max_length)
        self.dev_set = BertDataSet(dev_contents_labels, self.tokenizer, self.tag2index, self.max_length)
        self.test_set = BertDataSet(test_contents_labels, self.tokenizer, self.tag2index, self.max_length)

    @staticmethod
    def collate_fn(batch):
        batch_input_ids, batch_attention_mask, batch_labels = [], [], []
        for sub_batch in batch:
            batch_input_ids.append(sub_batch[0])
            batch_attention_mask.append(sub_batch[1])
            batch_labels.append(sub_batch[2])
        batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        return batch_input_ids.cuda(), batch_attention_mask.cuda(), batch_labels.cuda()

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dev_set, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def save_dict(self, save_dir):
        with open(os.path.join(save_dir, "index2tag.txt"), 'w', encoding='utf8') as writer:
            writer.write(json.dumps(self.index2tag, ensure_ascii=False))
