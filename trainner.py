#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: trainner.py
@time:2022/03/13
@description:
"""
import json
import os
from argparse import ArgumentParser
from transformers import BertTokenizer
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from model import BertFuneTunePl
from dataloader import TextDataModule


def training(param):
    dm = TextDataModule(batch_size=param.batch_size)
    checkpoint_callback = ModelCheckpoint(monitor='f1_score',
                                          filename="bert-{epoch:03d}-{val_loss:.2f}-{f1_score:.3f}",
                                          dirpath=param.save_dir,
                                          save_top_k=3)

    param.vocab_size = len(dm.token2index)
    param.output_size = len(dm.tag2index)
    model = BertFuneTunePl(param)
    if param.load_pre:
        model = model.load_from_checkpoint(param.pre_ckpt_path, param)
    logger = TensorBoardLogger("log_dir", name="bert_pl")

    trainer = pl.Trainer(logger=logger, gpus=1,
                         callbacks=[checkpoint_callback],
                         max_epochs=param.epoch,
                         # precision=16,
                         accumulate_grad_batches=10,  # 由于使用bert时，批次数据量太少，设置多个批次后再进行梯度处理
                         # limit_train_batches=0.1,
                         # limit_val_batches=0.1,
                         gradient_clip_val=0.5
                         )
    if param.train:
        trainer.fit(model=model, datamodule=dm)
        dm.save_dict(param.save_dir)
    if param.test:
        trainer.test(model, dm)


def model_use(param):
    tokenizer = BertTokenizer.from_pretrained("chinese-roberta-wwm-ext")

    def _load_dict(dir_name):

        with open(os.path.join(dir_name, 'index2tag.txt'), 'r', encoding='utf8') as reader:
            index2tag = json.load(reader)

        return index2tag

    def _number_data(content):
        res = tokenizer.encode_plus(content,
                              pad_to_max_length=True,
                              max_length=param.max_length,)
        return torch.tensor([res["input_ids"]], dtype=torch.long), torch.tensor([res["attention_mask"]], dtype=torch.long)

    index2tag = _load_dict(param.save_dir)
    param.output_size = len(index2tag)
    model = BertFuneTunePl.load_from_checkpoint(param.pre_ckpt_path, param=param)
    test_data = "空间大，上路很有面子"
    result_index = model.forward(*_number_data(content=test_data)).argmax(dim=-1)[0].item()
    print(index2tag.get(str(result_index)))  # 1


if __name__ == '__main__':
    model_parser = BertFuneTunePl.add_argparse_args()
    parser = ArgumentParser(parents=[model_parser])
    parser.add_argument('-lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('-batch_size', type=int, default=6, help='批次数据大小')
    parser.add_argument('-epoch', type=int, default=2)
    parser.add_argument('-save_dir', type=str, default="model_save/bert", help='模型存储位置')
    parser.add_argument('-load_pre', type=bool, default=False, help='是否加载已经训练好的ckpt')
    parser.add_argument('-test', type=bool, default=True, help='是否测试数据')
    parser.add_argument('-train', type=bool, default=False, help='是否训练')
    parser.add_argument('-max_length', type=int, default=200, help='截取句子的最大长度')
    parser.add_argument('-pre_ckpt_path', type=str,
                        default="model_save/bert/bert-epoch=001-val_loss=0.10-f1_score=0.968.ckpt",
                        help='是否加载已经训练好的ckpt')

    args = parser.parse_args()
    # training(args)
    model_use(args)
