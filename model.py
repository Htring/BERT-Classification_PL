#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: model.py
@time:2022/03/13
@description:
"""
from typing import Optional, Any
import torchmetrics
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
import torch.nn.functional as F
from sklearn.metrics import classification_report
from argparse import ArgumentParser
from transformers import BertForSequenceClassification, AdamW


class BertFuneTunePl(pl.LightningModule):

    @staticmethod
    def add_argparse_args() -> ArgumentParser:
        parser = ArgumentParser(description='TextCNN text classifier', add_help=False)
        parser.add_argument('-weight_decay', type=float, default=1e-2, help='权重衰减， default 0.5')
        return parser

    def __init__(self, param):
        super().__init__()
        self.lr = param.lr
        self.weight_decay = param.weight_decay
        self.model = BertForSequenceClassification.from_pretrained("hfl/chinese-bert-wwm-ext",
                                                                   num_labels=param.output_size,
                                                                   )

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch_input_ids, batch_attention_mask, batch_labels = batch
        out = self.forward(batch_input_ids, batch_attention_mask)
        loss = F.cross_entropy(out, batch_labels)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        batch_input_ids, batch_attention_mask, y = batch
        pred: torch.Tensor = self.forward(batch_input_ids, batch_attention_mask)
        loss = F.cross_entropy(pred, y)
        pred_index = pred.argmax(dim=-1)
        f1_score = torchmetrics.functional.f1_score(pred_index, y, average="micro")
        accuracy = torchmetrics.functional.accuracy(pred_index, y, average="micro")
        recall = torchmetrics.functional.recall(pred_index, y, average="micro")
        self.log("val_loss", loss)
        self.log("f1_score", f1_score)
        self.log("recall", recall)
        self.log("accuracy", accuracy)
        return {"true": y, "pred": pred_index, "loss": loss, "f1_score": f1_score}

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._test_unit(outputs)

    def _test_unit(self, outputs):
        pred_lst, true_lst = [], []
        for batch_result in outputs:
            pred_lst.extend(batch_result["pred"].cpu().tolist())
            true_lst.extend(batch_result['true'].cpu().tolist())
        report = classification_report(true_lst, pred_lst)
        print("\n", report)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        batch_input_ids, batch_attention_mask, y = batch
        pred: torch.Tensor = self.forward(batch_input_ids, batch_attention_mask)
        loss = F.cross_entropy(pred, y)
        pred_index = pred.argmax(dim=-1)
        f1_score = torchmetrics.functional.f1_score(pred_index, y, average="micro")
        self.log("val_loss", loss)
        self.log("f1_score", f1_score)
        return {"true": y, "pred": pred_index, "loss": loss, "f1_score": f1_score}

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._test_unit(outputs)


    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        return optimizer

    def forward(self, batch_input_ids, batch_attention_mask) -> Any:
        output = self.model(batch_input_ids, batch_attention_mask)[0]
        return output
