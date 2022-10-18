import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from multiprocessing import cpu_count
from torchmetrics import Accuracy
import torch.nn as nn

from dataloading import FaceDataset
import os.path as osp


class FaceDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=24):
        super().__init__()
        self.train_dataset = FaceDataset(osp.join("face_dataset", "train"))
        self.val_dataset = FaceDataset(osp.join("face_dataset", "val"))
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=2, num_workers=cpu_count())

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=2, num_workers=cpu_count())


class FriendFaceDetector(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

        # Criterion
        self.criterion = nn.CrossEntropyLoss()

        # Accuracy
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-2, total_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        feat, label = batch

        logits = self.forward(feat)
        loss = self.criterion(logits, label.squeeze(1))
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        feat, label = batch

        logits = self.forward(feat)
        loss = self.criterion(logits, label.squeeze(1))
        preds = F.log_softmax(logits, dim=1).argmax(dim=1)
        self.val_accuracy.update(preds, label.squeeze(1))

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        feat, label = batch

        logits = self.forward(feat)
        loss = self.criterion(logits, label.squeeze(1))
        preds = F.log_softmax(logits, dim=1).argmax(dim=1)
        self.test_accuracy.update(preds, label.squeeze(1))

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

        return loss
