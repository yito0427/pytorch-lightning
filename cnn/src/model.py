import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as ptl


class CNN(ptl.LightningModule):
    # モデルの定義(PyTorchと一緒)
    def __init__(self):
        super(CNN, self).__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.linear = nn.Linear(12544, 10)
    # フィードフォワードの計算の定義(PyTorchと一緒)
    def forward(self, image):

        h = self.c1(image)
        h = self.c2(h)

        batch_size = h.size(0)
        h = self.linear(h.view(batch_size, -1))
        return F.log_softmax(h, dim=1)

    # accuacyを計算するためのプライベート関数
    def _accuracy(self, preds, labels):
        _, preds = torch.max(preds, dim=1)
        return (preds == labels).sum().float() / preds.size(0)

    # ミニバッチに対するトレーニングの関数
    # 'loss'をキーにしないとバックワードが入らない 
    def training_step(self, batch, batch_nb):
        images, labels = batch
        preds = self.forward(images)
        return {'loss': F.nll_loss(preds, labels)}

    # ミニバッチに対するバリデーションの関数
    def validation_step(self, batch, batch_nb):
        images, labels = batch
        preds = self.forward(images)
        return {'val_nll_loss': F.nll_loss(preds, labels),
                'val_accuracy': self._accuracy(preds, labels)}

    # バリデーションループが終わったときに実行される関数
    def validation_end(self, outputs):
        avg_val_accuracy = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        return {'avg_val_accuracy': avg_val_accuracy}

    # 最適化アルゴリズムの指定
    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.001)]

    # データローダーの定義
    @ptl.data_loader  
    def tng_dataloader(self):
        return DataLoader(
            MNIST(
                os.getcwd(),
                train=True,
                download=True,
                transform=transforms.ToTensor()),
            batch_size=32)

    @ptl.data_loader
    def val_dataloader(self):
        return DataLoader(
            MNIST(
                os.getcwd(),
                train=True,
                download=True,
                transform=transforms.ToTensor()),
            batch_size=32)

    @ptl.data_loader
    def test_dataloader(self):
        return DataLoader(
            MNIST(
                os.getcwd(),
                train=True,
                download=True,
                transform=transforms.ToTensor()),
            batch_size=32)

