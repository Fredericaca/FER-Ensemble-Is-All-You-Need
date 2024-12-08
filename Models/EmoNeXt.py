import argparse
import itertools
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ema_pytorch import EMA
from sklearn.metrics import confusion_matrix
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import wandb
from scheduler import CosineAnnealingWithWarmRestartsLR
from matplotlib import rc
rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"]})
rc("text", usetex=True)

seed = 2001
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper functions for transforms
def repeat_channels(x):
    """Repeat grayscale channel to create 3-channel image."""
    return x.repeat(3, 1, 1)


def crop_and_stack(crops):
    """Convert list of PIL crops to stacked tensor."""
    return torch.stack([transforms.ToTensor()(crop) for crop in crops])


def crop_and_repeat(crops):
    """Repeat channels for each crop to create 3-channel images."""
    return torch.stack([crop.repeat(3, 1, 1) for crop in crops])


# Trainer Class
class Trainer:
    def __init__(
        self,
        model,
        training_dataloader,
        validation_dataloader,
        testing_dataloader,
        classes,
        output_dir,
        max_epochs: int = 10000,
        early_stopping_patience: int = 12,
        execution_name=None,
        lr: float = 1e-4,
        amp: bool = False,
        ema_decay: float = 0.99,
        ema_update_every: int = 16,
        gradient_accumulation_steps: int = 1,
        checkpoint_path: str = None,
    ):
        self.epochs = max_epochs
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.testing_dataloader = testing_dataloader

        self.classes = classes
        self.num_classes = len(classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device used: " + self.device.type)

        self.amp = amp
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.model = model.to(self.device)
        self.optimizer = AdamW(model.parameters(), lr=lr)
        self.scaler = torch.amp.GradScaler('cuda',enabled=self.amp)
        self.scheduler = CosineAnnealingWithWarmRestartsLR(
            self.optimizer, warmup_steps=128, cycle_steps=1024
        )
        self.ema = EMA(model, beta=ema_decay, update_every=ema_update_every).to(
            self.device
        )

        self.early_stopping_patience = early_stopping_patience
        self.output_directory = Path(output_dir)
        self.output_directory.mkdir(exist_ok=True)
        self.best_val_accuracy = 0
        self.execution_name = "model" if execution_name is None else execution_name

        if checkpoint_path:
            self.load(checkpoint_path)

        wandb.watch(model, log="all")

    def run(self):
        counter = 0
        for epoch in range(self.epochs):
            print(f"[Epoch: {epoch + 1}/{self.epochs}]")
            train_loss, train_accuracy = self.train_epoch()
            val_loss, val_accuracy = self.val_epoch()

            wandb.log(
                {
                    "Train Loss": train_loss,
                    "Val Loss": val_loss,
                    "Train Accuracy": train_accuracy,
                    "Val Accuracy": val_accuracy,
                    "Epoch": epoch + 1,
                }
            )

            if val_accuracy > self.best_val_accuracy:
                self.save()
                counter = 0
                self.best_val_accuracy = val_accuracy
            else:
                counter += 1
                if counter >= self.early_stopping_patience:
                    print(
                        f"Validation accuracy did not improve for {self.early_stopping_patience} epochs. Stopping training."
                    )
                    break

        self.test_model()
        wandb.finish()

    def train_epoch(self):
        self.model.train()
        avg_loss, avg_accuracy = [], []

        pbar = tqdm(unit="batch", total=len(self.training_dataloader))
        for batch_idx, (inputs, labels) in enumerate(self.training_dataloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            with torch.amp.autocast('cuda',enabled=self.amp):
                predictions, _, loss = self.model(inputs, labels)

            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                self.ema.update()
                self.scheduler.step()

            avg_loss.append(loss.item())
            avg_accuracy.append((predictions == labels).sum().item() / labels.size(0))

            pbar.set_postfix(
                {"loss": np.mean(avg_loss), "acc": np.mean(avg_accuracy) * 100.0}
            )
            pbar.update(1)

        pbar.close()
        return np.mean(avg_loss), np.mean(avg_accuracy) * 100.0

    def val_epoch(self):
        self.model.eval()
        avg_loss, predicted_labels, true_labels = [], [], []

        pbar = tqdm(unit="batch", total=len(self.validation_dataloader))
        with torch.no_grad():
            for inputs, labels in self.validation_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                with torch.amp.autocast('cuda',enabled=self.amp):
                    predictions, _, loss = self.model(inputs, labels)

                avg_loss.append(loss.item())
                predicted_labels.extend(predictions.tolist())
                true_labels.extend(labels.tolist())
                pbar.update(1)

        pbar.close()
        accuracy = (
            torch.eq(torch.tensor(predicted_labels), torch.tensor(true_labels))
            .float()
            .mean()
            .item()
        )
        return np.mean(avg_loss), accuracy * 100.0

    def test_model(self):
        self.ema.eval()
        predicted_labels, true_labels = [], []

        pbar = tqdm(unit="batch", total=len(self.testing_dataloader))
        with torch.no_grad():
            for inputs, labels in self.testing_dataloader:
                # inputs: [batch_size, 10, 3, 224, 224]
                batch_size, n_crops, C, H, W = inputs.size()
                # Reshape to [batch_size * 10, 3, 224, 224]
                inputs = inputs.view(batch_size * n_crops, C, H, W).to(self.device)
                # Repeat labels for each crop
                labels_repeated = labels.repeat(n_crops).to(self.device)

                with torch.amp.autocast('cuda',enabled=self.amp):
                    _, logits = self.ema(inputs)  # logits: [batch_size * 10, num_classes]

                # Reshape logits to [batch_size, 10, num_classes]
                logits = logits.view(batch_size, n_crops, -1)
                # Average logits across crops
                avg_logits = logits.mean(dim=1)  # [batch_size, num_classes]

                # Get predictions
                predictions = torch.argmax(avg_logits, dim=1)
                predicted_labels.extend(predictions.tolist())
                true_labels.extend(labels.tolist())
                pbar.update(1)

        pbar.close()
        accuracy = (
            torch.eq(torch.tensor(predicted_labels), torch.tensor(true_labels))
            .float()
            .mean()
            .item()
        )
        print(f"Test Accuracy: {accuracy * 100.0:.2f}%")

    def save(self):
        torch.save(
            {
                "model": self.model.state_dict(),
                "opt": self.optimizer.state_dict(),
                "ema": self.ema.state_dict(),
                "scaler": self.scaler.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "best_acc": self.best_val_accuracy,
            },
            str(self.output_directory / f"{self.execution_name}.pt"),
        )
        print(f"Model saved to {self.output_directory / f'{self.execution_name}.pt'}")

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["opt"])
        self.ema.load_state_dict(checkpoint["ema"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.best_val_accuracy = checkpoint.get("best_acc", 0)
        print(f"Loaded checkpoint from {checkpoint_path}")

import math
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torchvision.models import vgg16_bn, VGG16_BN_Weights
from torchvision.ops import StochasticDepth
import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from ema_pytorch import EMA
from torch.optim import AdamW
from torchvision import transforms
from tqdm import tqdm

import wandb
from scheduler import CosineAnnealingWithWarmRestartsLR
model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DotProductSelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(DotProductSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.norm = nn.LayerNorm(input_dim)
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        x = self.norm(x)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        scale = 1 / math.sqrt(math.sqrt(self.input_dim))
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        attention_weights = torch.softmax(scores, dim=-1)

        attended_values = torch.matmul(attention_weights, value)
        output = attended_values + x

        return output, attention_weights


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.stochastic_depth = StochasticDepth(drop_path, "row")

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.stochastic_depth(x)
        return x


class EmoNeXt(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=None,
        dims=None,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
    ):
        super().__init__()

        if dims is None:
            dims = [96, 192, 384, 768]
        if depths is None:
            depths = [3, 3, 9, 3]

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 52 * 52, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        )

        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                SELayer(dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.attention = DotProductSelfAttention(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 52 * 52)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(
            x.mean([-2, -1])
        )  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x, labels=None):
        x = self.stn(x)
        x = self.forward_features(x)
        _, weights = self.attention(x)
        logits = self.head(x)

        if labels is not None:
            mean_attention_weight = torch.mean(weights)
            attention_loss = torch.mean((weights - mean_attention_weight) ** 2)

            loss = F.cross_entropy(logits, labels, label_smoothing=0.2) + attention_loss
            return torch.argmax(logits, dim=1), logits, loss

        return torch.argmax(logits, dim=1), logits


def get_model(num_classes, model_size="base", in_22k=False):
    # if model_size == "tiny":
    #     depths = [3, 3, 9, 3]
    #     dims = [96, 192, 384, 768]
    #     url = (
    #         model_urls["convnext_tiny_22k"]
    #         if in_22k
    #         else model_urls["convnext_tiny_1k"]
    #     )
    # elif model_size == "small":
    #     depths = [3, 3, 27, 3]
    #     dims = [96, 192, 384, 768]
    #     url = (
    #         model_urls["convnext_small_22k"]
    #         if in_22k
    #         else model_urls["convnext_small_1k"]
    #     )
    if model_size == "base":
        depths = [3, 3, 27, 3]
        dims = [128, 256, 512, 1024]
        url = (
            model_urls["convnext_base_22k"]
            if in_22k
            else model_urls["convnext_base_1k"]
        )
    elif model_size == "large":
        depths = [3, 3, 27, 3]
        dims = [192, 384, 768, 1536]
        url = (
            model_urls["convnext_large_22k"]
            if in_22k
            else model_urls["convnext_large_1k"]
        )
    else:
        depths = [3, 3, 27, 3]
        dims = [256, 512, 1024, 2048]
        url = model_urls["convnext_xlarge_22k"]

    default_num_classes = 1000
    if in_22k:
        default_num_classes = 21841

    net = EmoNeXt(
        depths=depths, dims=dims, num_classes=default_num_classes, drop_path_rate=0.1
    )

    state_dict = load_state_dict_from_url(url=url)
    net.load_state_dict(state_dict["model"], strict=False)
    net.head = nn.Linear(dims[-1], num_classes)

    return net