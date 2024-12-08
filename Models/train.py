import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import wandb
from scheduler import CosineAnnealingWithWarmRestartsLR

# Seed for reproducibility
seed = 2001
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["opt"])
        self.ema.load_state_dict(checkpoint["ema"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.best_val_accuracy = checkpoint.get("best_acc", 0)
        print(f"Loaded checkpoint from {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EmoNeXt on Fer2013")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset directory structured with train, val, test subdirectories",
    )
    parser.add_argument("--output-dir", type=str, default="out", help="Directory to save outputs")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="Use Automatic Mixed Precision for faster training with reduced memory usage",
    )
    parser.add_argument(
        "--in_22k",
        action="store_true",
        default=False,
        help="Use ImageNet-22k pretrained weights if available",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="Number of worker threads for data loading"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume training from",
    )
    parser.add_argument(
        "--model-size",
        choices=["tiny", "small", "base", "large", "xlarge"],
        default="tiny",
        help="Size of the model architecture to use",
    )

    opt = parser.parse_args()
    print(opt)

    # Initialize Weights & Biases
    wandb.init(project="EmoNeXt", name=f"EmoNeXt_{opt.model_size}", config=vars(opt))

    # Define transforms
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Grayscale(),
            transforms.Resize(236),
            transforms.RandomRotation(degrees=20),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(repeat_channels),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(236),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(repeat_channels),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(236),
            transforms.TenCrop(224),
            transforms.Lambda(crop_and_stack),
            transforms.Lambda(crop_and_repeat),
        ]
    )

    # Create datasets
    train_dataset = datasets.ImageFolder(Path(opt.dataset_path) / "train", train_transform)
    val_dataset = datasets.ImageFolder(Path(opt.dataset_path) / "val", val_transform)
    test_dataset = datasets.ImageFolder(Path(opt.dataset_path) / "test", test_transform)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Initialize model
    net = get_model(len(train_dataset.classes), opt.model_size, in_22k=opt.in_22k)

    # Initialize and run Trainer
    trainer = Trainer(
        model=net,
        training_dataloader=train_loader,
        validation_dataloader=val_loader,
        testing_dataloader=test_loader,
        classes=train_dataset.classes,
        execution_name=f"EmoNeXt_{opt.model_size}",
        lr=opt.lr,
        output_dir=opt.output_dir,
        checkpoint_path=opt.checkpoint,
        max_epochs=opt.epochs,
        amp=opt.amp,
    )
    trainer.run()