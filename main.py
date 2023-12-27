import os
import torch
import wandb
import pickle
import random
import logging
import argparse
import torchaudio
import numpy as np
import torch.nn as nn


from tqdm import tqdm
from src.unet import UNet
from src.dataset import SynthesisGlueDataset
from torch.utils.data import DataLoader
from sklearn.metrics import top_k_accuracy_score

# code reference: https://github.com/milesial/Pytorch-UNet/blob/master/train.py

# logging.basicConfig(level="DEBUG")
DEVICE = "cuda"
BEST_SCORE = float("inf")


def set_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


# create model
class Base(nn.Module):
    def __init__(self, *args, **kargs):
        super().__init__()
        self.model = UNet(*args, **kargs)

    def get_features(self, x):
        return self.model(x)

    def forward(self, x):
        features = self.get_features(x)
        return features

    def evaluate(self, x, y):
        features = self(x)

        loss = torch.nn.functional.cross_entropy(
            features,
            y,
            torch.tensor([0.5, 2.0], device=y.device)
        )

        return dict(
            loss=loss,
            features=features
        )


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# validation function
@torch.no_grad()
def valid(model, dataloaders, epoch):
    global BEST_SCORE
    model.eval()
    dataset_loss = []

    for batch in tqdm(dataloaders["valid"]):
        x = batch["img"]
        y = batch["mask"]
        x = x.to(dtype=torch.float32)
        y = y.to(dtype=torch.long)
        result = model.evaluate(
            x=x.to(DEVICE),
            y=y.to(DEVICE)
        )
        loss = result["loss"]
        dataset_loss.append(loss.mean().detach().cpu().item())

    # top1 = top_k_accuracy_score(dataset_labels, dataset_probs, k=1)
    # top3 = top_k_accuracy_score(dataset_labels, dataset_probs, k=3)
    loss = sum(dataset_loss) / len(dataset_loss)

    if (loss < BEST_SCORE):
        BEST_SCORE = loss
        os.makedirs("models/", exist_ok=True)
        torch.save(model.state_dict(), f"models/{wandb.run.id}_{epoch}.pt")

    return dict(
        # top1=top1,
        # top3=top3,
        loss=loss
    )


@torch.no_grad()
def test(model, dataloaders):
    model.eval()
    dataset_loss = []

    for batch in tqdm(dataloaders["test"]):
        x = batch["img"]
        y = batch["mask"]
        x = x.to(dtype=torch.float32)
        y = y.to(dtype=torch.long)
        result = model.evaluate(
            x=x.to(DEVICE),
            y=y.to(DEVICE)
        )
        loss = result["loss"]
        dataset_loss.append(loss.mean().detach().cpu().item())

    loss = sum(dataset_loss) / len(dataset_loss)

    return dict(
        loss=loss
    )


# train function
def train(model, dataloaders, optimizer, epochs, lr_scheduler):
    global_step = 0
    train_dataloader = dataloaders["train"]

    for _epoch in range(epochs):
        wandb.log(
            {"epoch": _epoch},
            step=global_step
        )

        model.zero_grad()
        model.train()
        wandb.log(
            {
                "lr": get_lr(optimizer)
            },
            step=global_step
        )

        for batch in tqdm(train_dataloader):
            global_step += 1

            x = batch["img"]
            y = batch["mask"]
            x = x.to(dtype=torch.float32)
            y = y.to(dtype=torch.long)

            result = model.evaluate(
                x=x.to(DEVICE),
                y=y.to(DEVICE)
            )

            loss = result["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()
            optimizer.zero_grad()

            wandb.log(
                {
                    "train/loss": loss
                },
                step=global_step
            )

        # validation
        metrics = valid(model, dataloaders, epoch=_epoch)
        wandb.log(
            {
                f"valid/{name}": value
                for name, value in metrics.items()
            },
            step=global_step
        )

        # test
        metrics = test(model, dataloaders)
        wandb.log(
            {
                f"test/{name}": value
                for name, value in metrics.items()
            },
            step=global_step
        )

        # restore status
        model.zero_grad()
        model.train()

        # step lr scheduler
        if (not type(lr_scheduler) == type(None)):
            lr_scheduler.step()


# main
def main(args):
    set_seed(1019)
    # create dataloaders
    dataloaders = {
        split: DataLoader(
            SynthesisGlueDataset(
                root_dir="./dataset",
                split=split,
                file_match_pairs=[
                    ("*/*", (1, 2), "bmp"),
                    ("*/*", (0, 1), "jpg")
                ]
            ),
            shuffle=(True if split == "train" else False),
            batch_size=(2 if args.test else args.batch_size),
            num_workers=(0 if args.test else args.num_workers),
            drop_last=(True if split == "train" else False)
        )
        for split in ["train", "valid", "test"]
    }

    # create model
    model = Base(n_channels=1, n_classes=2, bilinear=True)
    model = model.to(memory_format=torch.channels_last)
    model.to(DEVICE)

    # create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.1
    )

    # create lr scheduler
    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer=optimizer,
    #     max_lr=args.lr,
    #     div_factor=10,
    #     final_div_factor=10,
    #     total_steps=args.epoch,
    #     pct_start=0.1,
    #     anneal_strategy="linear"
    # )
    lr_scheduler = None

    # init wandb
    wandb.init(
        project="GlueFinder",
        mode=("offline"if args.test else "online")
    )
    wandb.watch(models=model, log="gradients", log_freq=100)

    # train model
    train(
        model,
        dataloaders,
        optimizer,
        epochs=args.epoch,
        lr_scheduler=lr_scheduler
    )

    # terminate
    wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--lr", type=float, default=5e-5)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
