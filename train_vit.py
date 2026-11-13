"""
train_vit.py
Fine-tune a Vision Transformer (ViT) on the HAM10000 dataset arranged in folder-per-class format:

dataset/
  train/
    akiec/
    bcc/
    ...
  val/
  test/

Outputs:
 - saves best model checkpoint to vit_weights.pth (in the vit_model directory or configured output_dir)
 - logs basic training/validation accuracy and loss

Usage (example):
python train_vit.py --config vit_config.json
"""
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd

from transformers import ViTForImageClassification, ViTFeatureExtractor, get_linear_schedule_with_warmup, AdamW
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------
# Dataset helper
# ---------------------------
class ImageFolderWithPaths(Dataset):
    """Custom dataset that returns image tensor and label, but uses ViTFeatureExtractor for preprocessing."""
    def __init__(self, root_dir, feature_extractor, split="train", transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.feature_extractor = feature_extractor
        # gather (path,label)
        self.samples = []
        classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        for cls in classes:
            cls_dir = self.root_dir / cls
            for img_file in cls_dir.iterdir():
                if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((str(img_file), self.class_to_idx[cls]))
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {self.root_dir}. Make sure dataset is prepared.")
        self.num_classes = len(classes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        # feature_extractor expects PIL.Image or numpy array
        enc = self.feature_extractor(images=image, return_tensors="pt")
        # enc['pixel_values'] shape (1, C, H, W)
        pixel_values = enc['pixel_values'].squeeze(0)  # (C, H, W)
        return {"pixel_values": pixel_values, "label": label, "path": path}

def collate_fn(batch):
    # stacks pixel_values and labels
    pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    paths = [item["path"] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels, "paths": paths}

# ---------------------------
# Training / Validation loops
# ---------------------------
def train_one_epoch(model, dataloader, optimizer, device, scheduler=None):
    model.train()
    losses = []
    preds = []
    trues = []
    for batch in tqdm(dataloader, desc="Train", leave=False):
        inputs = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(pixel_values=inputs, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        losses.append(loss.item())
        batch_preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        preds.extend(batch_preds.tolist())
        trues.extend(labels.detach().cpu().numpy().tolist())
    avg_loss = float(np.mean(losses))
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="weighted")
    return avg_loss, acc, f1

def eval_model(model, dataloader, device):
    model.eval()
    losses = []
    preds = []
    trues = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            inputs = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            losses.append(loss.item())
            batch_preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            preds.extend(batch_preds.tolist())
            trues.extend(labels.detach().cpu().numpy().tolist())
    avg_loss = float(np.mean(losses)) if losses else 0.0
    acc = accuracy_score(trues, preds) if len(trues)>0 else 0.0
    f1 = f1_score(trues, preds, average="weighted") if len(trues)>0 else 0.0
    return avg_loss, acc, f1

# ---------------------------
# Main
# ---------------------------
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["use_cuda"] else "cpu")
    print("Using device:", device)

    # Feature extractor & model
    model_name = cfg.get("pretrained_model_name", "google/vit-base-patch16-224-in21k")
    print("Loading feature extractor and model:", model_name)
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    # create datasets to infer number of classes
    train_dataset = ImageFolderWithPaths(cfg["data_dir"], feature_extractor, split="train")
    val_dataset = ImageFolderWithPaths(cfg["data_dir"], feature_extractor, split="val")
    num_labels = train_dataset.num_classes
    print("Detected classes:", train_dataset.class_to_idx)

    model = ViTForImageClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=cfg.get("num_workers",2))
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=cfg.get("num_workers",2))

    # optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg.get("weight_decay", 0.0))
    total_steps = int(len(train_loader) * cfg["epochs"])
    warmup_steps = int(cfg.get("warmup_proportion", 0.1) * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_val_acc = 0.0
    out_dir = Path(cfg.get("output_dir", "vit_model"))
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / cfg.get("save_name", "vit_weights.pth")

    for epoch in range(1, cfg["epochs"] + 1):
        print(f"Epoch {epoch}/{cfg['epochs']}")
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, optimizer, device, scheduler)
        val_loss, val_acc, val_f1 = eval_model(model, val_loader, device)
        print(f" Train loss: {train_loss:.4f} | acc: {train_acc:.4f} | f1: {train_f1:.4f}")
        print(f" Val   loss: {val_loss:.4f} | acc: {val_acc:.4f} | f1: {val_f1:.4f}")

        # Save best model by val acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_to_idx": train_dataset.class_to_idx,
                "config": cfg
            }, checkpoint_path)
            print("Saved best checkpoint to", checkpoint_path)

    # Optionally evaluate on test set and save final model
    print("Training complete. Best val acc:", best_val_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="vit_config.json", help="path to JSON config")
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    cfg = json.load(open(args.config, "r"))
    main(cfg)
