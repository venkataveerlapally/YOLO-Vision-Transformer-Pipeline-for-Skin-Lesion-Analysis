"""
test_vit.py
Evaluate a saved ViT checkpoint (vit_weights.pth) on the test split.

Usage:
python test_vit.py --config vit_config.json --checkpoint vit_weights.pth
"""
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTFeatureExtractor
from sklearn.metrics import classification_report, accuracy_score, f1_score
from PIL import Image
import numpy as np

# reuse the dataset class from train script (lightweight copy)
class TestImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, feature_extractor, split="test"):
        self.root_dir = Path(root_dir) / split
        self.feature_extractor = feature_extractor
        self.samples = []
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            for img_file in cls_dir.iterdir():
                if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((str(img_file), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        enc = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = enc['pixel_values'].squeeze(0)
        return {"pixel_values": pixel_values, "label": label, "path": path}

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    paths = [item["path"] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels, "paths": paths}

def load_checkpoint(checkpoint_path, model, device):
    ck = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in ck:
        model.load_state_dict(ck["model_state_dict"])
    else:
        model.load_state_dict(ck)
    return ck

def evaluate(cfg, checkpoint_path, device):
    model_name = cfg.get("pretrained_model_name", "google/vit-base-patch16-224-in21k")
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    test_dataset = TestImageDataset(cfg["data_dir"], feature_extractor, split="test")
    test_loader = DataLoader(test_dataset, batch_size=cfg.get("batch_size",16), shuffle=False, collate_fn=collate_fn, num_workers=cfg.get("num_workers",2))
    num_labels = len(test_dataset.classes)
    model = ViTForImageClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)
    ck = load_checkpoint(checkpoint_path, model, device)
    model.eval()
    all_preds = []
    all_trues = []
    all_paths = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            inputs = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_trues.extend(labels.detach().cpu().numpy().tolist())
            all_paths.extend(batch["paths"])
    acc = accuracy_score(all_trues, all_preds)
    f1 = f1_score(all_trues, all_preds, average="weighted")
    print("Test accuracy:", acc)
    print("Test weighted F1:", f1)
    print("Classification report:")
    print(classification_report(all_trues, all_preds, target_names=test_dataset.classes, zero_division=0))
    # Optionally save csv of results
    out_csv = Path(cfg.get("output_dir", "vit_model")) / "test_predictions.csv"
    import csv
    with open(out_csv, "w", newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout)
        writer.writerow(["path", "true_label", "pred_label"])
        for p, t, pred in zip(all_paths, all_trues, all_preds):
            writer.writerow([p, test_dataset.classes[t], test_dataset.classes[pred]])
    print("Saved detailed predictions to", out_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="vit_config.json")
    parser.add_argument("--checkpoint", default="vit_model/vit_weights.pth", help="path to saved checkpoint")
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError("Config not found: " + args.config)
    cfg = json.load(open(args.config, "r"))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_cuda", True) else "cpu")
    evaluate(cfg, args.checkpoint, device)
