import argparse
from ultralytics import YOLO
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image, ImageDraw, ImageFont
import torch
import os

def load_models(yolo_weights, vit_model_name_or_path, device='cpu'):
    yolo = YOLO(yolo_weights)
    vit = ViTForImageClassification.from_pretrained(vit_model_name_or_path)
    vit.to(device)
    feature_extractor = ViTFeatureExtractor.from_pretrained(vit_model_name_or_path)
    return yolo, vit, feature_extractor

def detect_and_classify(image_path, yolo, vit, feature_extractor, device='cpu', save_out=None):
    results = yolo(image_path)  # Ultralytics returns a Results object
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    detections = results[0].boxes if hasattr(results[0],'boxes') else None
    if detections is None or len(detections)==0:
        print("No detections")
        if save_out:
            img.save(save_out)
        return
    for box in detections:
        # box.xyxy -> tensor
        x1,y1,x2,y2 = box.xyxy[0].tolist()
        crop = img.crop((x1,y1,x2,y2)).resize((224,224))
        inputs = feature_extractor(images=crop, return_tensors='pt')
        inputs = {k:v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            out = vit(**inputs)
            pred = torch.argmax(out.logits, dim=-1).item()
        draw.rectangle([x1,y1,x2,y2], outline='red', width=3)
        draw.text((x1, y1-12), f"Class {pred}", fill='red')
    if save_out:
        os.makedirs(os.path.dirname(save_out), exist_ok=True)
        img.save(save_out)
    print("Done. Saved to", save_out)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--yolo_weights', default='yolo_model/best.pt')
    p.add_argument('--vit_model', default='google/vit-base-patch16-224-in21k')
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', default='integration/results/detections/out.jpg')
    args = p.parse_args()
    yolo, vit, fe = load_models(args.yolo_weights, args.vit_model, device=args.device)
    detect_and_classify(args.image, yolo, vit, fe, device=args.device, save_out=args.out)
