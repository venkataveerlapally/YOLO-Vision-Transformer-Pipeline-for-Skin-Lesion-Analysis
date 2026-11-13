"""
train.py — YOLOv8 Training Script for HAM10000 dataset

Usage:
    python train.py --data data.yaml --weights yolov8s.pt --epochs 50 --img 640
"""

from ultralytics import YOLO
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data.yaml', help='path to dataset yaml')
    parser.add_argument('--weights', type=str, default='yolov8s.pt', help='pretrained weights')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--img', type=int, default=640, help='image size (pixels)')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--project', type=str, default='runs/train', help='output folder')
    parser.add_argument('--name', type=str, default='yolo_ham10000', help='run name')
    args = parser.parse_args()

    print("✅ Starting YOLOv8 Training on HAM10000 Dataset")

    model = YOLO(args.weights)  # load pretrained YOLO model
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
    )

    print("✅ Training Completed! Best weights saved in the runs/train folder.")

if __name__ == "__main__":
    main()
