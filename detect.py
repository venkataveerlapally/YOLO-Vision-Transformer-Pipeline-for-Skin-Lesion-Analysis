"""
detect.py — Run YOLO lesion detection on test or new images.

Usage:
    python detect.py --weights best.pt --source ../dataset/test/images --out results/
"""

from ultralytics import YOLO
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best.pt', help='path to YOLO weights')
    parser.add_argument('--source', type=str, default='../dataset/test/images', help='source folder or image')
    parser.add_argument('--out', type=str, default='results', help='output folder for detections')
    parser.add_argument('--img', type=int, default=640, help='image size')
    args = parser.parse_args()

    print("🔍 Running YOLOv8 detection on:", args.source)

    model = YOLO(args.weights)
    results = model.predict(source=args.source, imgsz=args.img, project=args.out, save=True)

    print(f"✅ Detection completed! Results saved in {args.out}")

if __name__ == "__main__":
    main()
