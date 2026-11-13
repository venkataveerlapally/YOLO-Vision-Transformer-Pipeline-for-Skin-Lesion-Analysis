"""
Template to convert existing masks/metadata to YOLO-format labels.
This script depends on having bounding box coordinates per image.
If you only have segmentation masks, derive bounding boxes from masks.
Output: dataset/labels/<image_id>.txt with lines: class_id x_center y_center width height (normalized)
"""
import os, argparse
from PIL import Image
import numpy as np

def bbox_from_mask(mask_path):
    # mask assumed binary (0/255) or single-channel
    m = Image.open(mask_path).convert('L')
    arr = np.array(m)
    ys, xs = np.where(arr > 0)
    if len(xs)==0:
        return None
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return x_min, y_min, x_max, y_max

def write_yolo_label(bbox, img_w, img_h, class_id, out_path):
    x_min, y_min, x_max, y_max = bbox
    x_c = (x_min + x_max) / 2.0 / img_w
    y_c = (y_min + y_max) / 2.0 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    with open(out_path, 'w') as f:
        f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--masks_dir', required=True, help='folder with mask images (one per image)')
    p.add_argument('--images_dir', required=True)
    p.add_argument('--out_labels_dir', default='dataset/labels')
    p.add_argument('--class_id', type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out_labels_dir, exist_ok=True)
    for fname in os.listdir(args.masks_dir):
        maskp = os.path.join(args.masks_dir, fname)
        img_name = os.path.splitext(fname)[0] + '.jpg'
        imgp = os.path.join(args.images_dir, img_name)
        if not os.path.exists(imgp): 
            print('Image not found for mask', fname); continue
        bbox = bbox_from_mask(maskp)
        if bbox is None: 
            print('No mask for', fname); continue
        from PIL import Image
        img = Image.open(imgp)
        write_yolo_label(bbox, img.width, img.height, args.class_id, os.path.join(args.out_labels_dir, os.path.splitext(img_name)[0]+'.txt'))
    print('Labels written to', args.out_labels_dir)
