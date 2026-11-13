Integration notes:
- detect_and_classify.py uses ultralytics YOLO and Hugging Face ViT.
- For YOLO: you can place YOLO weights at yolo_model/best.pt. Alternatively, set --yolo_weights to a path.
- For ViT: you can pass a Hugging Face model name or a local path to saved model weights (transformers format).
- Example:
    python integration/detect_and_classify.py --image dataset/combined_images/ISIC_0000000.jpg --yolo_weights yolo_model/best.pt --vit_model google/vit-base-patch16-224-in21k --out integration/results/detections/example_out.jpg
