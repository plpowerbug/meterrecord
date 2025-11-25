yolo pose train model=yolov8n-pose.pt data=yolo_dataset/data.yaml epochs=100 imgsz=640
python coco_to_yolo_pose.py
