from ultralytics import YOLOv10


model_path = "/data2/YLW/yolo10/yolov10/runs/detect/train.o20240706.03/weights/best.pt"

data_path = "/data2/YLW/yolo10/yolov10/datasets/v1.10.20240704.huawei.gray.val.yolo.sliced/config.yaml"

# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
model = YOLOv10(model_path)

model.val(data=data_path, batch=100, device=[0,1])