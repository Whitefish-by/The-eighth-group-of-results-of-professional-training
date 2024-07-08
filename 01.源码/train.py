from ultralytics import YOLOv10

pre_model_name = 'yolov10x'

data_yaml_path = "/data2/YLW/yolo10/yolov10/datasets/o20240706.04/config.yaml"

model_yaml_path = f'{pre_model_name}.pt'

model = YOLOv10(model_yaml_path)
# If you want to finetune the model with pretrained weights, you could load the
# pretrained weights like below
# model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
# model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')

model.train(data=data_yaml_path, epochs=100, batch=40, imgsz=640, device=[0,1,2,3])