from ultralytics import YOLOv10
import cv2
import os
from tqdm import tqdm

map_class = ["Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]

def get_labels(label_path):
    with open(label_path, "r") as f:
        # 读取所有的行
        labels = f.readlines()
        # 去掉每行的空格
        labels = [label.strip().split(" ") for label in labels]
        return labels

class Predict:
    def __init__(self):
        self.model_path = "/data2/YLW/yolo10/yolov10/runs/detect/train.o20240706.03/weights/best.pt"
        self.dataset_huawei_path = "/data2/YLW/yolo10/yolov10/datasets/v1.10.20240704.huawei.gray.val.yolo.sliced"
        self.model = YOLOv10(self.model_path)

    def predict_single_image(self, image_path):
        model = YOLOv10(self.model_path)

        res = {}
        detection_classes = []
        detection_boxes = []
        detection_scores = []
        results = model.predict(source=image_path, device=3)

        # print(results)

        # 打印预测结果
        for result in results:
            print("Detected objects:")
            names = result.names.values()
            names = list(names)
            # print(names)
            for box in result.boxes:
                # 获取边界框坐标并转换为整数
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf.item()
                cls = int(box.cls.item())
                detection_classes.append(names[cls])
                detection_boxes.append([y_min, x_min, y_max, x_max])
                detection_scores.append(conf)
            res['detection_classes'] = detection_classes
            res['detection_boxes'] = detection_boxes
            res['detection_scores'] = detection_scores
        return res

    def predict_huawei_datasets(self):
        data_path = os.path.join(self.dataset_huawei_path, "images")
        os.makedirs(os.path.join(self.dataset_huawei_path, "results"), exist_ok=True)
        for dir in os.listdir(data_path):
            os.makedirs(os.path.join(self.dataset_huawei_path, "results", dir), exist_ok=True)
            images_path = os.path.join(data_path, dir)
            res = self.model.predict(source=images_path, device=3)
            # print(res[0])
            # print("-----------------------------------------------------------------")
            for i in tqdm(range(len(res)), desc=f"Processing {dir}\n"):
                result = res[i]
                image_path = result.path.strip().replace(' ', '_')
                image = cv2.imread(image_path)
                for box in result.boxes:
                    x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf.item()
                    # 保留两位小数
                    conf = round(conf, 2)
                    cls = int(box.cls.item())
                    print(map_class[cls] + str(conf) + " " + str(x_min) + " " + str(y_min) + " " + str(x_max) + " " + str(y_max) + " " + image_path)
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                    cv2.putText(image, map_class[cls] + str(conf), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                label_path = os.path.join(self.dataset_huawei_path, "labels", dir, os.path.splitext(os.path.basename(image_path))[0]+".txt")
                labels = get_labels(label_path)
                for i in range(len(labels)):
                    x_label, y_label, width_label, height_label = map(float, labels[i][1:])
                    y_min = int(y_label * image.shape[0] - height_label * image.shape[0] / 2)
                    x_min = int(x_label * image.shape[1] - width_label * image.shape[1] / 2)
                    y_max = int(y_label * image.shape[0] + height_label * image.shape[0] / 2)
                    x_max = int(x_label * image.shape[1] + width_label * image.shape[1] / 2)
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(image, map_class[int(labels[i][0])], (x_min, y_max + int(height_label * image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                output_image_path = os.path.join(self.dataset_huawei_path, "results", dir, os.path.basename(image_path))
                cv2.imwrite(output_image_path, image)

        def predict_huawei_datasets__(self):
            data_path = os.path.join(self.dataset_huawei_path, "images")
            os.makedirs(os.path.join(self.dataset_huawei_path, "results"), exist_ok=True)
            for dir in os.listdir(data_path):
                os.makedirs(os.path.join(self.dataset_huawei_path, "results", dir), exist_ok=True)
                images_path = os.path.join(data_path, dir)
                res = self.model.predict(source=images_path, device=3)
                # print(res[0])
                # print("-----------------------------------------------------------------")
                for i in tqdm(range(len(res)), desc=f"Processing {dir}\n"):
                    result = res[i]
                    image_path = result.path.strip().replace(' ', '_')
                    image = cv2.imread(image_path)
                    for box in result.boxes:
                        x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf.item()
                        # 保留两位小数
                        conf = round(conf, 2)
                        cls = int(box.cls.item())
                        print(map_class[cls] + str(conf) + " " + str(x_min) + " " + str(y_min) + " " + str(
                            x_max) + " " + str(y_max) + " " + image_path)
                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                        cv2.putText(image, map_class[cls] + str(conf), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)

                    label_path = os.path.join(self.dataset_huawei_path, "labels", dir,
                                              os.path.splitext(os.path.basename(image_path))[0] + ".txt")
                    labels = get_labels(label_path)
                    for i in range(len(labels)):
                        x_label, y_label, width_label, height_label = map(float, labels[i][1:])
                        y_min = int(y_label * image.shape[0] - height_label * image.shape[0] / 2)
                        x_min = int(x_label * image.shape[1] - width_label * image.shape[1] / 2)
                        y_max = int(y_label * image.shape[0] + height_label * image.shape[0] / 2)
                        x_max = int(x_label * image.shape[1] + width_label * image.shape[1] / 2)
                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(image, map_class[int(labels[i][0])],
                                    (x_min, y_max + int(height_label * image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 0), 2)

                    output_image_path = os.path.join(self.dataset_huawei_path, "results", dir,
                                                     os.path.basename(image_path))
                    cv2.imwrite(output_image_path,image)









        # images_path = os.path.join(self.dataset_huawei_path, "images")
        # labels_path = os.path.join(self.dataset_huawei_path, "labels")
        # for dir in tqdm(os.listdir(images_path), desc="Directories"):
        #     for file in tqdm(os.listdir(os.path.join(images_path, dir)), desc=f"Processing {dir}"):
        #
        #         # 创建result文件夹
        #         result_path = os.path.join(self.dataset_huawei_path, "results")
        #         os.makedirs(result_path, exist_ok=True)
        #         # 创建子文件夹dir文件夹
        #         dir_path = os.path.join(result_path, dir)
        #         os.makedirs(dir_path, exist_ok=True)
        #         file_path = os.path.join(dir_path, file)
        #         # 将images/dir/file的这个图片
        #         image = cv2.imread(os.path.join(images_path, dir, file))
        #         # 实际label的文件路径
        #         label_path = os.path.join(labels_path, dir, file.replace(".jpg", ".txt"))
        #         labels = get_labels(label_path)
        #         for i in range(len(res['detection_classes'])):
        #             y_min, x_min, y_max, x_max = res['detection_boxes'][i]
        #             # 将y_min, x_min, y_max, x_max原本为比例转换为像素
        #             y_min = int(y_min * image.shape[0])
        #             x_min = int(x_min * image.shape[1])
        #             y_max = int(y_max * image.shape[0])
        #             x_max = int(x_max * image.shape[1])
        #
        #             cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        #             cv2.putText(image, res['detection_classes'][i], (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #
                # for i in range(len(labels)):
                #     x_label, y_label, width_label, height_label = map(float, labels[i][1:])
                #     y_min = int(y_label * image.shape[0] - height_label * image.shape[0] / 2)
                #     x_min = int(x_label * image.shape[1] - width_label * image.shape[1] / 2)
                #     y_max = int(y_label * image.shape[0] + height_label * image.shape[0] / 2)
                #     x_max = int(x_label * image.shape[1] + width_label * image.shape[1] / 2)
                #     cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                #     cv2.putText(image, map_class[int(labels[i][0])], (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        #         cv2.imwrite(file_path, image)

predict = Predict()
# res = predict.predict_single_image("/data2/YLW/yolo10/yolov10/ultralytics/assets/bus.jpg")
predict.predict_huawei_datasets()
