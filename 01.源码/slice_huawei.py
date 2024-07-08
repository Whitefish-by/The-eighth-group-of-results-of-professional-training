from tqdm import tqdm
import os
import cv2

source_path = 'PCB_瑕疵初赛样例集'
dirs = os.listdir(source_path)

target_path = "PCB_瑕疵初赛样例集_slice"


def split_huawei(source_path, split_percent=0.8, overlap = 0.33):
    target_path = source_path + '_slice'
    os.makedirs(target_path, exist_ok=True)
    for dir in dirs:
        if not dir.endswith('Img'):
         continue
        iamge_rootpath = os.path.join(source_path, dir)
        label_rootpath = iamge_rootpath.replace('Img', 'txt')

        os.makedirs(os.path.join(target_path, dir), exist_ok=True)
        os.makedirs(os.path.join(target_path, dir.replace('Img', 'txt')), exist_ok=True)

    for file in tqdm(os.listdir(source_path + '/' + dir), desc=dir):
        image_path = os.path.join(iamge_rootpath, file)
        label_path = os.path.join(label_rootpath, file.replace('bmp', 'txt'))

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # 分割图片为四部分
        h, w = img.shape[:2]
        h0, w0 = h // split_percent, w // split_percent
        y0, x0 = 0, 0
        x_min_set = []
        y_min_set = []
        while y0 < h:
            y_min_set.append(y0)
            y0 += h0 // overlap
        while x0 < w:
            x_min_set.append(x0)
            x0 += w0 // overlap
        for i, x_min in enumerate(x_min_set):
            for j, y_min in enumerate(y_min_set):
                x_max = x_min + w0
                y_max = y_min + h0
                if x_max > w:
                    x_max = w
                if y_max > h:
                    y_max = h

                target_label_path = os.path.join(target_path, dir.replace('Img', 'txt'),
                                                 file.replace('bmp', f'_{i}_{j}.txt'))
                target_image_path = os.path.join(target_path, dir, file.replace('bmp', f'_{i}_{j}.bmp'))
                # 读取和存储label文件

                with open(target_label_path, 'w') as g:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        lines = [line.strip().split('') for line in lines]
                        for line in lines:
                            label = int(line[0])
                            x_center = float(line[1])
                            y_center = float(line[2])
                            rect_width = float(line[3])
                            rect_height = float(line[4])
                            x_center_pixel = round(x_center * w)
                            y_center_pixel = round(y_center * h)
                            if x_center_pixel >= x_min and x_center_pixel <= x_max and y_center_pixel >= y_min and y_center_pixel <= y_max:
                                g.write(f"{label} {(x_center_pixel - x_min) / w0} {(y_center_pixel - y_min) / h0} {rect_width * (w / w0)} {rect_height * (h / h0)}\n")

                patch_img = img[y_min:y_max, x_min:x_max]
                cv2.imwrite(target_image_path, patch_img)


split_huawei(source_path, split_percent=0.8, overlap=0.33)




