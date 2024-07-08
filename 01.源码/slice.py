import math

import cv2

types = [
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper"
]

image_dir_suffix = "_Img"
label_dir_suffix = "_txt"

slice_dir = "sliced"



# label is in yolov5 format
def slice_image(image_path, label_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    '''
    $$W_{\text{sliced}} = \frac{W_\text{raw}}{\text{round}(W_\text{raw}/800)}$$
    $$H_{\text{sliced}} = \frac{H_\text{raw}}{\text{round}(H_\text{raw}/600)}$$
    '''

    # 第一版v1.1:  width:800   height:600  0.196
    # 第二版v1.2:  width:600   height:400  0.222
    # 第三版v1.3:  width:400   height:300  0.161
    # 第四版v1.4:  width:600   height:500  0.208
    # 第五版v1.5:  width:500   height:500  0.212
    # 第六版v1.6:  width:600   height:600  0.208
    # 第七版v1.7:  width:500   height:600  0.214
    # 第八版v1.8:  width:550   height:350  0.222
    # 第七版v1.9:  width:500   height:300  0.208
    # 第十版v1.10:  width:200   height:200  0.00849
    # 第十一版v1.11:  width:200   height:100  0.23
    # 第十一版v1.12:  width:200   height:300  0.175
    target_width = 200
    target_height = 300

    # W_{\text{sliced}} = \frac{W_\text{raw}}{\text{round}(W_\text{raw}/800)}
    sliced_width = math.ceil(width/(width //target_width))
    # print(sliced_width)
    # H_{\text{sliced}} = \frac{H_\text{raw}}{\text{round}(H_\text{raw}/600)}
    sliced_height = math.ceil(height/(height //target_height))

    # print(type(width//800))
    #
    # print(sliced_height)

    aa = height//target_height
    bb = width//target_width
    for i in range(aa):
        for j in range(bb):
            sliced_image = image[i*sliced_height:(i+1)*sliced_height, j*sliced_width:(j+1)*sliced_width]

            if not os.path.exists(f"{slice_dir}/images/{os.path.dirname(image_path)}"):
                os.makedirs(f"{slice_dir}/images/{os.path.dirname(image_path)}")
            if not os.path.exists(f"{slice_dir}/labels/{os.path.dirname(image_path)}"):
                os.makedirs(f"{slice_dir}/labels/{os.path.dirname(image_path)}")
            cv2.imwrite(f"{slice_dir}/images/{os.path.splitext(image_path)[0]}_{i}_{j}.bmp", sliced_image)
            with open(f"{slice_dir}/labels/{os.path.splitext(image_path)[0]}_{i}_{j}.txt", "w") as f:
                with open(label_path, "r") as label_file:
                    for line in label_file:
                        line = line.strip().split()
                        label = int(line[0])
                        x_center = float(line[1])
                        y_center = float(line[2])
                        rect_width = float(line[3])
                        rect_height = float(line[4])
                        x_center_pixel= round(x_center * width)
                        y_center_pixel = round(y_center * height)
                        if x_center_pixel >= j*sliced_width and x_center_pixel <= (j+1)*sliced_width and y_center_pixel >= i*sliced_height and y_center_pixel <= (i+1)*sliced_height:
                            f.write(f"{label} {(x_center_pixel - j*sliced_width)/sliced_width} {(y_center_pixel - i*sliced_height)/sliced_height} {rect_width*(width/sliced_width)} {rect_height*(height/sliced_height)}\n")

import tqdm,os

for t in types:
    image_dir = f"{t}{image_dir_suffix}"
    label_dir = f"{t}{label_dir_suffix}"
    image_files = os.listdir(image_dir)
    label_files = os.listdir(label_dir)
    new_image_files = []
    new_label_files = []
    # delete files that are not in pair
    for f in image_files:
        if os.path.splitext(f)[0] + ".txt" in label_files:
            new_image_files.append(f)
            new_label_files.append(os.path.splitext(f)[0] + ".txt")
    image_files = new_image_files
    label_files = new_label_files

    for i in tqdm.tqdm(range(len(image_files)), desc=f"Processing {t}"):

        slice_image(f"{image_dir}/{image_files[i]}", f"{label_dir}/{label_files[i]}")


