import tqdm
import xml
import cv2
import os

save_path = "Spurious_copper_Img_with_rects"
image_path = "Spurious_copper_Img"
label_path = "Spurious_copper_txt"

use_xml=False

os.makedirs(save_path, exist_ok=True)

file = ""
desc = ""

for file in tqdm.tqdm(os.listdir(image_path),f"Drawing labels on images... {file} {desc}"):
    file_without_ext = os.path.splitext(file)[0]
    if file.endswith(".jpg") or file.endswith(".bmp"):
        image = cv2.imread(os.path.join(image_path, file))
        # xml_file = file.replace(".jpg", ".xml")
        # xml_file = os.path.join(label_path, xml_file)
        # with open(xml_file, "r") as f:
        #     labels =
        if use_xml:
            pass
        else:
            # open txt file
            if not os.path.exists(os.path.join(label_path,file_without_ext+".txt")):
                desc = "---FAIL---"
                continue
            with open(os.path.join(label_path, file_without_ext+".txt"), "r") as f:
                labels = f.readlines()
                labels = [label.strip().split(" ") for label in labels]
                labels = [[label[0],float(label[1]), float(label[2]), float(label[3]), float(label[4])] for label in labels]
                for label in labels:
                    text,center_x_proportion, center_y_proportion, width_proportion, height_proportion = label
                    center_x = int(center_x_proportion * image.shape[1])
                    center_y = int(center_y_proportion * image.shape[0])
                    width = int(width_proportion * image.shape[1])
                    height = int(height_proportion * image.shape[0])
                    # draw red rectangle
                    cv2.rectangle(image, (center_x - width//2, center_y - height//2), (center_x + width//2, center_y + height//2), (0, 0, 255), 2)
                    # draw text at top left corner
                    cv2.putText(image, text, (center_x - width//2, center_y - height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                desc = "-SUCCCESS-"
        cv2.imwrite(os.path.join(save_path, file), image)