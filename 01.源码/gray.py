import cv2
import os
import tqdm

for dir in os.listdir('images'):
    for file in tqdm.tqdm(os.listdir('images/' + dir), desc=dir):
        img = cv2.imread('images/' + dir + '/' + file, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite('images/' + dir + '/' + file, img)