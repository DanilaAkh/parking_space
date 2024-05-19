import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
import json
from ultralytics import YOLO
#load nano detection model
model = YOLO("yolov8n.pt")


def segment_pic(file):
    #load image
    image = cv2.imread(f"{file}")
    objects = model(image, save=True)
    
    for result in objects:
        boxes = result.boxes
        for box in boxes:
            cls = box.cls
            if len(cls) > 0:
                # Получение координат
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Постороение прямоугольника
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                # Текст к прямоугольнику
                text = "car"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 4
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = int(x1 + 5)
                text_y = int(y1 + text_size[1] + 5)
                cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)

                plt.figure(figsize=(10,10))
                plt.imshow(image)
                plt.axis("off")
                plt.savefig(f"./out_file.png")
                #plt.show()


segment_pic("./autopark.png")