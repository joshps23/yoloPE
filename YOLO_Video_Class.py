from ultralytics import YOLO

import cv2
import numpy as np
import math
import time

def video_classify(path_x):
    video_capture = path_x
    #Create a Webcam Object
    cap=cv2.VideoCapture(video_capture)
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))
    model=YOLO("waste_management4.pt")
    classNames = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    count=0
    while True:
      count+=1
      success, img = cap.read()
      if not success:
        print("finished running video")
        return
      results = model(img)
      for r in results:
        boxes = r.boxes
        for box in boxes:
          x1,y1,x2,y2=box.xyxy[0]
          x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
          cls=int(box.cls[0])
          
          class_name=classNames[cls]
          print(class_name)
          if count == 1:
            font_size = 10
            font_thickness = 5
          else:
            font_size = 1
            font_thickness = 2
          label=f'{class_name}'
          t_size = cv2.getTextSize(label, 0, fontScale=font_size, thickness=font_thickness)[0]
          text_w, text_h = t_size
          c2 = x2, y2 - t_size[1] - 3
          cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255),3)
          cv2.rectangle(img, (x2-text_w,y2), c2, (0,0,255), -1, cv2.LINE_AA)  # filled
          cv2.putText(img, label, (x2-text_w,y2),0, font_size,[255,255,255], thickness=font_thickness,lineType=cv2.LINE_AA)


      yield img