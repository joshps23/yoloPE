from ultralytics import YOLO

import cv2
import numpy as np

def video_classify(path_x):
    video_capture = path_x
    #Create a Webcam Object
    cap=cv2.VideoCapture(video_capture)
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))
    model=YOLO("recycling_class.pt")

    while True:
      success, img = cap.read()
      if not success:
        print("finished running video")
        return
      results = model(path_x)

      names_dict=results[0].names
      probs = results[0].probs.top1
      classification = names_dict[probs]
      label = f'{classification}'
      font=cv2.FONT_HERSHEY_SIMPLEX
      font_scale=20
      font_thickness=20
      text_color=(0,255,0)
      text_color_bg=(0,0,0)
      text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
      text_w, text_h = text_size
      cv2.rectangle(img,((frame_width//2)-(text_w//2),(frame_height//2)-text_h),((frame_width//2)+(text_w//2),(frame_height//2)),text_color_bg,-1)
      cv2.putText(img, label, ((frame_width//2)-(text_w//2),(frame_height//2)),cv2.FONT_HERSHEY_SIMPLEX, 20,[255,255,255], thickness=20,lineType=cv2.LINE_AA)

      yield img