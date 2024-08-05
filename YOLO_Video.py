from ultralytics import YOLO

import cv2
import numpy as np
import math
import time


def video_detection(path_x, mode, path_dl):
    



    video_capture = path_x
    #Create a Webcam Object
    cap=cv2.VideoCapture(video_capture)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    fps_out = 10

    classes_for_heatmap = [0]
    index_in = -1
    index_out = -1
    fps = 1/100
    fps_ms = int(fps * 1000)
    # heatmap_obj = heatmap.Heatmap()
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))
    label=f'Green pixels'
    out=cv2.VideoWriter(path_dl, cv2.VideoWriter_fourcc('M', 'J', 'P','G'), 10, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    if mode == "space":
        model=YOLO("yolov8n.pt")
        classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                    "teddy bear", "hair drier", "toothbrush"
                    ]
        class_to_detect = "person"
    elif mode == "ball":
        model=YOLO("ballDetectBestV2.pt")
        classNames = ["ball"]
        class_to_detect = "ball"
    
    # classNames = ['ball']
    success, img = cap.read()

    h,w,c = img.shape
    heatmap = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))), dtype=np.float32)
    # heatmap_obj=solutions.Heatmap(
    # heatmap_alpha=0.5,
    # # colormap=cv2.COLORMAP_PARULA,
    # imw=int(w),
    # imh=int(h),
    # view_img=False,
    # shape="rect",
    # classes_names=model.names,
    # decay_factor=0.99,
    # )
    count=0
    while True:
        success_grab = cap.grab()
        if not success_grab: 
            print("finished running video")
            return
        index_in += 1

        out_due = int(index_in/fps_in*fps_out)
        if out_due>index_out:


            success, img = cap.read()
            
                
            


            results=model(img,stream=True)
            for r in results:
                boxes=r.boxes
                for box in boxes:
                    x1,y1,x2,y2=box.xyxy[0]
                    x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    conf=math.ceil((box.conf[0]*100))/100
                    cls=int(box.cls[0])
                    class_name=classNames[cls]
                    # label=f'{class_name}{conf}'
                    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=5)[0]
                    # print(t_size)
                    c2 = x1 + t_size[0], y1 - t_size[1] - 3
                    if class_name == class_to_detect and conf > 0.5:
                        if (class_to_detect == "person") and (y2-y1>200):
                            print(f"person size: {y2-y1}")
                            heatmap[y2-20:y2, center_x-10:center_x+10] += 20
                        elif class_to_detect == "ball":
                            heatmap[center_y-25:center_y+25, center_x-25:center_x+25] += 20
            # track=model.track(img,persist=True,classes=classes_for_heatmap)
            # final_img=heatmap_obj.generate_heatmap(img,track)
            # img=final_img
            # time.sleep(fps)
        cv2.putText(img, label, (100,100),cv2.FONT_HERSHEY_SIMPLEX, 2,[255,255,255], thickness=5,lineType=cv2.LINE_AA)
        # normalized_heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        colored_map = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_HSV)
        
        frame_with_heatmap = cv2.addWeighted(img, 0.5, colored_map, 0.5, 0)
        hsv = cv2.cvtColor(frame_with_heatmap, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        lower = (20,100,100)
        upper = (60,255,255)
        mask = cv2.inRange(hsv, lower, upper)
        count_green = np.count_nonzero(mask)
        space_green = (count_green // 1300)
        if class_to_detect == "person":
            label = f'Space Coverage: {space_green}'
        elif class_to_detect == "ball":
            label = f'Ball Coverage: {count_green//1600}'

        out.write(frame_with_heatmap)
        yield frame_with_heatmap
        
        # cv2.imshow("image", frame_with_heatmap)
        # if cv2.waitKey(1) & 0xFF==ord('1'):
            # break
    out.release()
# cv2.destroyAllWindows()


# video_detection('static/files/IMG_0679 3.MOV',"space")