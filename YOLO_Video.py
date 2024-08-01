from ultralytics import YOLO, solutions

import cv2
import math

def video_detection(path_x):
    video_capture = path_x
    #Create a Webcam Object
    cap=cv2.VideoCapture(video_capture)
    # heatmap_obj = heatmap.Heatmap()
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))
    #out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P','G'), 10, (frame_width, frame_height))
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
    
    model=YOLO("yolov8n.pt")
    # classNames = ['ball']
    success, img = cap.read()
    h,w,c = img.shape
    heatmap_obj=solutions.Heatmap(colormap=cv2.COLORMAP_PARULA,
    imw=int(w),
    imh=int(h),
    view_img=False,
    shape="circle",
    classes_names=model.names
    )
    while True:
        success, img = cap.read()
        results=model(img,stream=True)
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                print(x1,y1,x2,y2)
                conf=math.ceil((box.conf[0]*100))/100
                cls=int(box.cls[0])
                class_name=classNames[cls]
                # label=f'{class_name}{conf}'
                # t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                # print(t_size)
                # c2 = x1 + t_size[0], y1 - t_size[1] - 3
                # if class_name == 'Dust Mask':
                #     color=(0, 204, 255)
                # elif class_name == "Glove":
                #     color = (222, 82, 175)
                # elif class_name == "Protective Helmet":
                #     color = (0, 149, 255)
                # else:
                #     color = (85,45,255)
                # if conf>0.5:
                #     cv2.rectangle(img, (x1,y1), (x2,y2), color,3)
                #     cv2.rectangle(img, (x1,y1), c2, color, -1, cv2.LINE_AA)  # filled
                #     cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)
        track=model.track(img)
        final_img=heatmap_obj.generate_heatmap(img,track)
        yield final_img
        #out.write(img)
        #cv2.imshow("image", img)
        #if cv2.waitKey(1) & 0xFF==ord('1'):
            #break
    #out.release()
cv2.destroyAllWindows()
