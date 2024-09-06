from ultralytics import YOLO

import cv2
import numpy as np
import math
import pathlib
# import time

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def lineLineIntersection(A, B, C, D):
    # Line AB represented as a1x + b1y = c1
    a1 = B.y - A.y
    b1 = A.x - B.x
    c1 = a1*(A.x) + b1*(A.y)
 
    # Line CD represented as a2x + b2y = c2
    a2 = D.y - C.y
    b2 = C.x - D.x
    c2 = a2*(C.x) + b2*(C.y)
 
    determinant = a1*b2 - a2*b1
 
    if (determinant == 0):
        # The lines are parallel. This is simplified
        # by returning a pair of FLT_MAX
        return None
    else:
        x = (b2*c1 - b1*c2)/determinant
        y = (a1*c2 - a2*c1)/determinant
        return (x,y)

def check(value,range1,range2):
    range1=float(range1)
    range2=float(range2)
    if range1 <= value <= range2:
        return True
    return False

def lop_detection(path_x, path_dl):
    



    video_capture = path_x
    #Create a Webcam Object
    cap=cv2.VideoCapture(video_capture)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    fps_out = 5

    
    index_in = -1
    index_out = -1
    fps = 1/100
    fps_ms = int(fps * 1000)
    # heatmap_obj = heatmap.Heatmap()
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))

    out=cv2.VideoWriter(path_dl, cv2.VideoWriter_fourcc('M', 'J', 'P','G'), 10, (int(cap.get(3)), int(cap.get(4))))
    classNames = ["vball","attacker","defender"
                ]
    model=YOLO("lop_8.pt")
    # classNames = ['ball']
    success, img = cap.read()

    h,w,c = img.shape

    count=0
    greencount=0
    redcount=0
    while True:
        success_grab = cap.grab()
        if not success_grab: 
            print("finished running video")
            out.release()
            file_to_rem = pathlib.Path(path_x)
            file_to_rem.unlink()
            return
        index_in += 1

        out_due = int(index_in/fps_in*fps_out)
        if out_due>index_out:


            success, img = cap.read()
            if not success:
                print("finished running video")
                out.release()
                file_to_rem = pathlib.Path(path_x)
                file_to_rem.unlink()
                return
            
                
            


            results=model(img,stream=True)
            for r in results:
                boxes=r.boxes
                attacker_midpoint_arr=[]
                defender_line_array=[]
                for box in boxes:
                    x1,y1,x2,y2=box.xyxy[0]
                    x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                    print(x1,y1,x2,y2)
                    w = int(((x2-x1)//2)*0.6)
                    h = int((y2-y1)*0.2)
                    center_x = int((x1+x2)//2)
                    xh1 = center_x-w
                    conf=math.ceil((box.conf[0]*100))/100
                    cls=int(box.cls[0])
                    class_name=classNames[cls]
                    label=f'{class_name}{conf}'
                    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]

                    print(t_size)
                    c2 = x1 + t_size[0], y1 - t_size[1] - 3
                    length=(x2-x1)//2
                    minor_axes=(length)//3
                    alpha=0.2
                    player_alpha=0.2
                    if class_name == 'vball':
                        color=(0, 204, 255)
                    elif class_name == "attacker" and (int(y2-y1)>300):
                        color = (222, 82, 175)
                        
                        attacker_midpoint=(center_x,y2)
                        if conf>0.7:
                          
                          attacker_midpoint_arr.append(attacker_midpoint)
                          attacker_overlay=img.copy()
                          attacker_overlay=cv2.ellipse(img=attacker_overlay, center=attacker_midpoint, axes=(length,minor_axes), angle=0, 
                          startAngle=0, endAngle=360, color=(0,255,0), thickness=-1)
                          img = cv2.addWeighted(attacker_overlay,player_alpha,img,1-player_alpha,0)
                          #cv2.line(img, (x1,y2), (x2,y2), (0,255,0), 30)
                          #cv2.circle(img, attacker_midpoint, 10, color, -1)
                        
                        roi_color = img[y1:y1+h, xh1:xh1+(2*w)]
                        blur_image = cv2.GaussianBlur(roi_color,(51,51),0)
                        img[y1:y1+h, xh1:xh1+(2*w)] = blur_image 

                    elif class_name == "defender":
                        color = (0, 149, 255)
                        deffender_midpoint=(center_x,y2-20)
                        
                        if conf>0.7:
                          defender_line_array=[(x1,y2-60), (x2,y2)]
                        #   defender_overlay=img.copy()
                        #   defender_overlay=cv2.ellipse(img=defender_overlay, center=deffender_midpoint, axes=(length,minor_axes), angle=0, 
                        #   startAngle=0, endAngle=360, color=(0,0,255), thickness=-1)
                        #   img = cv2.addWeighted(defender_overlay,player_alpha,img,1-player_alpha,0)
                          #cv2.line(img, (x1,y2), (x2,y2), (0,0,255), 30)
                          #cv2.circle(img, deffender_midpoint, 10, color, -1)
                        roi_color = img[y1:y1+h, xh1:xh1+(2*w)]
                        blur_image = cv2.GaussianBlur(roi_color,(51,51),0)
                        img[y1:y1+h, xh1:xh1+(2*w)] = blur_image 
                    else:
                        color = (85,45,255)
                   # if conf>0.7:
                        # cv2.rectangle(img, (x1,y1), (x2,y2), color,3)
                        # cv2.rectangle(img, (x1,y1), c2, color, -1, cv2.LINE_AA)  # filled
                        # cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)

                    np_attacker_arr = np.array(attacker_midpoint_arr)
                    # cv2.drawContours(img, [np_attacker_arr], 0, (0,255,0), 4)
                    ind_used=[]
                    for ind in range (0,len(attacker_midpoint_arr)):
                      
                      for ind2 in range (0,len(attacker_midpoint_arr)):
                         if not ind2 == ind and ind2 not in ind_used:
                            overlay=img.copy()
                    #   if 0 <= ind+1 < len(attacker_midpoint_arr) and not defender_line_array == []:
                            if not defender_line_array == []:
                                (xa,ya)=attacker_midpoint_arr[ind]
                                (xb,yb)=attacker_midpoint_arr[ind2]
                                (xc,yc)=defender_line_array[0]
                                (xd,yd)=defender_line_array[1]
                                A=Point(xa,ya)
                                B=Point(xb,yb)
                                C=Point(xc,yc)
                                D=Point(xd,yd)
                                intersect = lineLineIntersection(A,B,C,D)
                                # print(f" intersect is at {intersect}")
                                
                                if not intersect == None:
                                    (xi,yi) = intersect
                                    # img = cv2.circle(img, (int(xi),int(yi)), radius=10, color=(0,0,0), thickness=-1)
                                    
                          
                                    if check(xi,xc,xd) and check(yi,yc,yd):
                                        print(f"INTERSECT: x range is {xa,xb} and xi is {xi}")
                                        print(f"INTERSECT: y range is {ya,yb} and yi is {yi}")
                                        xi = int(xi)
                                        yi = int(yi)
                                        overlay = cv2.line(overlay,(xa,ya),(xb,yb),(0,0,255),50)
                                        img = cv2.addWeighted(overlay,alpha,img,1-alpha,0)
                                        redcount+=1
                                        count+=1
                                        # img = cv2.circle(img, (xi,yi), radius=10, color=(255,0,0), thickness=-1)
                                        # cv2.imwrite("intersect.jpg", img)
                                    else:
                                        overlay = cv2.line(overlay,(xa,ya),(xb,yb),(0,255,0),50)
                                        img = cv2.addWeighted(overlay,alpha,img,1-alpha,0)
                                        greencount+=1
                                        count+=1
                                defender_overlay=img.copy()
                                defender_overlay=cv2.ellipse(img=defender_overlay, center=deffender_midpoint, axes=(length,minor_axes), angle=0, 
                                startAngle=0, endAngle=360, color=(0,0,255), thickness=-1)
                                img = cv2.addWeighted(defender_overlay,player_alpha,img,1-player_alpha,0)
                            else:
                                (xa,ya)=attacker_midpoint_arr[ind]
                                (xb,yb)=attacker_midpoint_arr[ind2]
                                overlay = cv2.line(overlay,(xa,ya),(xb,yb),(0,255,0),50)
                                img = cv2.addWeighted(overlay,alpha,img,1-alpha,0)
                                greencount+=1
                                count+=1
                      ind_used.append(ind)
        label=f'Clear: {greencount/10} Blocked: {redcount/10} Total: {count/10}'
        t_size = cv2.getTextSize(label, 0, fontScale=2, thickness=5)[0]
        text_w, text_h = t_size
        c2 = 120 + text_w, 80 + text_h
        cv2.rectangle(img, (100,50), c2, (0,0,0), -1, cv2.LINE_AA)
        cv2.putText(img, label, (100,65 + text_h),cv2.FONT_HERSHEY_SIMPLEX, 2,[0,255,0], thickness=5,lineType=cv2.LINE_AA)
        out.write(img)
        yield img
        
        # cv2.imshow("image", frame_with_heatmap)
        # if cv2.waitKey(1) & 0xFF==ord('1'):
            # break
    
# cv2.destroyAllWindows()


# video_detection('static/files/IMG_0679 3.MOV',"space")