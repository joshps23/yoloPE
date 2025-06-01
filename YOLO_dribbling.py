import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import random
import pathlib
from scipy.signal import find_peaks
from shapely.geometry import LineString, Polygon
from collections import deque, Counter
import time
from ultralytics import YOLO


video_path = 'IMG_0116.MOV'  # or use 0 for webcam
dribble = "-"
shielding_hand_history = deque(maxlen=10)
shield = "-"
next_id = 0
tracked_people = []
iou_threshold = 0.4
defender_center = [0.0,0.0]
body_shield = "-"
body_polygon = []
frame_idx = 1
classNames = ["vball","attacker","defender"
                ]
rescan = False

current_img = None

max_age = 60

class TrackedPerson:
    def __init__(self, person_id, bbox, frame_idx, det=None):
        self.id = person_id
        self.bbox = bbox  # [x_center, y_center, width, height]
        self.last_seen = frame_idx
        self.age = 0  # how many frames it’s been active
        self.missed_frames = 0
        self.righthand_y = []
        self.righthand_bent = False
        self.hand_dir_count = 0
        self.lefthand_y = []
        self.lefthand_dribble = False
        self.elbow_r = []
        self.elbow_l = []
        self.hand_dir_timestamps = []
        self.dribble_status = False
        self.righthand_angle = 0
        self.dribble_non_count = 0
        self.det = det
        self.kpts = None
        self.shielding_angle = 0
        self.shielding = False
        self.dribble_height = ""
        self.role = "Player"
        self.dribbling_hand = None
        self.rolehistory = deque(maxlen=10)
        self.peak_counts = 0
        self.tracked = False
        self.centroid = None
        
        if self.id == 1:

            self.color = (0,255,0)
        else:
            self.color = (255,0,0)
        

    def update(self, bbox, frame_idx, det):
        
        self.bbox = bbox
        self.last_seen = frame_idx
        


        kpts = det[:51].reshape(17, 3)
        self.kpts = kpts
        if self.kpts is None:
            print(f'None kpts detected in frame {frame_idx}')
        dribble_height = 0
        global tracked_people, defender_center, body_shield, body_polygon
        feet = kpts[16][0]
        feet_l = kpts[15][0]
        r_hand = kpts[10][0]
        l_hand = kpts[9][0]
        l_waist = kpts[11][0]
        r_waist = kpts[12][0]
        r_elbow = kpts[8][0]
        l_elbow = kpts[7][0]
        c_lhand = kpts[9][2]
        c_feet = kpts[16][2]
        c_rhand = kpts[10][2]
        c_rwaist = kpts[12][2]
        c_relbow = kpts[8][0]
        c_lelbow = kpts[7][0]

        body_polygon = [(kpts[0][1],kpts[0][0]),(kpts[5][1],kpts[5][0]),(kpts[15][1],kpts[15][0]),(kpts[16][1],kpts[16][0]),(kpts[6][1],kpts[6][0])]
        peaks = []
        
        
        if c_feet < 0.3 or c_rhand < 0.3 or c_rwaist < 0.3 or c_relbow < 0.3 or c_lhand < 0.3 or c_lelbow < 0.3:
            feet = None
            r_elbow = None
            r_hand = None
            r_waist = None
            l_hand = None
            c_lelbow = None
        if feet is not None and r_hand is not None and r_elbow is not None and l_hand is not None and feet_l is not None:

            dribble_height = feet - r_hand
            r_elbow_height = feet - r_elbow
            dribble_height_l = feet_l - l_hand
            l_elbow_height = feet_l - l_elbow
            self.righthand_y.append(dribble_height)
            self.elbow_r.append(r_elbow_height)
            self.lefthand_y.append(dribble_height_l)
            self.elbow_l.append(l_elbow_height)
        # self.righthand_angle = righthand
        self.missed_frames = 0
        self.age += 1
        self.det = det
        self.righthand_y = self.righthand_y[-60:]
        self.elbow_r = self.elbow_r[-60:]
        self.lefthand_y = self.lefthand_y[-60:]
        self.elbow_l = self.elbow_l[-60:]
        global dribble, shield
        if self.age > 90:
            # self.dribble_status = False
            # dribble = "-"
            # shield = "-"
            
            # self.shielding = False
            # self.shielding_angle = 0
            self.age = 0

        # print(f'id {self.id} none count is {self.dribble_non_count}')
        y =  np.array(self.righthand_y)
        elbow_r_y = np.array(self.elbow_r)
        y_l = np.array(self.lefthand_y)
        elbow_l_y = np.array(self.elbow_l)
        peaks, _ = find_peaks(y, prominence=0.02)
        peaks_l, _ = find_peaks(y_l, prominence=0.02)
        self.peak_counts = len(peaks) + len(peaks_l)
        
        # most_active_person = max(tracked_people, key=lambda p: p.peak_counts, default=None)
        # if self is most_active_person and self.peak_counts > 1:
            
        #     self.dribble_status = True
        #     new_role = "Attacker"
        #     # print(f'assigning attacker to id: {self.id}')
        #     self.rolehistory.append(new_role)
        #     for p in tracked_people:
        #         if p != self:
        #             p.rolehistory.append("Defender")

            # if len(tracked_people) > 1:
            #     tracked_people[1-self.id].rolehistory.append("Defender")
        self.dribble_non_count = 0
            
            # attacker = max(tracked_people, key=lambda p: p.rolehistory.count("Attacker"), default=None)
            # if self is attacker and Counter(self.rolehistory).most_common(1)[0][0] == "Attacker" and self.rolehistory.count("Attacker") > 2:
            #     self.role = "Attacker"
            #     for p in tracked_people:
            #         if p != self:
            #             p.role = "Defender"
        for p in tracked_people:
            if p.role == "Attacker":

                if len(peaks)>len(peaks_l):
                    self.dribbling_hand = "Right"
                    self.dribble_status = True
                elif len(peaks_l) > len(peaks):
                    self.dribbling_hand = "Left"
                    self.dribble_status = True

        # if self.righthand_bent == False:
        #     if righthand is not None and righthand < 140.0:
        #         if self.dribble_non_count < 10:
        #             self.righthand_bent = True
        #             self.hand_dir_count += 1
        #             self.hand_dir_timestamps.append(frame_idx)
        #             if len(self.hand_dir_timestamps) > 1:
        #                 rate = (self.hand_dir_timestamps[-1] - self.hand_dir_timestamps[-2])
        #                 if rate < 50:
        #                     self.dribble_status = True
                            
        #                     print('shielding arm is left hand andgle')
        #                 # else:
        #                 #     self.dribble_status = False
        #             self.dribble_non_count = 0
                
        #     elif righthand is None:
        #         self.dribble_non_count += 1
        # if self.righthand_bent == True:
        #     if righthand is not None and righthand > 150:
        #         if self.dribble_non_count < 10:

        #             self.righthand_bent = False
        #             self.hand_dir_count += 1
        #             self.hand_dir_timestamps.append(frame_idx)
        #             if len(self.hand_dir_timestamps) > 1:
        #                 rate = (self.hand_dir_timestamps[-1] - self.hand_dir_timestamps[-2])
        #                 print(f'rate of id: {self.id} is {rate}')
        #                 if rate < 50:
        #                     self.dribble_status = True
                            
                        
        #             self.dribble_non_count = 0
        #         # else:
        #         #     self.dribble_status = False
        #     elif righthand is None:
        #         self.dribble_non_count += 1
        
        # elif is_back_facing(self.kpts) and (c_lhand < 0.3 or c_rhand < 0.3):
        #     new_role = "Attacker"
        #     self.rolehistory.append(new_role)
        #     print(f'back facing id:{self.id}')
            

        if self.dribble_status:
            if self.dribbling_hand == "Right":

                if len(peaks)>1:
                    if y[peaks[-1]] < elbow_r_y[peaks[-1]]:
                        self.dribble_height = 'Good'
                        
                    else:
                        self.dribble_height = "Bounce lower"
            else:
                if len(peaks_l)>1:
                    if y_l[peaks_l[-1]] < elbow_l_y[peaks_l[-1]]:
                        self.dribble_height = 'Good'
                        
                    else:
                        self.dribble_height = "Bounce lower"

            self.shielding_angle = get_arm_body_angle(self.det,self.dribbling_hand)
            if self.shielding_angle is not None and self.shielding_angle > 30:
                # if self.dribble_non_count > 10:
                new_shielding_hand = "Good"
                shielding_hand_history.append(new_shielding_hand)
                
                self.shielding = True
            else:
                new_shielding_hand = "Non-dribbling hands higher"
                shielding_hand_history.append(new_shielding_hand)
                self.shielding = False
            shield = Counter(shielding_hand_history).most_common(1)[0][0]


            if len(tracked_people) > 1 and frame_idx % 3 == 0:
                # defender_kpts = tracked_people[1-self.id].kpts
                # defender_center_x = defender_kpts[0][1]
                # defender_center_y = defender_kpts[12][0]
                # defender_center = (defender_center_y,defender_center_x)
                # check_intersect((kpts[10][0],kpts[10][1]),defender_center,body_polygon)
                if tracked_people[0].kpts is not None and tracked_people[1].kpts is not None:
                    facing, angle1, angle2 = are_facing_each_other(tracked_people[0].kpts,tracked_people[1].kpts)
                    if facing != True:
                        body_shield = "Good"
                    else:
                        body_shield ="Turn shoulder to defender"
                
        if self.dribble_non_count > 20:
            self.dribble_non_count = 0        
            



        
        # print("Peak indices:", peaks)
        # print("Peak values:", y[peaks])
        # print("Troughs", y[troughs])
        
        # print(f'id: {self.id}: {y}')
        # print(f'{self.id} hand dir counts : {self.hand_dir_count}')
        # if len(self.righthand_y) > 2:

        #     if (self.righthand_y[-1] - self.righthand_y[-2]) * (self.righthand_y[-2] - self.righthand_y[-3]) < 0:
        #         print(f'{self.id} is dribbling')
        #         print(f'{self.righthand_y[-1] - self.righthand_y[-2]} and {self.righthand_y[-2] - self.righthand_y[-3]}') 

class YOLOCache:
    def __init__(self, yolo_model):
        self.yolo_model = yolo_model
        self.last_frame_id = None
        self.last_yolo_results = None

    def get_yolo_results(self, frame, frame_id):
        # If we've already processed this frame, return cached result
        if self.last_frame_id == frame_id:
            return self.last_yolo_results

        # Otherwise, run YOLO once and cache the result
        results = self.yolo_model(frame, stream=False)
        self.last_yolo_results = results
        self.last_frame_id = frame_id
        return results

yolo_model=YOLO("lop_8.pt")
yolo_cache = YOLOCache(yolo_model)

# Define COCO keypoint connections
KEYPOINT_EDGES = [
    (0, 1), (1, 3), (0, 2), (2, 4),
    (0, 5), (5, 7), (7, 9),
    (0, 6), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (11, 12)
]

# Load MoveNet MultiPose model from TensorFlow Hub
model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet = model.signatures['serving_default']

# Function to preprocess frames
def preprocess_frame(frame):
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = tf.convert_to_tensor(img_rgb)
    # resized = tf.image.resize_with_pad(img_tensor,160,224)
    resized = tf.image.resize_with_pad(img_tensor,256,384)
    input_tensor = tf.expand_dims(tf.cast(resized, tf.int32), axis=0)
    return input_tensor, h, w



def calculate_iou(box1, box2):
    # box = [x_center, y_center, width, height] in normalized format (0 to 1)
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    # Intersection box
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    inter_w = max(0, xi_max - xi_min)
    inter_h = max(0, yi_max - yi_min)
    intersection = inter_w * inter_h

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def calculate_iou_movenet(box1, box2):
    # box = [x_center, y_center, width, height] in normalized format (0 to 1)
    y1_min = box1[0]
    x1_min = box1[1]
    y1_max = box1[0] + box1[2]
    x1_max = box1[1] + box1[3]

    y2_min = box2[0]
    x2_min = box2[1]
    y2_max = box2[0] + box2[2]
    x2_max = box2[1] + box2[3]

    # Intersection box
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    inter_w = max(0, xi_max - xi_min)
    inter_h = max(0, yi_max - yi_min)
    intersection = inter_w * inter_h

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def loss_fn(centroid1,centroid2):
    x1,y1 = centroid1
    x2,y2 = centroid2
    x_dist = x1 - x2
    y_dist = y1 - y2
    loss = np.sqrt(x_dist**2 + y_dist**2)
    return loss


  # max frames to keep unseen track

def extract_yolo_centroids(results, image_width, image_height):
    yolo_centroids = []

    for r in results:
        boxes = r.boxes
        if boxes is not None and boxes.xyxy.shape[0] > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
            xc = (x1 + x2) / 2 / image_width
            yc = (y1 + y2) / 2 / image_height
            centroids = np.stack([xc, yc, classes], axis=1)
            yolo_centroids.append(centroids)

    if len(yolo_centroids) > 0:
        yolo_centroids = np.vstack(yolo_centroids)  # flatten multiple batches into one array
    else:
        yolo_centroids = np.zeros((0, 3))

    return yolo_centroids


def update_tracker(detections, frame_idx, yolo_centroids):
    global next_id, tracked_people, current_img, rescan
    matches = []
    unmatched_detections = list(range(len(detections)))
    unmatched_tracks = list(range(len(tracked_people)))
    unmatched_bestloss = None
    roles = ["Attacker", "Defender"]
    ids = [-1,1,0]
    rescan = False
    
    if frame_idx % 10 != 0:
        # Match detections to existing tracks
        for d_idx, det in enumerate(detections):
            
            best_iou = 0
            best_loss = 1
            best_track_idx = -1
            best_untracked_bbox = None
            untracked_bbox = get_bbox_for_tracking(det)
            
            ymin,xmin,h,w = untracked_bbox
            center_y=ymin + h/2
            center_x=xmin + w/2
            #centroid for untracked_bbox
            center_xy = (center_x,center_y)
            
            
            for t_idx, track in enumerate(tracked_people):

                tracked_bbox = get_bbox_for_tracking(track.det)
                ymin,xmin,h,w = tracked_bbox
                
                loss = loss_fn(track.centroid,center_xy)

                

                
                # iou = calculate_iou_movenet(tracked_bbox, untracked_bbox)
                # print(f'iou is {iou}')
                

                if loss < best_loss:
                    best_loss = loss
                    best_track_idx = t_idx
                    best_untracked_bbox = untracked_bbox

                # if iou > iou_threshold and iou > best_iou:
                #     best_iou = iou
                #     best_track_idx = t_idx
                #     best_untracked_bbox = untracked_bbox
                
            
            print(f'best loss is {best_loss}')

            if best_track_idx != -1 and best_loss < 0.07:
                # kpts = det[:51].reshape(17, 3)
                # print(f'best_track_idx is {best_track_idx} for d_idx:{d_idx} with loss {best_loss}')
                tracked_people[best_track_idx].update(best_untracked_bbox, frame_idx, det)
                # print(f'updating centroid of {tracked_people[best_track_idx].id} to {center_xy}')
                tracked_people[best_track_idx].centroid = center_xy
                if best_track_idx in unmatched_tracks:
                    unmatched_tracks.remove(best_track_idx)
                else:
                    print('didnt remove best_track_idx from unmatched tracks')
                matches.append((tracked_people[best_track_idx], d_idx))
                unmatched_detections.remove(d_idx)
            else:
                unmatched_bestloss = best_loss

        unmatched_track_counts = len(unmatched_tracks)
        # print(f'left {unmatched_track_counts} tracks')
        
        if unmatched_track_counts > 0 and unmatched_bestloss is not None:
            print('getting yolo for unmatched tracks')
            results = yolo_cache.get_yolo_results(current_img, frame_idx)
            height, width = current_img.shape[:2]
            yolo_centroids = extract_yolo_centroids(results, width, height)         





            for track_idx in unmatched_tracks:
                tracked_id = tracked_people[track_idx].id
                idx = ids.index(tracked_id)
                
                filtered_centroids = yolo_centroids[yolo_centroids[:,2]==idx]
                if filtered_centroids.shape[0] > 0:
                    current_centroid = tracked_people[track_idx].centroid
                    candidate_centroids = filtered_centroids[:, 0:2]
                    distances = np.linalg.norm(candidate_centroids - np.array(current_centroid), axis=1)
                    best_idx = np.argmin(distances)
                    yolo_centroid = filtered_centroids[best_idx]
                    # yolo_centroid = filtered_centroids[0]
                    tracked_people[track_idx].centroid = (yolo_centroid[0],yolo_centroid[1])
                    # print(f'unmatched centroid {tracked_people[track_idx]} assigned to {tracked_id}')
                else:
                    print('no matching centroids found, skipping')
                    continue
    else:
        for track_idx in unmatched_tracks:
                tracked_id = tracked_people[track_idx].id
                idx = ids.index(tracked_id)
                
                filtered_centroids = yolo_centroids[yolo_centroids[:,2]==idx]
                if filtered_centroids.shape[0] > 0:
                    current_centroid = tracked_people[track_idx].centroid
                    candidate_centroids = filtered_centroids[:, 0:2]
                    distances = np.linalg.norm(candidate_centroids - np.array(current_centroid), axis=1)
                    best_idx = np.argmin(distances)
                    yolo_centroid = filtered_centroids[best_idx]
                    # yolo_centroid = filtered_centroids[0]
                    tracked_people[track_idx].centroid = (yolo_centroid[0],yolo_centroid[1])
                    # print(f'unmatched centroid {tracked_people[track_idx]} assigned to {tracked_id}')
                else:
                    print('no matching centroids found, skipping')
                    continue                    
            
            
        
        
        
        
        # else:
        #     print('no match')
        #     results = yolo_model(current_img,stream=False)
        #     for r in results:
        #         boxes = r.boxes
        #         for box in boxes:
        #             cls = int(box.cls[0])
            
        #             for unmatched_track_idx in unmatched_tracks:
        #                 unmatched = tracked_people[unmatched_track_idx]
        #                 idx = classNames.index(unmatched.role.lower())
        #                 if cls != idx:
        #                     print('skipping class not equal to idx')
        #                     continue
        #                 conf=math.ceil((box.conf[0]*100))/100
        #                 if conf < 0.5:
        #                     continue
        #                 x1,y1,x2,y2 = map(int, box.xyxy[0])
        #                 xc = (x1 + x2)//2
        #                 yc = (y1 + y2)//2
        #                 centroid = (xc,yc)
        #                 unmatched.centroid = centroid
        #                 loss = loss_fn(center_xy,unmatched.centroid)
        #                 if loss < 0.2:
        #                     print('updating unmatched')
        #                     unmatched.update(untracked_bbox, frame_idx, det)
        #                     matches.append((unmatched,d_idx))
        #                     unmatched_tracks.remove(unmatched_track_idx)
        #                     unmatched_detections.remove(d_idx)
        #                 else:
        #                     print('unmatched loss > 0.2')


                        
                        
        #                 print(f'no. of unmatched detections left: {len(unmatched_detections)}')


            
            # print('removed d_idx from unmatched_detections')
            # print('removed best_track from unmatched')
    # print(f'number of unmatched tracks: {len(unmatched_tracks)}')
    # for unmatched_track_idx in unmatched_tracks:
    #     unmatched = tracked_people[unmatched_track_idx]
    #     idx = classNames.index(unmatched.role.lower())
    #     results = yolo_model(current_img, stream=False)
    #     for r in results:
    #         boxes = r.boxes
    #         for box in boxes:
    #             cls = int(box.cls[0])
    #             if cls == idx:
    #                 x1, y1, x2, y2 = map(int, box.xyxy[0])
    #                 w = int((x2-x1)//2)
    #                 h = int((y2-y1))
    #                 center_x = int((x1+x2)//2)
    #                 center_y = int((y1+y2)//2)
    #                 yolo_bbox = (center_x,center_y,w,h)
    #                 best_yolo_iou = 0
    #                 best_yolo_idx = -1
    #                 for unmatched_idx in unmatched_detections:
    #                     untracked_bbox = get_bbox_from_keypoints(detections[unmatched_idx])

    #                     iou = calculate_iou(yolo_bbox,untracked_bbox)
    #                 #get most matching bounding box from defender and detections
    #                 #update tracks
    #                     if iou > iou_threshold and iou > best_yolo_iou:
    #                         best_yolo_iou = iou
    #                         best_yolo_idx = unmatched_idx
    #     if best_yolo_idx != -1:
    #         role_idx = roles.index(unmatched.role)
    #         # print(f'best yolo idx: {best_yolo_idx}, unmatched_detections: {unmatched_detections}')
    #         tracked_people[role_idx].update(detections[best_yolo_idx][51:55],frame_idx,detections[best_yolo_idx])
    #         unmatched_tracks.remove(role_idx)
    #         matches.append(tracked_people[role_idx], best_yolo_idx)
    #         unmatched_detections.remove(best_yolo_idx)
                        
                            


    # Create new tracks for unmatched detections
    # for d_idx in unmatched_detections:
    #     # best_u_iou = 0
    #     # best_d_idx = -1
    #     # u_iou = 0
    #     # new_track = TrackedPerson(next_id, detections[d_idx][51:55], frame_idx, detections[d_idx])
    #     # if len(tracked_people) < 2:

    #     #     tracked_people.append(new_track)
    #     #     # print('added new traacker')

    #     #     next_id += 1
    #     # else:
    #     #     # kpts = detections[d_idx][:51].reshape(17, 3)
    #     print("Fallback: Tracking failed, using YOLO to recover roles.")

    #     results = yolo_model(current_img, stream=False)
    #     for r in results:
    #         boxes = r.boxes
    #         for box in boxes:
    #             cls = int(box.cls[0])  # 1 = attacker, 2 = defender
    #             x1, y1, x2, y2 = map(int, box.xyxy[0])
    #             crop = current_img[y1:y2, x1:x2]
    #             input_tensor, _, _ = preprocess_frame(crop)
    #             pose = movenet(input_tensor)['output_0'].numpy()[0]
    #             kpts = denormalize_keypoints(pose, x1, y1, x2 - x1, y2 - y1)

    #             if cls == 1:  # Attacker
    #                 tracked_people[1].update(pose[51:55], frame_idx, pose)
    #                 tracked_people[1].kpts = kpts
    #                 tracked_people[1].role = "Attacker"
    #                 tracked_people[1].color = (0, 255, 0)
    #             elif cls == 2:  # Defender
    #                 tracked_people[0].update(pose[51:55], frame_idx, pose)
    #                 tracked_people[0].kpts = kpts
    #                 tracked_people[0].role = "Defender"
    #                 tracked_people[0].color = (255, 0, 0)
                          


            # oldest_index = max(enumerate(tracked_people), key=lambda x: frame_idx - x[1].last_seen)[0]
            # tracked_people[oldest_index].update(detections[d_idx][51:55], frame_idx, detections[d_idx])
            # print(f'updating oldest index {oldest_index} with unmatched index {d_idx}')
            # unmatched_detections.remove(d_idx)
            # for i in tracked_people:

            #     print(f'length of tracked_people:{len(tracked_people)}')

    # Remove stale tracks
    # for t in tracked_people:
        # current_age = frame_idx-t.last_seen
        # print(f'current age of {t.id} is {current_age}')
    # print(f'previous tracked_people:{len(tracked_people)} in frame:{frame_idx}')
    # tracked_people = [t for t in tracked_people if (frame_idx - t.last_seen) <= max_age]

        # Create new tracks for unmatched detections



    # for d_idx in unmatched_detections:
    #     new_track = TrackedPerson(next_id, detections[d_idx][51:55], frame_idx, detections[d_idx])
    #     if next_id < 2:

    #         tracked_people.append(new_track)
    #         # print('added new traacker')

    #         next_id += 1
    #     else:
    #         # kpts = detections[d_idx][:51].reshape(17, 3)
    #         oldest_index = max(enumerate(tracked_people), key=lambda x: frame_idx - x[1].last_seen)[0]
    #         tracked_people[oldest_index].update(detections[d_idx][51:55], frame_idx, detections[d_idx])
    #         matches.append((tracked_people[oldest_index], d_idx))
    #         unmatched_detections.remove(d_idx)


    return matches

def draw_tracked_keypoints(image, matches, frame_h, frame_w):
    global dribble, shield, defender_center, frame_idx

    # Define color mapping
    color_map = {
        "Good": (0, 255, 0),      # Green
        "Bounce lower": (0, 0, 255),       # Red
        "Non-dribbling hands higher": (0, 0, 255),   # Red
        "Turn shoulder to defender": (0,0,255),  # Red
        "Not Observed": (0,255,255)  
    }
    status = "Not Observed"
    # Get color based on status
    color_dribble = color_map.get(dribble, (0, 255, 255))  # Default to black if status unknown
    # Get color based on status
    color_shield = color_map.get(shield, (0, 255, 255))  # Default to black if status unknown
        # Get color based on status
    color_bodyshield = color_map.get(body_shield, (0, 255, 255))  # Default to black if status unknown
    output = image.copy()
    
        
    for tracked_person, d_idx in matches:
        if tracked_person.dribble_status:
            dribble = tracked_person.dribble_height
            # if tracked_person.shielding:
            #     shield = 'Good'
            # elif tracked_person.shielding == False:
            #     shield = 'Too Low'

        track_id = tracked_person.id
        # person = detections[d_idx]
        # track_id = tracked_people[t_idx].id
        kpts = tracked_person.kpts
        if kpts is None:
            print('kpts is none')
            continue
        for i, (y, x, conf) in enumerate(kpts):
            if conf > 0.3:
                px, py = int(x * frame_w), int(y * frame_h)
                cv2.circle(output, (px, py), 3, tracked_person.color, -1)
        for edge in KEYPOINT_EDGES:
            p1, p2 = edge
            y1, x1, c1 = kpts[p1]
            y2, x2, c2 = kpts[p2]

            if c1 > 0.3 and c2 > 0.3:
                pt1 = int(x1 * frame_w), int(y1 * frame_h)
                pt2 = int(x2 * frame_w), int(y2 * frame_h)
                
                cv2.line(output, pt1, pt2, tracked_person.color, 4)
        # bbox = get_bbox_from_keypoints(tracked_person.det)
        # xmin,ymin,xmax,ymax = to_pixel_bbox(bbox,frame_w,frame_h)
        # cv2.rectangle(output,(xmin,ymin),(xmax,ymax),color=(0,255,0),thickness=2) 
  
        # if tracked_person.role == "Attacker":
        #     output = draw_nose_to_bbox_polygon_from_person(output,tracked_person.det,frame_w,frame_h)
        # Draw ID label at nose
        x_nose = int(kpts[0][1] * frame_w)
        y_nose = int(kpts[0][0] * frame_h)
        cv2.putText(output, f"{tracked_person.id}:{tracked_person.role}", (x_nose, y_nose - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, tracked_person.color, 6)
        # cv2.putText(output, f"Sheilding: {tracked_person.shielding}", (x_nose, y_nose + 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, tracked_person.color, 6)
        # cv2.putText(output, f"Dribble: {tracked_person.dribble_status}", (x_nose, y_nose + 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, tracked_person.color, 6)
        
    # output = draw_rounded_rectangle_alpha(output, (50,30), (600,300),10,(0,128,0),alpha=0.5)
    cv2.putText(output, f"Dribble: ", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
    cv2.putText(output, f"Dribble: ", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    (text_width, _), _ = cv2.getTextSize("Dribble: ", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    
    cv2.putText(output, f"{dribble}", (100 + text_width, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
    cv2.putText(output, f"{dribble}", (100 + text_width, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_dribble, 2)
    
    cv2.putText(output, f"Shielding hand: ", (100, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
    cv2.putText(output, f"Shielding hand: ", (100, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    (text_width, _), _ = cv2.getTextSize("Shielding hand: ", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    
    cv2.putText(output, f"{shield}", (text_width + 100, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
    cv2.putText(output, f"{shield}", (text_width + 100, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_shield, 2)
    
    cv2.putText(output, f"Body Shield: ", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
    cv2.putText(output, f"Body Shield: ", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    (text_width, _), _ = cv2.getTextSize("Body shield: ", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    
    cv2.putText(output, f"{body_shield}", (text_width+100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
    cv2.putText(output, f"{body_shield}", (text_width+100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_bodyshield, 2)

    cv2.putText(output, f"Frame: {frame_idx}", (frame_w - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4)
    return output



def get_right_elbow_angle(person, conf_threshold=0.3):
    """
    Computes the angle at the right elbow (shoulder–elbow–wrist) using MoveNet keypoints.

    Args:
        person (np.ndarray): 1D array of shape (56,) — one person's keypoints from MoveNet.
        conf_threshold (float): Minimum confidence required for all 3 keypoints.

    Returns:
        float or None: Angle in degrees (0–180), or None if confidence is too low.
    """
    kpts = person[:51].reshape(17, 3)

    shoulder = kpts[6][:2]
    elbow = kpts[8][:2]
    wrist = kpts[10][:2]

    c_shoulder = kpts[6][2]
    c_elbow = kpts[8][2]
    c_wrist = kpts[10][2]

    if c_shoulder < conf_threshold or c_elbow < conf_threshold or c_wrist < conf_threshold:
        return None  # skip unreliable measurements

    vec_a = np.array(shoulder) - np.array(elbow)
    vec_b = np.array(wrist) - np.array(elbow)

    cosine_angle = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle_rad)


def get_arm_body_angle(person,hand,conf_threshold=0.3):
    """
    Computes the angle at the right elbow (shoulder–elbow–wrist) using MoveNet keypoints.

    Args:
        person (np.ndarray): 1D array of shape (56,) — one person's keypoints from MoveNet.
        conf_threshold (float): Minimum confidence required for all 3 keypoints.

    Returns:
        float or None: Angle in degrees (0–180), or None if confidence is too low.
    """
    kpts = person[:51].reshape(17, 3)

    shoulder = kpts[5][:2]
    r_shoulder = kpts[6][:2]
    wrist = kpts[9][:2]
    r_wrist = kpts[10][:2]
    waist = kpts[11][:2]
    r_waist = kpts[12][:2]

    c_shoulder = kpts[5][2]
    c_rshoulder = kpts[6][2]
    c_wrist = kpts[9][2]
    c_rwrist = kpts[10][2]
    c_waist = kpts[11][2]
    c_rwaist = kpts[12][2]

    if c_shoulder < conf_threshold or c_wrist < conf_threshold or c_waist < conf_threshold or c_rwrist < conf_threshold or c_rwaist < conf_threshold or c_rshoulder < conf_threshold:
        return None  # skip unreliable measurements

    if hand == "Right":

        vec_a = np.array(wrist) - np.array(shoulder)
        vec_b = np.array(waist) - np.array(shoulder)
    elif hand == "Left":
        vec_a = np.array(r_wrist) - np.array(r_shoulder)
        vec_b = np.array(r_waist) - np.array(r_shoulder)

    cosine_angle = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle_rad)

def check_intersect(hand_cp,defender_cp,attacker_body):
    global body_shield
    # Define your two points
    y1,x1 = hand_cp
    y2,x2 = tuple(defender_cp)
    A = (x1, y1)
    B = (x2, y2)

    # Define the polygon as a list of (x, y) points
    polygon_points = attacker_body  # etc.

    # Create Shapely objects
    line = LineString([A, B])
    polygon = Polygon(polygon_points)

    # Check if line intersects polygon
    intersects = line.intersects(polygon)
    if intersects:
        body_shield += 1
    
        
    
import cv2
import numpy as np

def draw_nose_to_bbox_polygon_from_person(image, det, frame_w, frame_h,
                                          color=(0, 255, 0), alpha=0.4):
    """
    Draws a polygon connecting the nose keypoint to the bottom bounding box corners,
    using normalized bbox coords from person[51:55].

    Parameters:
        image (np.ndarray): Image to draw on.
        nose_kpt (tuple): (y, x, conf) for the nose keypoint (kpts[0]).
        bbox_coords (array-like): [xmin_norm, ymin_norm, width_norm, height_norm] (person[51:55]).
        frame_w (int): Width of the image.
        frame_h (int): Height of the image.
        color (tuple): BGR fill color.
        alpha (float): Transparency for blending.

    Returns:
        np.ndarray: Annotated image.
    """

    kpts = det[:51].reshape(17, 3)
    y_nose, x_nose, conf = kpts[0]
    

    # Scale nose coordinates
    px_nose = int(x_nose * frame_w)
    py_nose = int(y_nose * frame_h)

    # right ankle and left ankle
    y_rankle, x_rankle, c_rankle = kpts[16]
    y_lankle, x_lankle, c_lankle = kpts[15]

    y_rshoulder, x_rshoulder, c_rshoulder = kpts[6]
    y_lshoulder, x_lshoulder, c_lshoulder = kpts[5]

    px_rshoulder = int(x_rshoulder * frame_w)
    py_rshoulder = int(y_rshoulder * frame_h)

    px_lshoulder = int(x_lshoulder * frame_w)
    py_lshoulder = int(y_lshoulder * frame_h)

    px_rankle = int(x_rankle * frame_w)
    py_rankle = int(y_rankle * frame_h)

    px_lankle = int(x_lankle * frame_w)
    py_lankle = int(y_lankle * frame_h)

    

    if conf < 0.3 or c_rankle < 0.3 or c_lankle < 0.3 or c_rshoulder < 0.3 or c_lshoulder < 0.3:
        return image

    # print(f'xmin{xmin} xmax {xmax} ymax {ymax}')
    # Define triangle: nose, bottom-left, bottom-right
    pts = np.array([[(px_nose, py_nose), (px_lshoulder, py_lshoulder), (px_lankle, py_lankle), (px_rankle, py_rankle), (px_rshoulder, py_rshoulder)]], dtype=np.int32)

    # Draw filled polygon with transparency
    overlay = image.copy()
    # cv2.rectangle(overlay, (x_rankle,y_rankle), (x_lankle, y_lankle), (0,255,0), 2)
    cv2.fillPoly(overlay, pts, color)
    output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return output

def get_bbox_from_keypoints(person, confidence_threshold=0.3):
    """
    keypoints: np.array of shape (17, 3), each row is (x, y, confidence)
    Returns: (xmin, ymin, xmax, ymax) as float (normalized)
    """
    # Filter out keypoints below confidence threshold
    keypoints = person[:51].reshape(17, 3)
    valid = keypoints[:, 2] > confidence_threshold
    if not np.any(valid):
        return None  # No valid keypoints

    x_coords = keypoints[valid, 1]
    y_coords = keypoints[valid, 0]

    xmin = float(np.min(x_coords))
    xmax = float(np.max(x_coords))
    ymin = float(np.min(y_coords))
    ymax = float(np.max(y_coords))
    x_center = (xmin + xmax)/2
    y_center = (ymin + ymax)/2
    w = xmax - xmin
    h = ymax - ymin

    return (x_center, y_center, w, h)

def get_bbox_for_tracking(person, confidence_threshold=0.3):
    """
    keypoints: np.array of shape (17, 3), each row is (x, y, confidence)
    Returns: (xmin, ymin, xmax, ymax) as float (normalized)
    """
    # Filter out keypoints below confidence threshold
    keypoints = person[:51].reshape(17, 3)
    valid = keypoints[:, 2] > confidence_threshold
    if not np.any(valid):
        return None  # No valid keypoints

    x_coords = keypoints[valid, 1]
    y_coords = keypoints[valid, 0]

    xmin = float(np.min(x_coords))
    xmax = float(np.max(x_coords))
    ymin = float(np.min(y_coords))
    ymax = float(np.max(y_coords))
    x_center = (xmin + xmax)/2
    y_center = (ymin + ymax)/2
    w = xmax - xmin
    h = ymax - ymin

    return (ymin, xmin, h, w)

def to_pixel_bbox(bbox, image_width, image_height):
    xmin, ymin, xmax, ymax = bbox
    return (
        int(xmin * image_width),
        int(ymin * image_height),
        int(xmax * image_width),
        int(ymax * image_height)
    )

def draw_rounded_rectangle_alpha(img, top_left, bottom_right, radius, color_bgr, alpha):
    overlay = img.copy()

    # Convert to float32 for blending
    overlay = overlay.astype(np.float32)
    img = img.astype(np.float32)

    x1, y1 = top_left
    x2, y2 = bottom_right

    # Draw rectangles
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color_bgr, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color_bgr, -1)

    # Draw rounded corners
    cv2.ellipse(overlay, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color_bgr, -1)
    cv2.ellipse(overlay, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color_bgr, -1)
    cv2.ellipse(overlay, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color_bgr, -1)
    cv2.ellipse(overlay, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color_bgr, -1)

    # Blend overlay with alpha
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)

    return img.astype(np.uint8)



def get_facing_vector(kpts):
    """Returns the facing direction (unit vector) from left to right shoulder."""
    l_shoulder = np.array(kpts[5][:2])
    r_shoulder = np.array(kpts[6][:2])
    facing_vec = r_shoulder - l_shoulder
    norm = np.linalg.norm(facing_vec)
    return facing_vec / norm if norm != 0 else np.zeros(2)

def get_torso_center(kpts):
    """Returns center of torso using mid-point of shoulders."""
    l_shoulder = np.array(kpts[5][:2])
    r_shoulder = np.array(kpts[6][:2])
    return (l_shoulder + r_shoulder) / 2

def angle_between_vectors(v1, v2):
    """Returns angle in degrees between vectors v1 and v2."""
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle_rad)

def are_facing_each_other(kpts1, kpts2, threshold=45):
    # Get facing directions
    face1 = get_facing_vector(kpts1)
    face2 = get_facing_vector(kpts2)

    # Get torso centers
    center1 = get_torso_center(kpts1)
    center2 = get_torso_center(kpts2)

    # Get vectors pointing to each other
    to_2_from_1 = center2 - center1
    to_1_from_2 = center1 - center2

    # Get angles between each person's facing and the direction of the other
    angle1 = angle_between_vectors(face1, to_2_from_1)
    angle2 = angle_between_vectors(face2, to_1_from_2)

    # Facing each other if both angles are near 0° (or ≤ threshold)
    return angle1 < threshold and angle2 < threshold, angle1, angle2

def is_back_facing(kpts, conf_threshold=0.3):
    # Unpack keypoints
    keypoints = kpts.reshape(17, 3)

    # Key indices
    nose = keypoints[0]
    l_shoulder = keypoints[5]
    r_shoulder = keypoints[6]
    l_ear = keypoints[3]
    r_ear = keypoints[4]

    # 1. Nose not visible
    if nose[2] < conf_threshold:
        nose_visible = False
    else:
        nose_visible = True

    # 2. Both ears low confidence (can't see side of face)
    ears_visible = (l_ear[2] > conf_threshold) + (r_ear[2] > conf_threshold)

    # 3. Shoulders are close together (in x direction)
    shoulder_x_diff = abs(l_shoulder[1] - r_shoulder[1])
    shoulder_confidence = l_shoulder[2] > conf_threshold and r_shoulder[2] > conf_threshold

    likely_back = (
        not nose_visible and 
        ears_visible < 2 and 
        shoulder_confidence and 
        shoulder_x_diff < 0.1  # normalized width (adjust as needed)
    )

    return likely_back

def denormalize_keypoints(keypoints, box_xmin, box_ymin, box_width, box_height):
    kpts = keypoints[0][:51].reshape(17, 3)  # 17 keypoints: (y, x, confidence)
    kpts_denorm = []

    for y_norm, x_norm, conf in kpts:
        y_px = box_ymin + y_norm * box_height
        x_px = box_xmin + x_norm * box_width
        kpts_denorm.append((y_px, x_px, conf))

    return np.array(kpts_denorm)

def get_relative_coords(x1,y1,x2,y2,width,height):

    w = int((x2-x1))
    relative_w = w / width
    h = int((y2-y1))
    relative_h = h / height
    xmin = x1 / width
    ymin = y1 / height

    return (ymin,xmin,relative_h,relative_w)




# Open video
def dribbling_pose(path_x, path_dl):
    global frame_idx, next_id, tracked_people, current_img, yolo_cache, rescan
    


    cap = cv2.VideoCapture(path_x)

    tracked_bbox = None  # Store previous frame's bbox

    right_wrist_ys = []  # Track right wrist y-coordinates

    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))
    matches = []
    match_idx = 0
    attacker = None
    defender = None
    results = None
    yolo_centroids = None
    # rescan = False

    # out=cv2.VideoWriter(path_dl, cv2.VideoWriter_fourcc(*'MJPG'), 10, (int(cap.get(3)), int(cap.get(4))))

    
    while cap.isOpened():
        # start_time = time.time()
        # t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            # out.release()
            file_to_rem = pathlib.Path(path_x)
            file_to_rem.unlink()
            return
        
        # t1 = time.time()
        current_img = frame.copy()
        height, width = current_img.shape[:2]
        if frame_idx == 1:
            rescan = True
        if rescan or frame_idx % 10 == 0:
            results = yolo_cache.get_yolo_results(frame, frame_idx)
            yolo_centroids = extract_yolo_centroids(results, width, height)

        if rescan:
            # t2 = time.time()

            # t3 = time.time()
            for r in results:
                boxes=r.boxes
              
                for box in boxes:
                    cls = int(box.cls[0])

                  
                    if cls == 2:
                      
                        x1,y1,x2,y2=box.xyxy[0]
                        x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                      
                        w = int((x2-x1))
                        relative_w = w / width
                        h = int((y2-y1))
                        relative_h = h / height
                        xmin = x1 / width
                        ymin = y1 / height
                        center_x = int((x1+x2)//2)
                        center_xr = center_x/width
                        center_y = int((y1+y2)//2)
                        center_yr = center_y/height
                        crop = frame[y1:y2,x1:x2]
                        input_tensor_d,crop_h,crop_w = preprocess_frame(crop)
                        pose = movenet(input_tensor_d)['output_0'].numpy()[0]
                        result = get_bbox_for_tracking(pose[0])
                        if result is not None:
                            ytrack, xtrack, htrack, wtrack = result
                            # continue processing
                        else:
                            # Handle missing detection safely
                          
                            continue  # or skip frame or assign default box
                        xtrack = (xtrack*w + x1)/width
                        ytrack = (ytrack*h + y1)/height
                        htrack = htrack*h /height
                        wtrack = wtrack*w /width
                      
                        defender = TrackedPerson(0, (ytrack,xtrack,htrack,wtrack), frame_idx, pose[0])
                        defender.role = "Defender"
                        print('Assigning Defender')
                        defender.centroid = (center_xr,center_yr)
                      
                        tracked_people.append(defender)
                        matches.append((defender,match_idx))
                        match_idx += 1
                        
                    elif cls == 1:
                      
                      
                      
                        x1,y1,x2,y2=box.xyxy[0]

                        x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                      
                        w = int((x2-x1))
                        h = int((y2-y1))
                        center_x = int((x1+x2)//2)
                        center_xr = center_x/width
                        center_y = int((y1+y2)//2)
                        center_yr = center_y/height
                        ymin,xmin,relative_h,relative_w = get_relative_coords(x1,y1,x2,y2,width,height)
                        crop = frame[y1:y2,x1:x2]
                        input_tensor_d,crop_h,crop_w = preprocess_frame(crop)
                        pose = movenet(input_tensor_d)['output_0'].numpy()[0]
                        result = get_bbox_for_tracking(pose[0])
                        if result is not None:
                            ytrack, xtrack, htrack, wtrack = result
                            # continue processing
                        else:
                            # Handle missing detection safely
                          
                            continue  # or skip frame or assign default box
                      
                        xtrack = (xtrack*w + x1)/width
                        ytrack = (ytrack*h + y1)/height
                        htrack = htrack*h /height
                        wtrack = wtrack*w /width
                      
                      
                      
                      
                        input_tensor_frame,_,_ = preprocess_frame(frame)
                        pose_frame = movenet(input_tensor_frame)['output_0'].numpy()[0]
                        valid_poses = [p for p in pose_frame if p[55] > 0.3]
                        # t4 = time.time()
                      
                          
                      
                        if attacker != None:
                            continue
                        attacker = TrackedPerson(1, (ytrack,xtrack,htrack,wtrack), frame_idx, pose[0])
                        attacker.role = "Attacker"
                        print('Assigning Attacker')
                        attacker.centroid = (center_xr,center_yr)
                      
                        tracked_people.append(attacker)
                        matches.append((attacker,match_idx))
                        match_idx += 1
                        # t5 = time.time()
                        
                        # t6 = time.time()
                    rescan = False
                    frame = draw_tracked_keypoints(frame, matches, height, width)
        else:

            if attacker is None or defender is None:
                attacker = None
                defender = None
                tracked_people = []
                rescan = True
                frame_idx += 1
                continue

            

            input_tensor, h, w = preprocess_frame(frame)
            # t2 = time.time()
            keypoints = movenet(input_tensor)['output_0'].numpy()[0]
            # t3 = time.time()

            # valid_persons = [p for p in keypoints if p[55] > 0.3]
            keypoints = keypoints[keypoints[:, 55] > 0.3]

            valid_persons = sorted(keypoints, key=lambda x: -x[55])[:2]
            print(f'valid_persons: {len(valid_persons)}')
            # t4 = time.time()
            

            matches = update_tracker(valid_persons, frame_idx, yolo_centroids)
            # t5 = time.time()

            
                
            frame = draw_tracked_keypoints(frame, matches, h, w)
            # t6 = time.time()

            # best_iou = 0
            # selected_person = None

            # for person in valid_persons:
            #     current_bbox = person[51:55]  # [x_center, y_center, width, height]
            #     if tracked_bbox is not None:
            #         iou = calculate_iou(tracked_bbox, current_bbox)
            #         if iou > best_iou:
            #             best_iou = iou
            #             selected_person = person
            #     else:
            #         selected_person = valid_persons[0]
            #         break

            # if selected_person is not None:
            #     tracked_bbox = selected_person[51:55]
            #     kpts = selected_person[:51].reshape(17, 3)
            #     y = kpts[10][1]  # Right wrist y
            #     y_pixel = int(y * h)
            #     right_wrist_ys.append(y_pixel)
            # else:
            #     right_wrist_ys.append(np.nan)


            
                
            

        # out.write(frame)
        # t7 = time.time()

        # Print profiling
        # print(f"""
        # Frame {frame_idx}
        # Read Frame       : {(t1 - t0)*1000:.1f} ms
        # Preprocessing    : {(t2 - t1)*1000:.1f} ms
        # Inference        : {(t3 - t2)*1000:.1f} ms
        # Post-processing  : {(t4 - t3)*1000:.1f} ms
        # Tracking         : {(t5 - t4)*1000:.1f} ms
        # Drawing          : {(t6 - t5)*1000:.1f} ms
        # Write Frame      : {(t7 - t6)*1000:.1f} ms
        # Total Frame Time : {(t7 - start_time)*1000:.1f} ms
        # """)

        
        # print(f'frame:{frame_idx}')
        frame_idx += 1

        yield frame

        # cv2.imshow("MoveNet MultiPose", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    # out.release()
    # cap.release()
    # cv2.destroyAllWindows()


