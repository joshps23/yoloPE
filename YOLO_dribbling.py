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


video_path = 'IMG_3438.MOV'  # or use 0 for webcam
dribble = "-"
shielding_hand_history = deque(maxlen=10)
shield = "-"
next_id = 0
tracked_people = []
iou_threshold = 0.5
defender_center = [0.0,0.0]
body_shield = "-"
body_polygon = []
frame_idx = 1


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
        
        if self.id == 1:

            self.color = (0,255,0)
        else:
            self.color = (255,0,0)
        

    def update(self, bbox, frame_idx, det):
        
        self.bbox = bbox
        self.last_seen = frame_idx
        kpts = det[:51].reshape(17, 3)
        self.kpts = kpts
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
            self.dribble_status = False
            dribble = "-"
            shield = "-"
            
            self.shielding = False
            self.shielding_angle = 0
            self.age = 0

        # print(f'id {self.id} none count is {self.dribble_non_count}')
        y =  np.array(self.righthand_y)
        elbow_r_y = np.array(self.elbow_r)
        y_l = np.array(self.lefthand_y)
        elbow_l_y = np.array(self.elbow_l)
        peaks, _ = find_peaks(y, prominence=0.05)
        peaks_l, _ = find_peaks(y_l, prominence=0.05)
        self.peak_counts = len(peaks) + len(peaks_l)
        
        most_active_person = max(tracked_people, key=lambda p: p.peak_counts, default=None)
        if self is most_active_person and self.peak_counts > 1:
            
            self.dribble_status = True
            new_role = "Attacker"
            
            self.rolehistory.append(new_role)
            for p in tracked_people:
                if p != self:
                    p.rolehistory.append("Defender")

            # if len(tracked_people) > 1:
            #     tracked_people[1-self.id].rolehistory.append("Defender")
            self.dribble_non_count = 0
            
            attacker = max(tracked_people, key=lambda p: p.rolehistory.count("Attacker"), default=None)
            if self is attacker and Counter(self.rolehistory).most_common(1)[0][0] == "Attacker" and self.rolehistory.count("Attacker") > 2:
                self.role = "Attacker"
                for p in tracked_people:
                    if p != self:
                        p.role = "Defender"
            if len(peaks)>1:
                self.dribbling_hand = "Right"
            else:
                self.dribbling_hand = "Left"

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
                facing, angle1, angle2 = are_facing_each_other(tracked_people[0].kpts,tracked_people[1].kpts)
                if facing != True:
                    body_shield = "Good"
                else:
                    body_shield ="Turn shoulder to defender"
                
        if self.dribble_non_count > 20:
            self.dribble_non_count = 0        
            



        troughs, _ = find_peaks(-y)
        # print("Peak indices:", peaks)
        # print("Peak values:", y[peaks])
        # print("Troughs", y[troughs])
        
        # print(f'id: {self.id}: {y}')
        # print(f'{self.id} hand dir counts : {self.hand_dir_count}')
        # if len(self.righthand_y) > 2:

        #     if (self.righthand_y[-1] - self.righthand_y[-2]) * (self.righthand_y[-2] - self.righthand_y[-3]) < 0:
        #         print(f'{self.id} is dribbling')
        #         print(f'{self.righthand_y[-1] - self.righthand_y[-2]} and {self.righthand_y[-2] - self.righthand_y[-3]}') 

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
    resized = tf.image.resize_with_pad(img_tensor,160,224)
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


  # max frames to keep unseen track

def update_tracker(detections, frame_idx):
    global next_id, tracked_people
    matches = []
    unmatched_detections = list(range(len(detections)))
    unmatched_tracks = list(range(len(tracked_people)))

    # Match detections to existing tracks
    for d_idx, det in enumerate(detections):
        best_iou = 0
        best_track_idx = -1
        for t_idx, track in enumerate(tracked_people):

            tracked_bbox = get_bbox_from_keypoints(track.det)
            untracked_bbox = get_bbox_from_keypoints(det)
            iou = calculate_iou(tracked_bbox, untracked_bbox)
            if iou > iou_threshold and iou > best_iou:
                best_iou = iou
                best_track_idx = t_idx

        if best_track_idx != -1:
            kpts = det[:51].reshape(17, 3)
            tracked_people[best_track_idx].update(det[51:55], frame_idx, det)
            matches.append((tracked_people[best_track_idx], d_idx))
            
        

            unmatched_detections.remove(d_idx)
            # print('removed d_idx from unmatched_detections')
        if best_track_idx in unmatched_tracks:
            unmatched_tracks.remove(best_track_idx)
            # print('removed best_track from unmatched')

    # Create new tracks for unmatched detections
    for d_idx in unmatched_detections:
        new_track = TrackedPerson(next_id, detections[d_idx][51:55], frame_idx, detections[d_idx])
        if next_id < 2:

            tracked_people.append(new_track)
            # print('added new traacker')

            next_id += 1
        else:
            kpts = detections[d_idx][:51].reshape(17, 3)
            oldest_index = max(enumerate(tracked_people), key=lambda x: frame_idx - x[1].last_seen)[0]
            tracked_people[oldest_index].update(detections[d_idx][51:55], frame_idx, detections[d_idx])
            unmatched_detections.remove(d_idx)
            # for i in tracked_people:

            #     print(f'length of tracked_people:{len(tracked_people)}')

    # Remove stale tracks
    # for t in tracked_people:
        # current_age = frame_idx-t.last_seen
        # print(f'current age of {t.id} is {current_age}')
    # print(f'previous tracked_people:{len(tracked_people)} in frame:{frame_idx}')
    # tracked_people = [t for t in tracked_people if (frame_idx - t.last_seen) <= max_age]

    return matches

def draw_tracked_keypoints(image, detections, matches, tracked_people, frame_h, frame_w):
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
        person = detections[d_idx]
        # track_id = tracked_people[t_idx].id
        kpts = tracked_person.kpts
        for i, (y, x, conf) in enumerate(kpts):
            if conf > 0.3:
                px, py = int(x * frame_w), int(y * frame_h)
                cv2.circle(output, (px, py), 3, (0, 255, 0), -1)
        for edge in KEYPOINT_EDGES:
            p1, p2 = edge
            y1, x1, c1 = kpts[p1]
            y2, x2, c2 = kpts[p2]

            if c1 > 0.3 and c2 > 0.3:
                pt1 = int(x1 * frame_w), int(y1 * frame_h)
                pt2 = int(x2 * frame_w), int(y2 * frame_h)
                
                cv2.line(output, pt1, pt2, (0, 255, 0), 4)
        # bbox = get_bbox_from_keypoints(tracked_person.det)
        # xmin,ymin,xmax,ymax = to_pixel_bbox(bbox,frame_w,frame_h)
        # cv2.rectangle(output,(xmin,ymin),(xmax,ymax),color=(0,255,0),thickness=2) 
  
        # if tracked_person.role == "Attacker":
        #     output = draw_nose_to_bbox_polygon_from_person(output,tracked_person.det,frame_w,frame_h)
        # Draw ID label at nose
        x_nose = int(kpts[0][1] * frame_w)
        y_nose = int(kpts[0][0] * frame_h)
        cv2.putText(output, f"{tracked_person.role}", (x_nose, y_nose - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, tracked_person.color, 6)
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

    return (xmin, ymin, xmax, ymax)

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





# Open video
def dribbling_pose(path_x, path_dl):
  global frame_idx  
  cap = cv2.VideoCapture(path_x)

  tracked_bbox = None  # Store previous frame's bbox

  right_wrist_ys = []  # Track right wrist y-coordinates

  frame_width=int(cap.get(3))
  frame_height=int(cap.get(4))

#   out=cv2.VideoWriter(path_dl, cv2.VideoWriter_fourcc(*'MJPG'), 10, (int(cap.get(3)), int(cap.get(4))))

  
  while cap.isOpened():
    #   start_time = time.time()
    #   t0 = time.time()
      ret, frame = cap.read()
      if not ret:
        #   out.release()
          file_to_rem = pathlib.Path(path_x)
          file_to_rem.unlink()
          return
    #   t1 = time.time()


      input_tensor, h, w = preprocess_frame(frame)
    #   t2 = time.time()
      keypoints = movenet(input_tensor)['output_0'].numpy()[0]
    #   t3 = time.time()

      valid_persons = [p for p in keypoints if p[55] > 0.3]
      valid_persons = sorted(valid_persons, key=lambda x: -x[55])[:2]
    #   t4 = time.time()
      

      matches = update_tracker(valid_persons, frame_idx)
    #   t5 = time.time()

      
          
      frame = draw_tracked_keypoints(frame, valid_persons, matches, tracked_people, h, w)
    #   t6 = time.time()

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


      
          
      

    #   out.write(frame)
    #   t7 = time.time()

      # Print profiling
    #   print(f"""
    #   Frame {frame_idx}
    #   Read Frame       : {(t1 - t0)*1000:.1f} ms
    #   Preprocessing    : {(t2 - t1)*1000:.1f} ms
    #   Inference        : {(t3 - t2)*1000:.1f} ms
    #   Post-processing  : {(t4 - t3)*1000:.1f} ms
    #   Tracking         : {(t5 - t4)*1000:.1f} ms
    #   Drawing          : {(t6 - t5)*1000:.1f} ms
    #   Write Frame      : {(t7 - t6)*1000:.1f} ms
    #   Total Frame Time : {(t7 - start_time)*1000:.1f} ms
    #   """)

      # frame = draw_keypoints(frame, keypoints, h, w)
      frame_idx += 1

      yield frame

      # cv2.imshow("MoveNet MultiPose", frame)
      # if cv2.waitKey(1) & 0xFF == ord('q'):
      #     break

  # cap.release()
  # cv2.destroyAllWindows()
