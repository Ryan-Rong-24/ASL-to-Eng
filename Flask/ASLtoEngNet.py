import cv2
import numpy as np
import argparse
import onnxruntime as ort
import math
import time
from functools import wraps
import mediapipe as mp

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # drawing utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False
    results = model. process(image) # Make prediction
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Time classification
def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        ret = func(*args, **kwargs)
        end = time.perf_counter()
        # print('used:', end - start)
        return ret
    return wrapper

class ASLtoEngNet():
    def __init__(self, model_pb_path, label_path, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.classes = np.load(label_path,allow_pickle=True).tolist()
        self.num_classes = len(self.classes)
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.net = ort.InferenceSession(model_pb_path, so)
    
    @timeit
    def detect(self, srcimg):
        with mp_holistic.Holistic(min_detection_confidence=self.min_detection_confidence, min_tracking_confidence=self.min_tracking_confidence) as holistic:
            image, results = mediapipe_detection(srcimg, holistic)
            draw_styled_landmarks(image, results)
            keypoints = extract_keypoints(results)

        return image, keypoints

    def predict(self, data):
        return self.net.predict(data)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='model_info/person.jpg', help="image path")
    parser.add_argument('--modelpath', type=str, default='model_info/asltoeng.onnx', help="onnx filepath")
    parser.add_argument('--classfile', type=str, default='model_info/actions_list.npy', help="classname filepath")
    parser.add_argument('--dectThreshold', default=0.5, type=float, help='mediapipe detection threshold')
    parser.add_argument('--tracThreshold', default=0.6, type=float, help='mediapipe track threshold')
    args = parser.parse_args()

    srcimg = cv2.imread(args.imgpath)
    net = ASLtoEngNet(args.modelpath, args.classfile, min_detection_confidence=args.dectThreshold, min_tracking_confidence=args.tracThreshold)
    srcimg, _ = net.detect(srcimg)

    winName = 'Deep learning sign detection in onnxruntime'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()