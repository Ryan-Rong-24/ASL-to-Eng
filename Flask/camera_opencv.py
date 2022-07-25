import os
import cv2
from base_camera import BaseCamera
from ASLtoEngNet import ASLtoEngNet
import numpy as np
net = ASLtoEngNet(model_pb_path='model_info/asltoeng.onnx', label_path='model_info/actions_list.npy', min_detection_confidence=0.5, min_tracking_confidence=0.5)

colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num%3], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        sequence = []
        sentence = []
        predictions = []
        threshold = 0.5
        sequence_length = 30

        while True:
            # read current frame
            _, img = camera.read()

            image, keypoints = net.detect(img)

            sequence.append(keypoints)
            sequence = sequence[-sequence_length:]
            
            if len(sequence) == sequence_length:
                res = net.predict(np.expand_dims(sequence, axis=0))[0]
                print(net.classes[np.argmax(res)])
                predictions.append(np.argmax(res))
                
                
                #3. Viz logic
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        
                        if len(sentence) > 0: 
                            if net.classes[np.argmax(res)] != sentence[-1]:
                                sentence.append(net.classes[np.argmax(res)])
                        else:
                            sentence.append(net.classes[np.argmax(res)])

                if len(sentence) > 4: 
                    sentence = sentence[-4:]

                # Viz probabilities
                # image = prob_viz(res, net.classes, image, colors)
                
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', image)[1].tobytes()
