# Import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
import streamlit as st
import requests

global name, course, group, module, duration, matric_id

def main():
    # Get the frames per second of the video device
    fps = getFPS()                      
    EYE_AR_THRESH = 1                   # Threshold for which the eye aspect ratio is counted as disengaged
    EYE_AR_CONSEC_FRAMES = 2 * fps      # Number of consecutive frames before user is counted as disengaged

    COUNTER = 0         
    TOTAL = 0                           # Total number of frames counted as disengaged

    print("Initiating facial landmark predictor...")  # For debug purpose
    detector = dlib.get_frontal_face_detector()      # dlib's face detector (HOG-based)
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Facial landmark predictor

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]   # Facial landmark index for left eye
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]  # Facial landmark index for right eye

    print("Initiating video stream thread...")
    vs = VideoStream(src=0).start()  # Start video stream thread
    time.sleep(1.0)  # Allow time for the camera to warm up

    # Check if the video stream is opened
    if vs is None or not vs.isOpened():
        print("Error: Could not open video stream.")
        return  # Exit the main function if the video stream cannot be opened

    _sum = 0                            # Sum variable for initial calibration for EAR threshold
    _counter = int(5 * fps)             # Number of frames for calibration (5 seconds) 
    disengaged = False                  # Initiate engagement state to be 'engaged'
    LOOKDOWN_COUNTER = 0                # Counts consecutive frames where eyes cannot be detected
    start = 0                           # Time since epoch for when calibration completes (and recording starts)
    engaged_status = []                 # List of binary classification of engagement status

    # Iterate through all frames until video stops              
    while True:
        # Grab the frame from the threaded video file stream
        frame = vs.read()
        
        # Check if the frame is None
        if frame is None:
            print("Error: Could not read frame from video stream.")
            continue  # Skip this iteration if the frame is None

        frame = imutils.resize(frame, width=450)  # Resize the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        rects = detector(gray, 0)  # Detect faces in the grayscale frame
        
        if len(rects) != 0:
            LOOKDOWN_COUNTER = 0
            shape = predictor(gray, rects[0])  # Determine facial landmarks
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0  # Average the eye aspect ratio

            # Run this section if calibration has not been done
            if EYE_AR_THRESH == 1:
                if _counter > 0:
                    _sum += ear
                    _counter -= 1
                else:
                    EYE_AR_THRESH = _sum / int(5 * fps) * 0.9
                    start = int(time.time())
            
            # Only run this section once calibration completes
            if _counter == 0:       
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    disengaged = True
                    TOTAL += 1
                    
                    if ear >= EYE_AR_THRESH:
                        COUNTER = 0
                    else:
                        COUNTER += 1
                elif COUNTER < EYE_AR_CONSEC_FRAMES:
                    disengaged = False
                    if ear < EYE_AR_THRESH:
                        COUNTER += 1
                    else:
                        COUNTER = 0
               
                if disengaged:
                    engaged_status.append(0)  # 0 as disengaged
                    cv2.putText(frame, "Disengaged", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    engaged_status.append(1)  # 1 as engaged
                    cv2.putText(frame, "Engaged", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, "Total: {:.2f}".format(TOTAL/fps), (300, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        elif EYE_AR_THRESH != 1:
            LOOKDOWN_COUNTER += 1
            ear = 0
            
            if LOOKDOWN_COUNTER >= EYE_AR:
