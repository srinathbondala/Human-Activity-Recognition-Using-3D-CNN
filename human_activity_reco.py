# # USAGE
# # python human_activity_reco.py --model resnet-34_kinetics.onnx --classes action_recognition_kinetics.txt --input example_activities.mp4
# # python human_activity_reco.py --model resnet-34_kinetics.onnx --classes action_recognition_kinetics.txt

# import numpy as np
# import argparse
# import imutils
# import sys
# import cv2

# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True,
# 	help="path to trained human activity recognition model")
# ap.add_argument("-c", "--classes", required=True,
# 	help="path to class labels file")
# ap.add_argument("-i", "--input", type=str, default="",
# 	help="optional path to video file")
# args = vars(ap.parse_args())


# CLASSES = open(args["classes"]).read().strip().split("\n")
# SAMPLE_DURATION = 16
# SAMPLE_SIZE = 112

# print("[INFO] loading human activity recognition model...")
# net = cv2.dnn.readNet(args["model"])

# print("[INFO] accessing video stream...")
# vs = cv2.VideoCapture(args["input"] if args["input"] else 0)

# while True:
	
# 	frames = []

	
# 	for i in range(0, SAMPLE_DURATION):
# 		(grabbed, frame) = vs.read()

		
# 		if not grabbed:
# 			print("[INFO] no frame read from stream - exiting")
# 			sys.exit(0)

		
# 		frame = imutils.resize(frame, width=400)
# 		frames.append(frame)

# 	blob = cv2.dnn.blobFromImages(frames, 1.0,
# 		(SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
# 		swapRB=True, crop=True)
# 	blob = np.transpose(blob, (1, 0, 2, 3))
# 	blob = np.expand_dims(blob, axis=0)

	
# 	net.setInput(blob)
# 	outputs = net.forward()
# 	label = CLASSES[np.argmax(outputs)]

# 	for frame in frames:
# 		cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
# 		cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
# 			0.8, (255, 255, 255), 2)

# 		cv2.imshow("Activity Recognition", frame)
# 		key = cv2.waitKey(1) & 0xFF

# 		if key == ord("q"):
# 			break
import streamlit as st
from collections import deque
import numpy as np
import cv2
import imutils
import tempfile
import os
import time

# Load the action recognition model
@st.cache_resource
def load_action_recognition_model(model_path):
    net = cv2.dnn.readNet(model_path)
    return net

def recognize_action(frames, net, classes):
    blob = cv2.dnn.blobFromImages(frames, 1.0, (112, 112), (114.7748, 107.7354, 99.4750), swapRB=True, crop=True)
    blob = np.transpose(blob, (1, 0, 2, 3))
    blob = np.expand_dims(blob, axis=0)
    
    net.setInput(blob)
    outputs = net.forward()
    return classes[np.argmax(outputs)]

def main():
    st.title("Human Activity Recognition")
    
    model_path = "resnet-34_kinetics.onnx"  # Change this to your model path
    classes_path = "action_recognition_kinetics.txt"  # Change this to your classes path

    # Load models
    action_net = load_action_recognition_model(model_path)
    classes = open(classes_path).read().strip().split("\n")
    
    SAMPLE_DURATION = 16
    
    # Upload video file or use webcam
    input_source = st.selectbox("Select Input Source:", ["Webcam", "Upload Video"])
    
    if input_source == "Upload Video":
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
        if uploaded_file is not None:
            # Save uploaded video temporarily
            temp_file_path = tempfile.mktemp(suffix=".mp4")
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.read())
            st.video(temp_file_path)

            # Process the video
            process_video(temp_file_path, action_net, classes, SAMPLE_DURATION)
            os.remove(temp_file_path)  # Remove the temporary file

    elif input_source == "Webcam":
        st.write("Press the button to start the webcam.")
        if st.button("Start Webcam"):
            process_webcam(action_net, classes, SAMPLE_DURATION)

def process_video(video_path, action_net, classes, SAMPLE_DURATION):
    frames = deque(maxlen=SAMPLE_DURATION)  # Initialize frames here
    vs = cv2.VideoCapture(video_path)

    # Create a placeholder for the frame display
    frame_placeholder = st.empty()

    while True:
        grabbed, frame = vs.read()
        if not grabbed:
            st.write("[INFO] no frame read from stream - exiting")
            break

        frame = imutils.resize(frame, width=400)
        frames.append(frame)

        if len(frames) < SAMPLE_DURATION:
            continue

        label = recognize_action(frames, action_net, classes)

        # Draw rectangle and label on the frame
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Update the placeholder with the latest processed frame
        frame_placeholder.image(frame, channels="BGR", use_column_width=True)

        # Introduce a short sleep to allow Streamlit to update the UI
        time.sleep(0.1)  # Adjust this value for better responsiveness

    vs.release()

def process_webcam(action_net, classes, SAMPLE_DURATION):
    frames = deque(maxlen=SAMPLE_DURATION)  # Initialize frames here
    vs = cv2.VideoCapture(0)

    # Create a placeholder for the frame display
    frame_placeholder = st.empty()

    while True:
        grabbed, frame = vs.read()
        if not grabbed:
            st.write("[INFO] no frame read from webcam - exiting")
            break

        frame = imutils.resize(frame, width=400)
        frames.append(frame)

        if len(frames) < SAMPLE_DURATION:
            continue

        label = recognize_action(frames, action_net, classes)

        # Draw rectangle and label on the frame
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Update the placeholder with the latest processed frame
        frame_placeholder.image(frame, channels="BGR", use_column_width=True)

        # Introduce a short sleep to allow Streamlit to update the UI
        # time.sleep(0.1)  # Adjust this value for better responsiveness

    vs.release()

if __name__ == "__main__":
    main()
