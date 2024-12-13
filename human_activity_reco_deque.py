# # USAGE
# # python human_activity_reco_deque.py --model resnet-34_kinetics.onnx --classes action_recognition_kinetics.txt --input example_activities.mp4
# # python human_activity_reco_deque.py --model resnet-34_kinetics.onnx --classes action_recognition_kinetics.txt

# from collections import deque
# import numpy as np
# import argparse
# import imutils
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

# frames = deque(maxlen=SAMPLE_DURATION)

# print("[INFO] loading human activity recognition model...")
# net = cv2.dnn.readNet(args["model"])

# print("[INFO] accessing video stream...")
# vs = cv2.VideoCapture(args["input"] if args["input"] else 0)

# while True:
# 	(grabbed, frame) = vs.read()

# 	if not grabbed:
# 		print("[INFO] no frame read from stream - exiting")
# 		break
	
# 	frame = imutils.resize(frame, width=400)
# 	frames.append(frame)
	
# 	if len(frames) < SAMPLE_DURATION:
# 		continue

# 	blob = cv2.dnn.blobFromImages(frames, 1.0,
# 		(SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
# 		swapRB=True, crop=True)
# 	blob = np.transpose(blob, (1, 0, 2, 3))
# 	blob = np.expand_dims(blob, axis=0)
	
# 	net.setInput(blob)
# 	outputs = net.forward()
# 	label = CLASSES[np.argmax(outputs)]

# 	cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
# 	cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
# 		0.8, (255, 255, 255), 2)

# 	cv2.imshow("Activity Recognition", frame)
# 	key = cv2.waitKey(1) & 0xFF

# 	if key == ord("q"):
# 		break

from collections import deque
import numpy as np
import argparse
import cv2
import torch
import time
from threading import Thread

# Load YOLOv5 model
def load_yolov5_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n')  # Use 'yolov5n' for faster inference
    return model

# Load action recognition model
def load_action_recognition_model(model_path):
    net = cv2.dnn.readNet(model_path)
    return net

# Perform action recognition on the ROIs
def recognize_action(frames, net, classes):
    blob = cv2.dnn.blobFromImages(frames, 1.0, (112, 112), (114.7748, 107.7354, 99.4750), swapRB=True, crop=True)
    blob = np.transpose(blob, (1, 0, 2, 3))
    blob = np.expand_dims(blob, axis=0)
    
    net.setInput(blob)
    outputs = net.forward()
    return classes[np.argmax(outputs)]

def video_stream(vs, output_queue):
    while True:
        grabbed, frame = vs.read()
        if not grabbed:
            print("[INFO] no frame read from stream - exiting")
            break
        output_queue.append(frame)

def main(model_path, classes_path, input_source):
    CLASSES = open(classes_path).read().strip().split("\n")
    SAMPLE_DURATION = 16
    frames = deque(maxlen=SAMPLE_DURATION)
    output_queue = []  # Queue for frames from the video stream

    # Load YOLOv5
    yolo_model = load_yolov5_model()

    # Load action recognition model
    action_net = load_action_recognition_model(model_path)

    # Start video stream
    vs = cv2.VideoCapture(input_source if input_source else 0)

    # Start video stream thread
    stream_thread = Thread(target=video_stream, args=(vs, output_queue), daemon=True)
    stream_thread.start()

    frame_count = 0  # Frame counter
    last_frame_time = time.time()  # Track time for frame rate limiting

    while True:
        # Check if there's a frame available
        if len(output_queue) > 0:
            frame = output_queue.pop(0)  # Get the latest frame
            frame_count += 1

            # Limit to processing every nth frame (e.g., every 2nd frame)
            if frame_count % 2 != 0:
                continue

            frame_resized = cv2.resize(frame, (640, 480))  # Resize for YOLOv5
            results = yolo_model(frame_resized)  # Run YOLOv5

            # Parse detections
            detections = results.xyxy[0].numpy()  # Get detections

            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection  # Get coordinates and class
                if int(cls) == 0:  # Class ID 0 is for 'person'
                    roi = frame[int(y1):int(y2), int(x1):int(x2)]  # Extract ROI for action recognition

                    # Add ROI to frames deque for activity recognition
                    frames.append(roi)

                    if len(frames) == SAMPLE_DURATION:
                        activity_label = recognize_action(frames, action_net, CLASSES)

                        # Draw bounding box and label
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, activity_label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow("Multi-Person Activity Recognition", frame)

            # Frame rate control (process at a fixed rate, e.g., 15 FPS)
            elapsed_time = time.time() - last_frame_time
            if elapsed_time < 1/15:  # If less than 15 FPS, wait
                time.sleep((1/15) - elapsed_time)

            last_frame_time = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    vs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="path to trained human activity recognition model")
    ap.add_argument("-c", "--classes", required=True, help="path to class labels file")
    ap.add_argument("-i", "--input", type=str, default="", help="optional path to video file or webcam")
    args = vars(ap.parse_args())
    main(args["model"], args["classes"], args["input"])

