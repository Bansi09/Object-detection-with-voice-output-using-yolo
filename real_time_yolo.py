import cv2
import numpy as np
import time
import os
import imutils
import subprocess
from gtts import gTTS
from pydub import AudioSegment
AudioSegment.converter="C:/Users/J & B/Downloads/ffmpeg-2021-11-03-git-08a501946f-essentials_build/bin/ffmpeg.exe"

# Load Yolo
net = cv2.dnn.readNet("weights/yolov3.weights", "cfg/yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
while True:
    _, frame = cap.read()
    frame_id += 1

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(  #converts img to blob
        frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

    net.setInput(blob)  # setting blob as an input to our network
    outs = net.forward(output_layers) 

    # Showing informations on the screen
    #filtering out detections to get only useful detections
    class_ids = []
    confidences = []
    boxes = []  #this bounding box contains value of x, y, width, ht 
    centers = []
    for out in outs:
        for detection in out:
            scores = detection[5:]  #remove first five elements from bbox output
            class_id = np.argmax(scores)  #finds the index of max value (max confidence of bboxes)
            confidence = scores[class_id]  # max value
            if confidence > 0.5:   # if confidence is above 50% then we will say it as good detection
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width) #gives pixel values
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2) #to get center we need to divide width by 2
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                centers.append((center_x, center_y))
    #print(len(bbox))
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)  #compress all bounding boxes that are less than the max confidence
    ''' here 0.8 is confidence threshold and 
    0.3 is nms threshold => Lower the nmsthreshold more aggresive ie lower numberof bbox'''



    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2) #bbox in output
            cv2.putText(frame, label + " " + str(round((confidence * 100), 2)),
                        (x, y + 30), font, 3, color, 3)
    elapsed_time = time.time() - starting_time
    # fps = frame_id / elapsed_time
    # cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 4, (0, 0, 0), 3)

    texts = []
    # ensure at least one detection exists
    if len(indexes) > 0:
        for i in indexes.flatten():  # loop over the indexes we are keeping
            # find positions
            center_x, center_y = centers[i][0], centers[i][1]
            if center_x <= 320/3:
                W_pos = "left "
            elif center_x <= (320/3 * 2):
                W_pos = "center "
            else:
                W_pos = "right "

            if center_y <= 320/3:
                H_pos = "top "
            elif center_y <= (320/3 * 2):
                H_pos = "mid "
            else:
                H_pos = "bottom "

            if((classes[classes.index(label)] == classes[0]) | (classes[classes.index(label)] >= classes[i])):
                break
            else:
                texts.append(H_pos + W_pos + classes[class_ids[i]])
            

        print(texts)

        if texts:
            description = ', '.join(texts)
            tts = gTTS(description, lang='en')
            tts.save('tts.mp3')
            tts = AudioSegment.from_mp3("tts.mp3")
            subprocess.call(["ffplay", "-nodisp", "-autoexit", "tts.mp3"])
        
   
    

    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

 

    

cap.release()
cv2.destroyAllWindows()
os.remove("tts.mp3")

