import cv2 as cv
import numpy as np
import os
import glob

modelConfigFile = glob.glob("*.cfg")[0]
print("CFG File:", modelConfigFile)
modelWeights = glob.glob("*.weights")[0]
print("Weights File:", modelWeights)

# Confidence threshold for checking which boxes we need to keep
confidenceThreshold = 0.25 
nmsThreshold = 0.40
inpWidth = 416  
inpHeight = 416
classes = ["object"]

# Setup Deep Learning Model
net = cv.dnn.readNetFromDarknet(modelConfigFile, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
print("----------- CFG & Weights File Loaded. -----------")

def getOutputNames(net):
    layerNames = net.getLayerNames()
    return [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

def drawPred(classId, conf, left, top, right, bottom, frame):
    cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    label = "%.2f" % conf
    if classes:
        assert(classId < len(classes))
        label = "%s:%s"%(classes[classId], label)
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

def postProcess(frame, outs):
    # Making a copy to avoid the bounding box to be displayed in the output images
    t_frame = frame.copy()
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    
    classIds = []
    confidences = []
    boxes = []

    frames = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confidenceThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)

                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)

                left = int(center_x - width / 2)
                top = int(center_y - height / 2)

                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        drawPred(classIds[i], confidences[i], left, top, left+width, top+height, frame)
        # For multiple detections in one image.
        frames.append(t_frame[top:top+height, left:left+width])
    return frames

def objDetector(image, i):
    name = image.split(".")[0]
    cap = cv.imread(image)

    blob = cv.dnn.blobFromImage(cap, 1/255, (inpWidth, inpHeight), (0,0,0), True, crop = False)

    net.setInput(blob)
    outs = net.forward(getOutputNames(net))
    
    x = postProcess(cap, outs)
    if(len(x)):
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        print(i+1, label)
        cv.imshow(name, cap)
        # For multiple detections in one image.
        count = 0
        for detection in x:
            cv.imwrite(f"Output/{count}{i}.jpg", detection)
            count += 1
            cv.waitKey(0)
