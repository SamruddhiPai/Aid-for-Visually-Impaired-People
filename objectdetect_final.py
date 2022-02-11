import cv2
import numpy as np
import pyttsx3
import time

engine = pyttsx3.init()

def audioOut(d, pos, *argv):                                #Audio output, d = 1 : distance available
    engine.say("Detected object is a,")                     #pos = 0 - center, 1 - right, 2 - left
    if d == 0:
        engine.say(argv)
    elif d == 1:
        engine.say(str(argv[0]) + 'at a relative distance of' +str(argv[1]))
    if pos == 0:
        engine.say('in front of you')
    elif pos == 1:
        engine.say('to your right')
    else:
        engine.say('to your left')
    engine.runAndWait()


def position(detected_classes,xc_coordinates):
    Xc = xc_coordinates[i]
    if Xc <= w:
        audioOut(0,2,detected_classes[i])
        
    elif Xc >= w*4:
        audioOut(0,1,detected_classes[i])
    else:
        audioOut(0,0,detected_classes[i])
       


def positionWithDistance(distance):
    if bool(not(distance)) == False : 
        d_min = min(distance.values())
    for obj,parameters in distance.items():
        dist = int(parameters[0])
        Xc = parameters[1]
        if dist <= int(d_min[0])+20:
            if Xc <= w:
                audioOut(1,2,obj,dist)    
            elif Xc >= w*4:
                audioOut(1,1,obj,dist)
            else:
                audioOut(1,0,obj,dist)

                    

      
#Load Yolo

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
height_dict = dict([('bus',4115),('car', 1700),('bicycle',550),('book',285),('bottle',190),('cell phone',150),('tvmonitor',300),('truck',2800),('motorbike',790),('remote',180)])
F = 3.85
distance = {}
d_min = 0


# Loading image
#img = cv2.imread("trafficImg.jpg")

video = cv2.VideoCapture(0)
while(True):
    start = time.time()
    check,img = video.read()
    img = cv2.resize(img, None, fx=0.8, fy=0.8)
    #print(img.shape)
    height1, width1, channels1 = img.shape
        #cropping image
    
    height, width, channels = img.shape
    w = width/5
        # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
        # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    detected_classes = []
    pixel_height = []
    xc_coordinates = []
    yc_coordinates = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.7:
                    # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                xc_coordinates.append(center_x)
                
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                    # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            pixel_height.append(h)
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
            detected_classes.append(label)
    cv2.imshow("Image", img)                          #display image

    #distance calculation 
    
    for i in range(0,len(detected_classes)):
        if detected_classes[i] in height_dict.keys():
            d = (F*height_dict[detected_classes[i]])/(pixel_height[i])
            distance[detected_classes[i]] = [d,xc_coordinates[i]]
        else:
            position(detected_classes,xc_coordinates)  
                
    positionWithDistance(distance)

    end = time.time()
    print(end-start)
    
    key = cv2.waitKey(1)
    


    
        


