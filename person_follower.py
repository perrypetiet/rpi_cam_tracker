#Perry Petiet | 483554

import numpy as np
import socket, cv2, pickle, struct, time
from picamera2 import Picamera2
from rpi_hardware_pwm import HardwarePWM

#
#   CONSTANTS
#
width          = 640
height         = 480
confidence_min = 0.2
minDutyCycle   = 0.1
maxDutyCycle   = 12
panServoPin    = 17
dutyCycleStep  = 0.2
pixelThreshold = 60

currentPanDutyCycle = 8.0

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

#
#   LOAD MODEL
#
print("Loading model...")
net = cv2.dnn.readNetFromCaffe("mobilenet_ssd/MobileNetSSD_deploy.prototxt", "mobilenet_ssd/MobileNetSSD_deploy.caffemodel")

#
#   CREATE SOCKET AND WAIT FOR CONNECTION
#
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
host_name = socket.gethostname()
host_ip   = socket.gethostbyname("raspberrypi.local")
print('HOST IP:', host_ip)
print('HOST NAME:', host_name)

port = 9999
socket_address = (host_ip, port)
server_socket.bind(socket_address)

server_socket.listen(5)
print('LISTENING AT: ', socket_address)

#
#   SETUP SERVO'S
#

panServo = HardwarePWM(pwm_channel=0, hz=50)
panServo.start(currentPanDutyCycle) 

#
#   SOCKET CONNECTION
#   TAKE FRAME, DETECT PEOPLE, SEND FRAME
#
while True:
    client_socket,addr = server_socket.accept()
    print("GOT CONNECTION FROM: ", addr)

    #Check if socket active
    if client_socket:
        vid = Picamera2()
        vid.configure(vid.create_preview_configuration(main={"format": 'RGB888', "size": (width, height)}))
        vid.start()
        #set properties
        #vid.set(3, width)
        #vid.set(4, height)
        while(True):
            #
            #   GET IMAGE
            #
            image = vid.capture_array()
            
            if(True):
                (h, w) = image.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
	                (300, 300), 127.5)

                #
                #   COMPUTING OBJECT DETECTION
                #
                net.setInput(blob)
                detections = net.forward()

                highestConfidence   = 0
                highestConfidenceX  = 0
                highestConfidencey  = 0

                # EACH FRAME
                for i in np.arange(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > confidence_min:
                        idx = int(detections[0, 0, i, 1])
                    
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        if confidence > highestConfidence and idx == 15:
                            highestConfidence = confidence
                            # Centre x of box
                            highestConfidenceX = ((endX - startX) / 2) + startX
                            highestConfidencey = ((endY - startY) / 2) + startY
                            print(CLASSES[idx], idx, confidence, "X value: ", highestConfidenceX)
                        

                        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                        cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
                        y = startY - 15 if startY -15 > 15 else startY + 15
                        cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_PLAIN, 2, COLORS[idx], 2)

                        #
                        #   ADJUST CAMERA
                        #
                        if highestConfidenceX > 0:
                            if highestConfidenceX > width / 2 and highestConfidenceX - (width / 2) > pixelThreshold:
                                if currentPanDutyCycle + dutyCycleStep < maxDutyCycle:
                                    currentPanDutyCycle = currentPanDutyCycle + dutyCycleStep
                                    panServo.change_duty_cycle(currentPanDutyCycle)
                            elif highestConfidenceX < width / 2 and (width / 2) - highestConfidenceX > pixelThreshold:
                                if currentPanDutyCycle - dutyCycleStep > minDutyCycle:
                                    currentPanDutyCycle = currentPanDutyCycle - dutyCycleStep
                                    panServo.change_duty_cycle(currentPanDutyCycle)

                # Reset Values
                highestConfidence  = 0
                highestConfidenceX = 0
                highestConfidencey = 7


                #
                #   SEND TO NETWORK
                #
                a = pickle.dumps(image)
                message = struct.pack("Q",len(a))+a
                client_socket.sendall(message)

