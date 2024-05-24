import cv2
import time
import csv
from datetime import datetime

cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('face_detect.xml')

prev_time = 0
interval = 5  # Set the interval to 5 seconds
max_faces = 0

while True:
    ret, img = cap.read()
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faceRect = faceCascade.detectMultiScale(gray, 1.1, 5)
        max_faces = max(max_faces, len(faceRect))

        for (x, y, w, h) in faceRect:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    current_time = time.time()
    if current_time - prev_time >= interval:
        prev_time = current_time
        with open('face_detection.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), max_faces])
        max_faces = 0

    if cv2.waitKey(1) == ord('q'):
        break