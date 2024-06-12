import torch
import torchvision.transforms as T
import cv2
import time
import csv
from datetime import datetime
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn

# Load the pre-trained model
model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
model.eval()

# Load the labels used by the pre-trained model
labels = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
          'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
          'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
          'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
          'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
          'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
          'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
          'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
          'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
          'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
          'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
          'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Initialize object counts
object_counts = {label: 0 for label in labels if label != 'N/A'}
max_object_counts = {label: 0 for label in labels if label != 'N/A'}

with open('object_detection.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time'] + list(object_counts.keys()))

cap = cv2.VideoCapture(0)

prev_time = time.time()
last_capture_time = prev_time
interval = 5  # Set the interval to 5 seconds for writing to CSV

while True:
    current_time = time.time()
    
    if current_time - last_capture_time >= 1:  # Capture a frame every second
        ret, img = cap.read()
        last_capture_time = current_time
        if ret:
            # Convert the image to PyTorch Tensor
            img_tensor = T.ToTensor()(img).unsqueeze(0)

            # Perform object detection
            with torch.no_grad():
                prediction = model(img_tensor)

            # Reset the object counts for the current frame
            object_counts = {label: 0 for label in labels if label != 'N/A'}

            # Count each type of object if confidence score > 0.5
            for i in range(len(prediction[0]['labels'])):
                if prediction[0]['scores'][i] > 0.5:
                    label_index = prediction[0]['labels'][i].item()
                    if label_index < len(labels) and labels[label_index] != 'N/A':
                        label = labels[label_index]
                        object_counts[label] += 1

                        # Draw bounding box and label on the image
                        box = prediction[0]['boxes'][i].detach().numpy().astype(int)
                        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                        cv2.putText(img, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

            # Update max object counts
            for label in object_counts:
                if object_counts[label] > max_object_counts[label]:
                    max_object_counts[label] = object_counts[label]

            # Show the image
            cv2.putText(img, "Press q to leave the program", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('Object Detection', img)

    if current_time - prev_time >= interval:
        prev_time = current_time
        # Write the max object counts to the CSV file
        with open('object_detection.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S')] + [max_object_counts[label] for label in object_counts])
        max_object_counts = {label: 0 for label in object_counts}  # Reset max counts after writing to CSV

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
