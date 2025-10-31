Real-Time Vehicle and Pedestrian Detection using YOLO and OpenCV

This project detects vehicles (cars, trucks, buses) and pedestrians in real-time using the YOLO (You Only Look Once) deep learning model and OpenCV for visualization. It demonstrates the use of a pre-trained deep learning model for object detection without relying on traditional Haar Cascade methods.

1. Project Overview

The program reads video frames, processes them through a pre-trained YOLO model, and displays bounding boxes around detected objects such as cars, buses, trucks, and people. Each object type can be shown in a different color for better visibility. The system runs in real time and can be stopped manually using a key press.

2. Key Features

Detects multiple object types: car, bus, truck, and person

Works with both video files and live webcam feeds

Uses a deep learning-based YOLO model (no Haar Cascade)

Displays real-time detection results with bounding boxes

Press ‘q’ or ‘Q’ to stop detection

3. Technologies Used

Python 3

OpenCV – for reading frames and visualizing detections

YOLOv8 (Ultralytics) – for deep learning-based object detection

NumPy – for array operations

4. How It Works

Each frame from the video or webcam is captured using OpenCV.

The YOLO model processes each frame and returns detected object coordinates and labels.

Rectangles are drawn around the detected cars, buses, trucks, and people.

The frame is displayed in real time until the user quits.
