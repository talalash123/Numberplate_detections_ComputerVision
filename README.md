# Automatic Number Plate Recognition (ANPR) System

This project implements a computer vision solution for detecting vehicle license plates and extracting registration numbers from video footage. Using fine-tuned YOLOv8 for detection and EasyOCR for text extraction, the system logs license plate data with timestamps in real time.

## Project Overview

The system processes video feeds to perform live license plate detection and optical character recognition:

- License Plate Detection: Real-time object detection using a fine-tuned YOLOv8 Nano model achieving 84.8% mAP50.
- Optical Character Recognition: Automatic alphanumeric text parsing from detected plate regions using EasyOCR.
- Dynamic Video Annotations: Visual bounding boxes and extracted text labels overlaid directly onto output frames.
- Automated Data Logging: Continuous recording of frame numbers, timestamps, and extracted plate numbers saved to a structured CSV file.

## Project Structure

- main.py: Main execution script for processing video input, applying OCR, and generating output files.
- best.pt: Fine-tuned YOLOv8 custom weights trained specifically for license plate detection.
- sample.mp4: Sample input video stream used for testing the detection pipeline.
- requirements.txt: Python package dependencies required to set up and run the environment.

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone [https://github.com/talalash123/Numberplate_detections_ComputerVision.git](https://github.com/talalash123/Numberplate_detections_ComputerVision.git)
   cd Numberplate_detections_ComputerVision
