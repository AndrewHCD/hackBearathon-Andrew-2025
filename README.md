Shoulder Press Analyzer
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Table of Contents**

Overview

Features

Requirements

Installation

How to Use

Directory Structure

Troubleshooting

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Overview**

This Shoulder Press Anazlyzer is a Python based application which uses automation to analzye your form during real time, it uses mediapipe for hand and face detection and opencv for video processing and display. This provides real time feedback on form, a video at the end, and what to improve.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Features**

Real-Time Feedback:

Detects uneven hand heights and distances.

Provides feedback to help maintain proper form.

Displays warnings for uneven hands or heights.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Repetition Counting:

Tracks the number of repetitions completed based on hand movements.

Ensures both top and bottom positions are reached before counting a rep.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Workout Recording

Records workout sessions as videos.

Saves videos in the recorded_videos directory.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Dynamic Feedback Messages:

Provides specific feedback on form improvement based on uneven height or hand distance durations.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Timers and Statistics:

Tracks total workout time.

Displays total time spent with uneven form (height or hand distance).

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Requirements**

Software:

Python 3.7 or higher

In terminal Type 

python --version

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Libraries:

OpenCV (cv2) :  pip install opencv-python

MediaPipe (mediapipe) : pip install mediapipe

NumPy (numpy) : pip install numpy

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Hardware**

A webcam for real-time video capture.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Installation**

Clone or download the repository.

Install the required Python libraries:

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**bash** 

if you didnt read above read these below

in terminal do the follow:

pip install opencv-python mediapipe numpy

Ensure your webcam is connected and functional.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**How to Use**

Run the program:

**bash**

python shoulder_press_analyzer.py

Follow the on-screen instructions:

Step 1: Raise one finger on both hands to set the highest point.

Step 2: Raise two fingers on both hands to start the workout.

Perform your shoulder press exercise while maintaining proper form.

Step 3: Raise one finger on both hands to mark the workout as completed.

View real-time feedback on screen, including:

Hand height differences.

Hand distance differences.

Repetition counts.

After completing the workout, review your saved video in the recorded_videos directory.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Directory Structure**
text

project/

├── shoulder_press_analyzer.py  # Main script

├── recorded_videos/            # Directory where videos are saved

└── README.md                   # Documentation file

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Troubleshooting**
Common Issues

Webcam Not Detected:

Ensure your webcam is connected properly.

Check if another application is using the webcam.

Ensure your hands are clearly visible to the webcam.

Avoid excessive background clutter.

Video Not Saving:

Ensure you have write permissions for the recorded_videos directory.
