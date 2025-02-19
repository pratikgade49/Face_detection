# Face_detection

Eye Tracking and Face Recognition Application

# Overview

This application uses OpenCV for real-time eye tracking and face recognition. It leverages the LBPH (Local Binary Patterns Histograms) face recognizer to authenticate users based on pre-trained models. Additionally, it tracks users' focus times and not-focus times using face and eye detection techniques.

# Features

Face Recognition: Utilizes OpenCV's LBPH face recognizer to authenticate users.

Eye Tracking: Detects user eyes to track focus and not-focus times.

Face Alignment: Aligns faces for better recognition accuracy.

Multiple Face and Eye Detection Models: Uses Haar Cascades for face and eye detection.

Focus Time Tracking: Monitors and records the user's focus and not-focus durations

# Project Structure

your_project/
├── controllers/
│   └── controller.py
├── models/
│   └── model.py
├── views/
│   └── view.py
├── main.py
├── dataset/
│   └── user1/
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
├── trained_model.yml
└── labels.pickle

# Dataset Preparation
Create a dataset/ directory in the root of your project.

Inside dataset/, create subdirectories for each user, named after the user's name.

Place multiple face images of each user in their respective subdirectories. Ensure the images are clear and show the face in various expressions and lighting conditions.

# Example structure:

dataset/
├── user1/
│   ├── img1.jpg
│   ├── img2.jpg
├── user2/
│   ├── img1.jpg
│   ├── img2.jpg

# Running the Application
Train the Face Recognizer Model: Run the main application to train the model with the images in the dataset/ directory. The model will be saved as trained_model.yml and label mappings will be saved as labels.pickle.

Start Eye Tracking and Face Recognition: The application captures video from the webcam, performs face recognition, and tracks focus times.

# Code Explanation
# 1. models/model.py
FaceRecognizerModel Class:

Initializes the LBPH face recognizer.

Prepares training data by reading images, aligning faces, and extracting face regions.

Aligns faces using Haar Cascades for eye detection.

Trains and saves the face recognizer model and label mappings.

Loads the trained model and label mappings.

Focus Tracking Functions:

update_focus_time(is_focusing): Updates the focus and not-focus times based on whether the user is focusing.

finalize_times(): Finalizes the tracking times when the program ends.

get_focus_times(): Retrieves the total focusing time and total not focusing time.

pause_focus_tracking(): Pauses focus tracking by resetting the start times.

# 2. controllers/controller.py
initialize_resources():

Initializes the face and eye cascade classifiers.

Sets up video capture from the webcam.

authenticate_user():

Aligns the face using the model's alignment method.

Performs face recognition and returns the authentication status and username.

run():

Main loop of the application.

Handles user authentication, processes video frames for eye detection, and updates focus times.

Re-authenticates the user every 200 frames.

detect_faces_and_eyes():

Detects faces and eyes in the given frame.

Draws rectangles around detected faces and eyes.

# 3. views/view.py
display_authentication_status():

Displays the authentication status on the frame.

display_frame():

Displays the given frame in a window.

display_warning():

Overlays a warning message on the frame when the user is not focusing.

display_focus_times():

Displays the focusing and not focusing times on the frame.

close_window():

Closes the display window.

display_multiple_faces_error():

Displays an error message when multiple faces are detected.

# 4. main.py
main():

Entry point of the application.

Calls the run() function to start the eye tracking application.
