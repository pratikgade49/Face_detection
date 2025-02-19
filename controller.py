# import cv2
# import numpy as np
# from models.model import (
#     FaceRecognizerModel,
#     update_focus_time,
#     finalize_times,
#     get_focus_times,
#     pause_focus_tracking,
#     focus_data
# )
# from views.view import (
#     display_frame,
#     display_warning,
#     display_focus_times,
#     close_window,
#     display_multiple_faces_error,
#     display_authentication_status
# )

# def initialize_resources():
#     """
#     Initializes the necessary resources including cascade classifiers and video capture.

#     Returns:
#         tuple: A tuple containing the face cascade, eye cascade, and video capture object.

#     Raises:
#         IOError: If the webcam cannot be opened.
#     """
#     face_cascade = cv2.CascadeClassifier(
#         cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     eye_cascade = cv2.CascadeClassifier(
#         cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
#     video_capture = cv2.VideoCapture(0)
#     if not video_capture.isOpened():
#         raise IOError("Cannot open webcam")
#     return face_cascade, eye_cascade, video_capture

# def detect_faces_and_eyes(face_cascade, eye_cascade, gray_frame, color_frame):
#     """
#     Detects faces and eyes in the given frame.

#     Args:
#         face_cascade (cv2.CascadeClassifier): The face detection cascade classifier.
#         eye_cascade (cv2.CascadeClassifier): The eye detection cascade classifier.
#         gray_frame (np.ndarray): The grayscale frame for detection.
#         color_frame (np.ndarray): The color frame for drawing rectangles.

#     Returns:
#         tuple: A tuple containing:
#             - eyes_detected (bool): True if eyes are detected.
#             - multiple_faces_detected (bool): True if more than one face is detected.

#     This function draws rectangles around detected faces and eyes on the color frame.
#     """
#     faces = face_cascade.detectMultiScale(
#         gray_frame,
#         scaleFactor=1.2,
#         minNeighbors=5,
#         minSize=(100, 100)
#     )
#     num_faces = len(faces)
#     multiple_faces_detected = num_faces > 1
#     eyes_detected = False

#     if multiple_faces_detected:
#         # Draw rectangles around all detected faces (red)
#         for (x, y, w, h) in faces:
#             cv2.rectangle(color_frame, (x, y), (x + w, y + h),
#                           (0, 0, 255), 2)
#     elif num_faces == 1:
#         # Only one face detected, proceed with eye detection
#         (x, y, w, h) = faces[0]
#         cv2.rectangle(color_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         face_roi_gray = gray_frame[y:y + h, x:x + w]
#         face_roi_color = color_frame[y:y + h, x:x + w]
#         eyes = eye_cascade.detectMultiScale(
#             face_roi_gray,
#             scaleFactor=1.1,
#             minNeighbors=8,
#             minSize=(25, 25)
#         )
#         if len(eyes) > 0:
#             eyes_detected = True
#             for (ex, ey, ew, eh) in eyes:
#                 cv2.rectangle(face_roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
#     else:
#         # No faces detected
#         pass

#     return eyes_detected, multiple_faces_detected

# def authenticate_user(face_recognizer, gray_frame, face_cascade, threshold, label_to_name):
#     """
#     Authenticates the user using the face recognizer.

#     Args:
#         face_recognizer (cv2.face.LBPHFaceRecognizer): The trained face recognizer.
#         gray_frame (np.ndarray): The grayscale frame for detection.
#         face_cascade (cv2.CascadeClassifier): The face detection cascade classifier.
#         threshold (float): The confidence threshold for authentication.
#         label_to_name (dict): A mapping from labels to usernames.

#     Returns:
#         tuple: A tuple containing:
#             - is_authenticated (bool): True if authentication is successful.
#             - username (str or None): The username if authenticated, None otherwise.

#     This function attempts to recognize the face in the frame and retrieves the username.
#     """
#     faces = face_cascade.detectMultiScale(
#         gray_frame, scaleFactor=1.2, minNeighbors=5)
#     for (x, y, w, h) in faces:
#         face = gray_frame[y:y + h, x:x + w]
#         face = cv2.resize(face, (200, 200))
#         label, confidence = face_recognizer.predict(face)
#         print(f"Predicted Label: {label}, Confidence: {confidence}")
#         if confidence < threshold:
#             username = label_to_name.get(label, "Unknown")
#             return True, username
#     return False, None

# def run():
#     """
#     The main loop of the application.

#     This function initializes resources, handles user authentication, processes video frames for eye detection,
#     and updates focus times. It displays the authentication status, warnings, and focus times on the video frame.
#     """
#     window_name = "Eye Tracking"
#     face_cascade, eye_cascade, video_capture = initialize_resources()

#     # Instantiate the FaceRecognizerModel
#     face_model = FaceRecognizerModel()
#     if not face_model.load_model():
#         # If model training failed, exit the application
#         print("Model training failed. Exiting application.")
#         return

#     face_recognizer = face_model.get_recognizer()
#     label_to_name = face_model.get_label_mapping()  # Retrieve the label to username mapping
#     user_authenticated = False
#     username = None
#     AUTH_THRESHOLD = 50 # Authentication confidence threshold
#     frame_count = 0
#     REAUTH_INTERVAL = 200  # Frames before re-authentication

#     try:
#         while True:
#             ret, frame = video_capture.read()
#             if not ret:
#                 print("Failed to capture video feed. Exiting...")
#                 break

#             gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             gray_frame = cv2.equalizeHist(gray_frame)

#             # Display authentication status with username
#             display_authentication_status(frame, user_authenticated, username)

#             if not user_authenticated:
#                 # Attempt to authenticate user
#                 user_authenticated, username = authenticate_user(
#                     face_recognizer, gray_frame, face_cascade, AUTH_THRESHOLD, label_to_name)
#                 if not user_authenticated:
#                     pause_focus_tracking()
#             else:
#                 # Re-authenticate periodically
#                 frame_count += 1
#                 if frame_count % REAUTH_INTERVAL == 0:
#                     user_authenticated, username = authenticate_user(
#                         face_recognizer, gray_frame, face_cascade, AUTH_THRESHOLD, label_to_name)
#                     if not user_authenticated:
#                         pause_focus_tracking()

#                 # Detect faces and eyes
#                 eyes_detected, multiple_faces_detected = detect_faces_and_eyes(
#                     face_cascade, eye_cascade, gray_frame, frame)

#                 if multiple_faces_detected:
#                     # Multiple faces detected, pause tracking
#                     display_multiple_faces_error(frame)
#                     pause_focus_tracking()
#                 else:
#                     # Update focus times based on whether eyes are detected
#                     update_focus_time(eyes_detected)
#                     focus_time, not_focus_time = get_focus_times()

#                     if not eyes_detected:
#                         # User is not focusing
#                         display_warning(frame)

#                     # Display focus times
#                     display_focus_times(frame, focus_time, not_focus_time)

#             # Display the video frame
#             display_frame(window_name, frame)

#             # Check for user input to exit
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 print("Exiting...")
#                 break

#     finally:
#         # Finalize times and release resources
#         finalize_times()
#         focus_time, not_focus_time = get_focus_times()
#         print(f"Total Focusing Time: {focus_time} seconds")
#         print(f"Total Not Focusing Time: {not_focus_time} seconds")
#         video_capture.release()
#         close_window(window_name)


# controllers/controller.py

import cv2
import numpy as np
import os
from models.model import (
    FaceRecognizerModel,
    update_focus_time,
    finalize_times,
    get_focus_times,
    pause_focus_tracking,
    focus_data
)
from views.view import (
    display_frame,
    display_warning,
    display_focus_times,
    close_window,
    display_multiple_faces_error,
    display_authentication_status
)

def initialize_resources():
    """
    Initializes the necessary resources including cascade classifiers and video capture.

    Returns:
        tuple: A tuple containing the face cascade, eye cascade, and video capture object.

    Raises:
        IOError: If the webcam cannot be opened.
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml')
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise IOError("Cannot open webcam")
    return face_cascade, eye_cascade, video_capture

def authenticate_user(face_model, gray_frame, frame, threshold):
    """
    Authenticates the user using the face recognizer.

    Args:
        face_model (FaceRecognizerModel): The face recognizer model.
        gray_frame (np.ndarray): The grayscale frame for processing.
        frame (np.ndarray): The original color frame.
        threshold (float): The confidence threshold for authentication.

    Returns:
        tuple: A tuple containing:
            - is_authenticated (bool): True if authentication is successful.
            - username (str or None): The username if authenticated, None otherwise.
    """
    aligned_face = face_model.align_face(frame)
    if aligned_face is None:
        return False, None

    face_gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
    face_gray = cv2.equalizeHist(face_gray)
    face_gray = cv2.resize(face_gray, (200, 200))

    label, confidence = face_model.face_recognizer.predict(face_gray)
    username = face_model.label_to_name.get(label, "Unknown")
    print(f"Predicted Label: {label}, Name: {username}, Confidence: {confidence}")
    if confidence < threshold:
        return True, username
    else:
        return False, None

def run():
    """
    The main loop of the application.

    Handles user authentication, processes video frames for eye detection,
    and updates focus times.
    """
    window_name = "Eye Tracking"
    face_cascade, eye_cascade, video_capture = initialize_resources()

    # Instantiate the FaceRecognizerModel
    face_model = FaceRecognizerModel()
    if not face_model.load_model():
        print("Model training failed. Exiting application.")
        return

    user_authenticated = False
    username = None
    AUTH_THRESHOLD = 50  # Adjusted threshold
    frame_count = 0
    REAUTH_INTERVAL = 200  # Re-authenticate every 200 frames

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to capture video feed. Exiting...")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.equalizeHist(gray_frame)

            display_authentication_status(frame, user_authenticated, username)

            if not user_authenticated or frame_count % REAUTH_INTERVAL == 0:
                # Attempt to authenticate user
                user_authenticated, username = authenticate_user(
                    face_model, gray_frame, frame, AUTH_THRESHOLD)
                if not user_authenticated:
                    pause_focus_tracking()

            if user_authenticated:
                # Proceed with focus tracking
                eyes_detected, multiple_faces_detected = detect_faces_and_eyes(
                    face_cascade, eye_cascade, gray_frame, frame)

                if multiple_faces_detected:
                    display_multiple_faces_error(frame)
                    pause_focus_tracking()
                else:
                    update_focus_time(eyes_detected)
                    focus_time, not_focus_time = get_focus_times()

                    if not eyes_detected:
                        display_warning(frame)

                    display_focus_times(frame, focus_time, not_focus_time)
            else:
                # Authentication failed
                pause_focus_tracking()

            # Display the video frame
            display_frame(window_name, frame)
            frame_count += 1  # Increment frame count

            # Check for user input to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break

    finally:
        # Finalize times and release resources
        finalize_times()
        focus_time, not_focus_time = get_focus_times()
        print(f"Total Focusing Time: {focus_time} seconds")
        print(f"Total Not Focusing Time: {not_focus_time} seconds")
        video_capture.release()
        close_window(window_name)

def detect_faces_and_eyes(face_cascade, eye_cascade, gray_frame, color_frame):
    """
    Detects faces and eyes in the given frame.

    Args:
        face_cascade (cv2.CascadeClassifier): The face detection cascade classifier.
        eye_cascade (cv2.CascadeClassifier): The eye detection cascade classifier.
        gray_frame (np.ndarray): The grayscale frame for detection.
        color_frame (np.ndarray): The color frame for drawing rectangles.

    Returns:
        tuple: A tuple containing:
            - eyes_detected (bool): True if eyes are detected.
            - multiple_faces_detected (bool): True if more than one face is detected.
    """
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )
    num_faces = len(faces)
    multiple_faces_detected = num_faces > 1
    eyes_detected = False

    if multiple_faces_detected:
        # Draw rectangles around all detected faces (red)
        for (x, y, w, h) in faces:
            cv2.rectangle(color_frame, (x, y), (x + w, y + h),
                          (0, 0, 255), 2)
    elif num_faces == 1:
        # Only one face detected, proceed with eye detection
        (x, y, w, h) = faces[0]
        cv2.rectangle(color_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_roi_gray = gray_frame[y:y + h, x:x + w]
        face_roi_color = color_frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(
            face_roi_gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(25, 25)
        )
        if len(eyes) >= 2:
            eyes_detected = True
            for (ex, ey, ew, eh) in eyes[:2]:
                cv2.rectangle(face_roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    else:
        # No faces detected
        pass

    return eyes_detected, multiple_faces_detected
