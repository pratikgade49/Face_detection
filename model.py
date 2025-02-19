# import cv2
# import os
# import numpy as np
# import pickle
# from datetime import datetime

# class FaceRecognizerModel:
#     """
#     A class to handle training, saving, and loading of the LBPH face recognizer model.

#     Attributes:
#         face_recognizer (cv2.face.LBPHFaceRecognizer): The LBPH face recognizer object.
#         data_folder (str): The path to the dataset folder containing training images.
#         model_path (str): The path where the trained model will be saved or loaded from.
#         labels_path (str): The path where the label mapping will be saved or loaded from.
#         label_to_name (dict): A mapping from numeric labels to usernames.
#     """

#     def __init__(self, data_folder='dataset/', model_path='trained_model.yml', labels_path='labels.pickle'):
#         """
#         Initializes the FaceRecognizerModel with the specified data folder and model path.

#         Args:
#             data_folder (str, optional): The path to the dataset folder. Defaults to 'dataset/'.
#             model_path (str, optional): The path to save/load the trained model. Defaults to 'trained_model.yml'.
#             labels_path (str, optional): The path to save/load the labels mapping. Defaults to 'labels.pickle'.

#         Initializes the face recognizer and sets up the data folder and model paths.
#         """
#         self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#         self.data_folder = data_folder
#         self.model_path = model_path
#         self.labels_path = labels_path
#         self.label_to_name = {}

#     def prepare_training_data(self):
#         """
#         Prepares the training data by reading images, detecting faces, and extracting face regions.

#         Returns:
#             tuple: A tuple containing:
#                 - faces (list of np.ndarray): A list of face images in grayscale.
#                 - labels (list of int): A list of numeric labels corresponding to each face image.

#         Reads images from the data folder, detects faces using Haar Cascade, and extracts face regions.
#         Each user's images are assigned a unique label.
#         """
#         faces = []
#         labels = []
#         label = 0

#         # Loop over each user in the dataset
#         for user_folder in os.listdir(self.data_folder):
#             user_path = os.path.join(self.data_folder, user_folder)
#             if not os.path.isdir(user_path):
#                 continue

#             label += 1  # Assign a unique label to each user
#             username = user_folder
#             self.label_to_name[label] = username  # Map label to username

#             # Loop over each image in the user's folder
#             for image_name in os.listdir(user_path):
#                 if image_name.startswith('.'):
#                     continue
#                 image_path = os.path.join(user_path, image_name)
#                 image = cv2.imread(image_path)
#                 if image is None:
#                     continue
#                 gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#                 face_cascade = cv2.CascadeClassifier(
#                     cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#                 faces_rects = face_cascade.detectMultiScale(
#                     gray_image, scaleFactor=1.2, minNeighbors=5)
#                 # For each face detected, extract the face region
#                 for (x, y, w, h) in faces_rects:
#                     face = gray_image[y:y + h, x:x + w]
#                     face = cv2.resize(face, (200, 200))
#                     faces.append(face)
#                     labels.append(label)
#         return faces, labels

#     def train_model(self):
#         """
#         Trains the face recognizer model using the prepared training data.

#         Returns:
#             bool: True if training is successful, False otherwise.

#         The trained model is saved to the specified model path.
#         """
#         faces, labels = self.prepare_training_data()
#         if len(faces) == 0:
#             print("No faces found in the training data.")
#             return False
#         self.face_recognizer.train(faces, np.array(labels))
#         self.face_recognizer.save(self.model_path)
#         # Save the label mapping
#         with open(self.labels_path, 'wb') as file:
#             pickle.dump(self.label_to_name, file)
#         print("Training completed and model saved.")
#         return True

#     def load_model(self):
#         """
#         Loads the trained face recognizer model and label mapping from the model path.

#         Returns:
#             bool: True if the model and labels are loaded successfully or trained successfully if not found. False if training fails.

#         If the model file does not exist, it attempts to train a new model.
#         """
#         if os.path.exists(self.model_path) and os.path.exists(self.labels_path):
#             self.face_recognizer.read(self.model_path)
#             with open(self.labels_path, 'rb') as file:
#                 self.label_to_name = pickle.load(file)
#             print("Model and labels loaded from disk.")
#             return True
#         else:
#             print("Model file or labels not found. Training a new model...")
#             return self.train_model()

#     def get_recognizer(self):
#         """
#         Retrieves the face recognizer object.

#         Returns:
#             cv2.face.LBPHFaceRecognizer: The LBPH face recognizer object.
#         """
#         return self.face_recognizer

#     def get_label_mapping(self):
#         """
#         Retrieves the label to username mapping.

#         Returns:
#             dict: A dictionary mapping labels to usernames.
#         """
#         return self.label_to_name

# models/model.py

import cv2
import os
import numpy as np
import pickle
from datetime import datetime

class FaceRecognizerModel:
    """
    A class to handle training, saving, and loading of the LBPH face recognizer model.
    Face alignment is implemented using OpenCV's Haar Cascades for eye detection.
    """

    def __init__(self, data_folder='dataset/', model_path='trained_model.yml', labels_path='labels.pickle'):
        """
        Initializes the FaceRecognizerModel with specified paths.

        Args:
            data_folder (str): Path to the dataset folder.
            model_path (str): Path to save/load the trained model.
            labels_path (str): Path to save/load the labels mapping.
        """
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,
            neighbors=8,
            grid_x=8,
            grid_y=8
        )
        self.data_folder = data_folder
        self.model_path = model_path
        self.labels_path = labels_path
        self.label_to_name = {}
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml')

    def prepare_training_data(self):
        """
        Prepares the training data by reading images, aligning faces, and extracting face regions.

        Returns:
            tuple: A tuple containing:
                - faces (list of np.ndarray): A list of aligned face images in grayscale.
                - labels (list of int): A list of numeric labels corresponding to each face.
        """
        faces = []
        labels = []
        label = 0

        for user_folder in os.listdir(self.data_folder):
            user_path = os.path.join(self.data_folder, user_folder)
            if not os.path.isdir(user_path):
                continue

            label += 1  # Assign a unique label to each user
            username = user_folder
            self.label_to_name[label] = username  # Map label to username

            for image_name in os.listdir(user_path):
                if image_name.startswith('.'):
                    continue
                image_path = os.path.join(user_path, image_name)
                image = cv2.imread(image_path)
                if image is None:
                    continue

                aligned_face = self.align_face(image)
                if aligned_face is not None:
                    face_gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
                    face_gray = cv2.equalizeHist(face_gray)
                    face_gray = cv2.resize(face_gray, (200, 200))
                    faces.append(face_gray)
                    labels.append(label)
        return faces, labels

    def align_face(self, image):
        """
        Aligns a face in the given image using eye detection.

        Args:
            image (np.ndarray): The input image.

        Returns:
            aligned_face (np.ndarray or None): The aligned face image or None if alignment fails.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        if len(faces) == 0:
            return None  # No face detected

        x, y, w, h = faces[0]
        face_roi_gray = gray[y:y + h, x:x + w]
        face_roi_color = image[y:y + h, x:x + w]

        eyes = self.eye_cascade.detectMultiScale(
            face_roi_gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30)
        )

        if len(eyes) >= 2:
            # Extract the two eyes
            eye_1 = eyes[0]
            eye_2 = eyes[1]

            # Determine the left and right eyes
            if eye_1[0] < eye_2[0]:
                left_eye = eye_1
                right_eye = eye_2
            else:
                left_eye = eye_2
                right_eye = eye_1

            # Get the centers of the eyes
            left_eye_center = (int(left_eye[0] + left_eye[2] / 2),
                               int(left_eye[1] + left_eye[3] / 2))
            right_eye_center = (int(right_eye[0] + right_eye[2] / 2),
                                int(right_eye[1] + right_eye[3] / 2))

            # Calculate the angle between the eyes
            delta_x = right_eye_center[0] - left_eye_center[0]
            delta_y = right_eye_center[1] - left_eye_center[1]
            angle = np.arctan2(delta_y, delta_x) * 180 / np.pi

            # Compute the center of the face
            face_center = (int(x + w / 2), int(y + h / 2))

            # Get the rotation matrix
            M = cv2.getRotationMatrix2D(face_center, angle, 1)

            # Rotate the image to align the eyes horizontally
            aligned_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                                           flags=cv2.INTER_CUBIC)

            # Crop the aligned face
            aligned_face = aligned_image[y:y + h, x:x + w]
            return aligned_face
        else:
            # If eyes are not detected, return the original face region
            aligned_face = image[y:y + h, x:x + w]
            return aligned_face

    def train_model(self):
        """
        Trains the face recognizer model using the prepared training data.

        Returns:
            bool: True if training is successful, False otherwise.
        """
        faces, labels = self.prepare_training_data()
        if len(faces) == 0:
            print("No faces found in the training data.")
            return False
        self.face_recognizer.train(faces, np.array(labels))
        self.face_recognizer.save(self.model_path)
        # Save the label mapping
        with open(self.labels_path, 'wb') as file:
            pickle.dump(self.label_to_name, file)
        print("Training completed and model saved.")
        return True

    def load_model(self):
        """
        Loads the trained face recognizer model and label mapping.

        Returns:
            bool: True if the model and labels are loaded successfully or trained successfully if not found.
        """
        if os.path.exists(self.model_path) and os.path.exists(self.labels_path):
            self.face_recognizer.read(self.model_path)
            with open(self.labels_path, 'rb') as file:
                self.label_to_name = pickle.load(file)
            print("Model and labels loaded from disk.")
            return True
        else:
            print("Model or labels not found. Training a new model...")
            return self.train_model()

    def get_recognizer(self):
        """
        Retrieves the face recognizer object.

        Returns:
            cv2.face.LBPHFaceRecognizer: The LBPH face recognizer object.
        """
        return self.face_recognizer

    def get_label_mapping(self):
        """
        Retrieves the label to username mapping.

        Returns:
            dict: A dictionary mapping labels to usernames.
        """
        return self.label_to_name


# =========================================================================
# Focus tracking data and functions
# =========================================================================

# Initialize focus tracking data
focus_data = {
    "start_focus_time": None,
    "start_not_focus_time": None,
    "total_focus_time": 0,
    "total_not_focus_time": 0
}

def update_focus_time(is_focusing):
    """
    Updates the focus and not-focus times based on whether the user is focusing.

    Args:
        is_focusing (bool): True if the user is focusing on the screen, False otherwise.

    This function updates the timing data stored in focus_data.
    """
    current_time = datetime.now()
    if is_focusing:
        if focus_data["start_not_focus_time"]:
            # User was previously not focusing; update total not focusing time
            focus_data["total_not_focus_time"] += (current_time - focus_data["start_not_focus_time"]).total_seconds()
            focus_data["start_not_focus_time"] = None
        if not focus_data["start_focus_time"]:
            # Start timing focus period
            focus_data["start_focus_time"] = current_time
    else:
        if focus_data["start_focus_time"]:
            # User was previously focusing; update total focusing time
            focus_data["total_focus_time"] += (current_time - focus_data["start_focus_time"]).total_seconds()
            focus_data["start_focus_time"] = None
        if not focus_data["start_not_focus_time"]:
            # Start timing not focusing period
            focus_data["start_not_focus_time"] = current_time

def finalize_times():
    """
    Finalizes the tracking times when the program ends.

    This function updates the total focusing and not focusing times based on any ongoing periods.
    """
    current_time = datetime.now()
    if focus_data["start_focus_time"]:
        focus_data["total_focus_time"] += (current_time - focus_data["start_focus_time"]).total_seconds()
        focus_data["start_focus_time"] = None
    if focus_data["start_not_focus_time"]:
        focus_data["total_not_focus_time"] += (current_time - focus_data["start_not_focus_time"]).total_seconds()
        focus_data["start_not_focus_time"] = None

def get_focus_times():
    """
    Retrieves the total focusing time and total not focusing time.

    Returns:
        tuple: A tuple containing total focusing time and total not focusing time in seconds.
    """
    return int(focus_data["total_focus_time"]), int(focus_data["total_not_focus_time"])

def pause_focus_tracking():
    """
    Pauses focus tracking by resetting the start times.

    This function is called when the focus tracking should be paused, such as when authentication fails.
    """
    focus_data["start_focus_time"] = None
    focus_data["start_not_focus_time"] = None


