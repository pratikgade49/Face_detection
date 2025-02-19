import cv2

def display_authentication_status(frame, is_authenticated, username=None):
    """
    Displays the authentication status on the frame.

    Args:
        frame (np.ndarray): The frame on which to display the status.
        is_authenticated (bool): True if the user is authenticated, False otherwise.
        username (str or None): The username of the authenticated user.

    If authenticated, displays "Authenticated: [Username]". Otherwise, displays "Authentication Failed".
    """
    if is_authenticated and username:
        message = f"Authenticated: {username}"
        color = (0, 255, 0)  # Green
    else:
        message = "Authentication Failed"
        color = (0, 0, 255)  # Red
    cv2.putText(frame, message,
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2)

def display_frame(window_name, frame):
    """
    Displays the given frame in a window.

    Args:
        window_name (str): The name of the window.
        frame (np.ndarray): The frame to display.
    """
    cv2.imshow(window_name, frame)

def display_warning(frame):
    """
    Overlays a warning message on the frame when the user is not focusing.

    Args:
        frame (np.ndarray): The frame on which to overlay the warning message.
    """
    cv2.putText(frame, "WARNING: Focus on the screen!",
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

def display_focus_times(frame, focus_time, not_focus_time):
    """
    Displays the focusing and not focusing times on the frame.

    Args:
        frame (np.ndarray): The frame on which to display the times.
        focus_time (int): The total focusing time in seconds.
        not_focus_time (int): The total not focusing time in seconds.
    """
    cv2.putText(frame, f"Focusing Time: {focus_time}s",
                (50, 110), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    cv2.putText(frame, f"Not Focusing Time: {not_focus_time}s",
                (50, 140), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2)

def close_window(window_name):
    """
    Closes the display window.

    Args:
        window_name (str): The name of the window to close.
    """
    cv2.destroyWindow(window_name)

def display_multiple_faces_error(frame):
    """
    Displays an error message when multiple faces are detected.

    Args:
        frame (np.ndarray): The frame on which to display the error message.
    """
    cv2.putText(frame, "ERROR: Multiple faces detected!",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 3)
