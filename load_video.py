import cv2
import numpy as np

MAX_SEQ_LENGTH = 120
IMG_SIZE = 224
STEP = 4

def crop_center_square(frame):
    """Crop the center square area of an image"""
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=MAX_SEQ_LENGTH, resize=(IMG_SIZE, IMG_SIZE), step=STEP):
    """
    Load a video, crop the center square and convert it into a numpy array.
    Return the numpy array along with the mask.
    Capture 1 frame every `step` steps.
    If the number of captured frames is less than `max_frames`, pad the numpy array with zeros.
    It the number of captrued frames is more than `max_frames`, ignore the rest.

    Parameters
    ----------
    path : str
        Path of the video
    max_frames : int, optional
        Maximum number of captured frames
    resize : int, optional
        Resolution of the returned video
    step : int, optional
        Capture 1 frame every `step` steps

    Returns
    -------
    tuple
        a tuple of the converted video and the mask
    """

    cap = cv2.VideoCapture(path)
    frames = []
    count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            if count % step:
                continue
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()

    # the mask indicates the frame is a captured frame and not a padded frame
    mask = np.zeros((max_frames,), dtype=bool)
    mask[:len(frames)] = 1

    return np.concatenate((np.array(frames), np.zeros((max_frames-len(frames), *resize, 3)))), mask
