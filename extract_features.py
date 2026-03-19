import os

os.environ["GLOG_minloglevel"] = "4"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
import json
import cv2
import mediapipe as mp
import numpy as np
import sys
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed


@contextmanager
def suppress_stderr():
    fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(fd, 2)
    os.close(fd)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)


DATASET_ROOT = os.path.join(os.getcwd(), "dataset")
VIDEO_DIR = os.path.join(DATASET_ROOT, "videos")
JSON_PATH = os.path.join(DATASET_ROOT, "nslt_100.json")
CLASS_LIST_PATH = os.path.join(DATASET_ROOT, "wlasl_class_list.txt")
OUTPUT_DIR = "wlasl100_features"
MODEL_PATH = os.path.join(os.getcwd(), "mediapipe_models", "hand_landmarker.task")


# shape: (126,)
def extract_keypoints(hand_landmarks_list, handedness_list):
    left_hand, right_hand = (
        np.zeros(63),
        np.zeros(63),
    )  # 21 landmarks, each landmark has x,y,z

    if hand_landmarks_list:
        for idx, hand_landmarks in enumerate(hand_landmarks_list):
            handedness = handedness_list[idx][0].category_name
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks]).flatten()
            if handedness == "Left":
                left_hand = coords
            elif handedness == "Right":
                right_hand = coords
    return np.concatenate([left_hand, right_hand])


def process_video_wrapper(args):
    video_id, info, class_map, output_dir, model_path, video_dir = args
    subset = info["subset"]
    class_id = info["action"][0]
    word = class_map.get(class_id, "unknown")

    video_path = os.path.join(video_dir, f"{video_id}.mp4")

    # Build save path
    save_dir = os.path.join(output_dir, subset, word)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{video_id}.npy")

    # Skip if not exist or already processed
    if not os.path.exists(video_path):
        return (video_id, False, True, subset, word, 0)
    if os.path.exists(save_path):
        return (video_id, True, False, subset, word, -1)

    # Process video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0  # fallback for corrupted metadata

    video_data = []
    frame_idx = 0

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_hands=2
    )

    with suppress_stderr():
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(1000 * frame_idx / fps)

                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                keypoints = extract_keypoints(result.hand_landmarks, result.handedness)

                video_data.append(keypoints)
                frame_idx += 1

    cap.release()

    if len(video_data) > 0:
        np.save(save_path, np.array(video_data))
        return (video_id, True, False, subset, word, len(video_data))
    else:
        return (video_id, True, False, subset, word, 0)


if __name__ == "__main__":
    # Load class map
    class_map = {}
    with open(CLASS_LIST_PATH, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                class_map[int(parts[0])] = parts[1]

    print("Starting Kaggle Dataset Extraction...")

    with open(JSON_PATH, "r") as f:
        metadata = json.load(f)

    tasks_args = [
        (vid, info, class_map, OUTPUT_DIR, MODEL_PATH, VIDEO_DIR)
        for vid, info in metadata.items()
    ]

    processed = 0
    missing = 0
    already_done = 0

    # Execute using all available CPU cores
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_video_wrapper, args) for args in tasks_args]

        for future in as_completed(futures):
            vid, was_processed, was_missing, subset, word, frames = future.result()

            if was_missing:
                missing += 1
            else:
                if frames == -1:
                    already_done += 1
                elif frames > 0:
                    print(f"Saved: {subset}/{word}/{vid}.npy | Frames: {frames}")
                    processed += 1
                else:
                    print(f"[!] Warning: MediaPipe extracted nothing from {vid}.mp4")
                    processed += 1

    print(
        f"\nProcessing complete! Processed {processed} new videos, skipped {already_done} already completed, skipped {missing} missing videos"
    )
