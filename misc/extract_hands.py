import cv2
import time
from pathlib import Path
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Drawing utils
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

# - Global variables -
# Drawing variables
MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)
# Other
latest_result = None
model_path = str(Path.cwd() / "mediapipe_models" / "hand_landmarker.task")

# FPS tracking variables
landmarker_fps = 0.0
last_landmarker_time = 0.0


# Callback function for handlandmarker
def update_result(
    result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int
):
    global latest_result, landmarker_fps, last_landmarker_time
    latest_result = result

    current_time = time.time()
    if last_landmarker_time > 0:
        landmarker_fps = 1.0 / (current_time - last_landmarker_time)
    last_landmarker_time = current_time


# Drawing function
def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(
            annotated_image,
            f"{'Right' if handedness[0].category_name == 'Left' else 'Right'}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated_image


base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=update_result,
)

print("Starting camera... Press 'q' to quit")

with vision.HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)

    prev_frame_time = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Flip the frame IMMEDIATELY after capturing it.
        # This acts like a mirror and solves the backwards text problem,
        # because the text will be drawn normally onto the already-mirrored image.
        frame = cv2.flip(frame, 1)

        # Calculate camera FPS
        new_frame_time = time.time()
        fps_camera = (
            1.0 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
        )
        prev_frame_time = new_frame_time

        # Convert opencv bgr to rgb
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)

        # Send frame to model
        landmarker.detect_async(mp_image, timestamp_ms)

        # Draw results
        if latest_result and latest_result.hand_landmarks:
            annotated_rgb = draw_landmarks_on_image(rgb_frame, latest_result)
            frame_to_show = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        else:
            frame_to_show = frame

        # Draw FPS counters onto the final image
        cv2.putText(
            frame_to_show,
            f"Camera FPS: {int(fps_camera)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame_to_show,
            f"Model FPS: {int(landmarker_fps)}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

        cv2.imshow("Tasks API - Skeletons", frame_to_show)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
