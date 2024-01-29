import cv2
import torch
import numpy as np
import dlib

from gradcam import overlay
from inference import load_model_and_preprocessing
from utils import LABEL_TO_STR

BLUE_COLOR = (255, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
WHITE_COLOR = (255, 255, 255)
INFO_TEXT_SIZE = 0.7
DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
MOUTH_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)
EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


def initialize_model(model_name: str):
    """Initialize the model with the given name"""
    _, model, preprocessing = load_model_and_preprocessing(model_name)
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, preprocessing, device


def initialize_cap(webcam: bool, cam_id: int, video_input: str):
    """Initialize the camera or video input"""
    if webcam:
        cap = cv2.VideoCapture(cam_id)
    else:
        cap = cv2.VideoCapture(video_input)

    if not cap.isOpened():
        raise IOError("Cannot open camera or video file")

    return cap


def initialize_out(
    cap: cv2.VideoCapture, 
    file: str, 
    codec: str = "MJPG"
    ):
    """Initialize the video output"""
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter(file, fourcc, fps, (width, height))


def top_prediction_with_label(predictions: torch.Tensor):
    """Return the top prediction with its label"""
    top_prediction = torch.argmax(predictions, 1)
    label = LABEL_TO_STR[top_prediction.item()]

    score = predictions[0][top_prediction].item()
    return label, score


def predict_emotions(
    image: np.ndarray, 
    model: torch.nn.Module, 
    preprocessing: torch.nn.Module, 
    device: str,
    ):
    """Predict the emotions of the given image"""
    resized_image = cv2.resize(image, (64, 64))
    normalized_image = resized_image / 255.0
    torch_image = (
        torch.from_numpy(normalized_image)
        .unsqueeze(0)
        .permute(0, 3, 1, 2)
        .float()
        .to(device)
    )

    if preprocessing:
        torch_image = preprocessing(torch_image)

    with torch.no_grad():
        output = model(torch_image)
        predictions = torch.nn.functional.softmax(output, dim=1)

    return predictions

def draw_face_rectangle(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    id: int,
    emotion: str,
):
    cv2.rectangle(frame, (x, y), (x + w, y + h), BLUE_COLOR, 2)

    dynamic_text_size = max(min(h / 300, 1), 0.5)
    cv2.putText(
        frame,
        f"Face #{id + 1}: {emotion}",
        (x, y - 10),
        DEFAULT_FONT,
        dynamic_text_size,
        BLUE_COLOR,
        2,
    )

def draw_mouth_and_eyes(
    frame: np.ndarray,
    roi: np.ndarray,
    x: int,
    y: int,
):
    mouths = MOUTH_CASCADE.detectMultiScale(roi, 1.6, 20)

    if len(mouths) > 0:
        (mx, my, mw, mh) = mouths[0]
        cv2.ellipse(
            roi,
            (mx + mw // 2, my + mh // 2),
            (mw // 2, mh // 2),
            0,
            0,
            360,
            RED_COLOR,
            2,
        )

    eyes = EYE_CASCADE.detectMultiScale(roi, 1.8, 10)
    for x2, y2, w2, h2 in eyes[:2]:
        eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
        radius = int(round((w2 + h2) * 0.25))
        cv2.circle(frame, eye_center, radius, RED_COLOR, 4)

def draw_explanation_overlay(
    frame: np.ndarray,
    id: int,
    picture_with_overlay: np.ndarray,
):
    OVERLAY_SIZE = 128
    picture_with_overlay = cv2.resize(picture_with_overlay, (OVERLAY_SIZE, OVERLAY_SIZE))
    cv2.putText(
        frame,
        f"Face #{id + 1}",
        (frame.shape[1] - 129 - id * 128 + 10, frame.shape[0] - 138),
        DEFAULT_FONT,
        0.5,
        BLUE_COLOR,
        2,
    )
    frame[-129 : -1, -129 - id * 129 : -1 - id * 129] = picture_with_overlay


def draw_info_box(
    frame: np.ndarray,
    id: int,
    predictions: torch.Tensor,
):
    """Draw the info box with the predictions"""
    offset = id * 150 + 60
    cv2.rectangle(frame, (10, offset), (220, offset + 150), WHITE_COLOR, 2)
    cv2.putText(
        frame,
        f"Face #{id + 1}",
        (15, offset + 20),
        DEFAULT_FONT,
        INFO_TEXT_SIZE,
        WHITE_COLOR,
        2,
    )

    for i, (emotion, score) in enumerate(
        zip(LABEL_TO_STR.values(), predictions[0])
    ):
        cv2.putText(
            frame,
            f"{emotion}: {score:.2f}",
            (15, offset + 40 + i * 20),
            DEFAULT_FONT,
            INFO_TEXT_SIZE,
            WHITE_COLOR,
            2,
        )


def process_frame(
    frame: np.ndarray, 
    model: torch.nn.Module, 
    device: str, 
    preprocessing: str, 
    emotion_score: dict,
    show_explanation: bool,
    show_details: bool,
    show_info_box: bool,
    use_hog: bool,
    ):
    """Process the frame and return the frame with the predicted emotions"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
    
    if use_hog:
        hog_detector = dlib.get_frontal_face_detector()
        dlib_faces = hog_detector(gray)
        faces = [(d.left(), d.top(), d.right() - d.left(), d.bottom() - d.top()) for d in dlib_faces]
    else: 
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

    if show_info_box:
        cv2.putText(
            frame,
            f"Number of faces detected: {len(faces)}",
            (10, 30),
            DEFAULT_FONT,
            1.1,
            WHITE_COLOR,
            2,
        )

    for face_id, (x, y, w, h) in enumerate(faces):
        roi = frame[y : y + h, x : x + w]

        if show_explanation:
            predictions, picture_with_overlay = overlay(roi, model)
            predictions = torch.nn.functional.softmax(predictions, dim=1)
        else :
            predictions = predict_emotions(roi, model, preprocessing, device)

        emotion, score = top_prediction_with_label(predictions)

        emotion_score[emotion]["total_score"] += score
        emotion_score[emotion]["count"] += 1

        draw_face_rectangle(frame, x, y, w, h, face_id, emotion)

        if show_details:
            draw_mouth_and_eyes(frame, roi, x, y)
                
        if show_explanation:
            draw_explanation_overlay(frame, face_id, picture_with_overlay)

        if show_info_box:
            draw_info_box(frame, face_id, predictions)
            
    return frame


def main_loop(
    cap: cv2.VideoCapture, 
    model: torch.nn.Module, 
    device: str, 
    preprocessing: str,
    webcam: bool,
    show_processing: bool, 
    show_explanation: bool,
    show_details: bool,
    show_info_box: bool,
    use_hog: bool,
    ):
    """Main loop for video prediction"""
    print("Starting video prediction...")

    emotion_scores = {
        emotion: {"total_score": 0, "count": 0} for emotion in LABEL_TO_STR.values()
    }

    while True:
        has_frame, frame = cap.read()

        if not has_frame:
            break

        if webcam:
            frame = cv2.flip(frame, 1)

        processed_frame = process_frame(
            frame, model, device, preprocessing, emotion_scores, show_explanation, show_details, show_info_box, use_hog
        )

        if show_processing:
            cv2.imshow("Facial Emotion Recognition", processed_frame)

        q_pressed = cv2.waitKey(1) == ord("q")
        window_closed = (
            show_processing
            and cv2.getWindowProperty(
                "Facial Emotion Recognition", cv2.WND_PROP_VISIBLE
            )
            < 1
        )
        if q_pressed or window_closed:
            print("Video prediction interrupted.")
            break

    for emotion, score in emotion_scores.items():
        average_score = (
            score["total_score"] / score["count"] if score["count"] > 0 else 0
        )
        print(f"Emotion: {emotion}, Average Score: {average_score:.2f}")


def make_video_prediction(
    model_name: str,
    record: bool,
    webcam: bool,
    cam_id: int,
    video_input: str,
    output_file: str,
    show_processing: bool,
    show_explanation: bool,
    show_details: bool,
    show_info_box: bool,
    use_hog: bool,
):
    """Make video prediction using the model with the given name"""

    model, preprocessing, device = initialize_model(model_name)
    cap = initialize_cap(webcam, cam_id, video_input)

    if record:
        out = initialize_out(cap, output_file)

    try:
        main_loop(
            cap, 
            model, 
            device, 
            preprocessing, 
            webcam, 
            show_processing, 
            show_explanation, 
            show_details, 
            show_info_box,
            use_hog,
        )
    except Exception as e:
        print(e)
    finally:
        cap.release()
        if record:
            out.release()
            print(f"Video successfully saved as {output_file}")
        cv2.destroyAllWindows()


