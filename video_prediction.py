import cv2
import torch

from gradcam import overlay
from inference import load_model_and_preprocessing
from utils import LABEL_TO_STR

BLUE_COLOR = (255, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
WHITE_COLOR = (255, 255, 255)
INFO_TEXT_SIZE = 0.7
DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX


def initialize_model(model_name: str):
    """Initialize the model with the given name"""
    _, model, preprocessing = load_model_and_preprocessing(model_name)
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, preprocessing, device


def initialize_cap(webcam: bool, video_input: str):
    """Initialize the camera or video input"""
    if not webcam and not video_input:
        raise IOError("Please specify a video file to use as input")

    if webcam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_input)

    if not cap.isOpened():
        raise IOError("Cannot open camera or video file")

    return cap


def initialize_out(cap, file, codec="MJPG"):
    """Initialize the video output"""
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter(file, fourcc, fps, (width, height))


def top_prediction_with_label(predictions):
    """Return the top prediction with its label"""
    top_prediction = torch.argmax(predictions, 1)
    label = LABEL_TO_STR[top_prediction.item()]

    score = predictions[0][top_prediction].item()
    return label, score


def predict_emotions(image, model, preprocessing, device):
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


def process_frame(frame, face_cascade, model, device, preprocessing, emotion_score):
    """Process the frame and return the frame with the predicted emotions"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cv2.putText(
        frame,
        f"Number of faces detected: {len(faces)}",
        (10, 30),
        DEFAULT_FONT,
        1.1,
        WHITE_COLOR,
        2,
    )

    offset = 60
    for face_nr, (x, y, w, h) in enumerate(faces):
        id = face_nr + 1

        roi = frame[y : y + h, x : x + w]

        # old way of doing stuff
        # predictions = predict_emotions(roi, model, preprocessing, device)
        predictions, picture_with_overlay = overlay(roi, model)

        emotion, score = top_prediction_with_label(predictions)

        emotion_score[emotion]["total_score"] += score
        emotion_score[emotion]["count"] += 1

        cv2.rectangle(frame, (x, y), (x + w, y + h), BLUE_COLOR, 2)

        dynamic_text_size = max(min(h / 300, 1), 0.5)
        cv2.putText(
            frame,
            f"Face #{id}: {emotion}",
            (x, y - 10),
            DEFAULT_FONT,
            dynamic_text_size,
            BLUE_COLOR,
            2,
        )

        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )

        show_info_box = True
        show_details = True
        if show_details:
            mouths = smile_cascade.detectMultiScale(roi, 1.6, 20)

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

            eyes = eye_cascade.detectMultiScale(roi, 1.8, 10)
            for x2, y2, w2, h2 in eyes[:2]:
                eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
                radius = int(round((w2 + h2) * 0.25))
                cv2.circle(frame, eye_center, radius, RED_COLOR, 4)

        if show_info_box:
            cv2.rectangle(frame, (10, offset), (200, offset + 150), WHITE_COLOR, 2)
            cv2.putText(
                frame,
                f"Face #{id}",
                (15, offset + 20),
                DEFAULT_FONT,
                INFO_TEXT_SIZE,
                WHITE_COLOR,
                2,
            )

            OVERLAY_SIZE = 128
            picture_with_overlay = cv2.resize(picture_with_overlay, (OVERLAY_SIZE, OVERLAY_SIZE))
            frame[- 129 : -1, -129 - face_nr * 129 : -1 - face_nr * 129] = picture_with_overlay


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

        offset += 150
    return frame


def main_loop(cap, face_cascade, model, device, preprocessing, show_processing, webcam):
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
            frame, face_cascade, model, device, preprocessing, emotion_scores
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
    video_input: str,
    output_file: str,
    show_processing: bool,
):
    """Make video prediction using the model with the given name"""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    model, preprocessing, device = initialize_model(model_name)

    cap = initialize_cap(webcam, video_input)

    if record:
        out = initialize_out(cap, output_file)

    try:
        main_loop(
            cap, face_cascade, model, device, preprocessing, show_processing, webcam
        )
    except Exception as e:
        print(e)
    finally:
        cap.release()
        if record:
            out.release()
        cv2.destroyAllWindows()

    print(f"Video successfully saved as {output_file}")
