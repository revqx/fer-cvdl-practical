import os

import cv2
import torch

from inference import load_model_and_preprocessing
from utils import LABEL_TO_STR

FACE_CASCADE_PATH = "cascades/haarcascade_frontalface_default.xml"

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

def initialize_out(cap, file, codec='XVID'):
    """Initialize the video output"""
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter(file, fourcc, fps, (width, height))

def predict_emotion(image, model, preprocessing, device):
    """Predict the emotion of the given image"""
    resized_image = cv2.resize(image, (64, 64))
    normalized_image = resized_image / 255.0
    torch_image = torch.from_numpy(normalized_image).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)

    if preprocessing:
        torch_image = preprocessing(torch_image)

    with torch.no_grad():
        output = model(torch_image)
        predictions = torch.nn.functional.softmax(output, dim=1) 
        predicted_class = torch.argmax(predictions, 1)
        emotion = LABEL_TO_STR[predicted_class.item()]
        
        score = predictions[0][predicted_class].item()
        
    return emotion, score

def process_frame(frame, face_cascade, model, device, preprocessing, emotion_score):
    """Process the frame and return the frame with the predicted emotions"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    RECTANGLE_COLOR = (255, 0, 0)

    for (x, y, w, h) in faces:
        emotion, score = predict_emotion(frame[y:y + h, x:x + w], model, preprocessing, device)
        
        emotion_score[emotion]['total_score'] += score
        emotion_score[emotion]['count'] += 1
    
        cv2.rectangle(frame, (x, y), (x + w, y + h), RECTANGLE_COLOR, 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, RECTANGLE_COLOR, 2)

    return frame


def main_loop(cap, face_cascade, model, device, preprocessing, show_processing):
    """Main loop for video prediction"""
    print("Starting video prediction...")
    
    emotion_scores = {emotion: {'total_score': 0, 'count': 0} for emotion in LABEL_TO_STR.values()}

    while True:
        has_frame, frame = cap.read()

        if not has_frame:
            break

        
        processed_frame = process_frame(frame, face_cascade, model, device, preprocessing, emotion_scores)
        if show_processing:
            cv2.imshow('Facial Emotion Recognition', processed_frame)

        q_pressed = cv2.waitKey(1) == ord('q')
        window_closed = show_processing and cv2.getWindowProperty('Facial Emotion Recognition',
                                                                  cv2.WND_PROP_VISIBLE) < 1
        if q_pressed or window_closed:
            print("Video prediction interrupted.")
            break
    
    for emotion, score in emotion_scores.items():
        average_score = score['total_score'] / score['count'] if score['count'] > 0 else 0
        print(f"Emotion: {emotion}, Average Score: {average_score:.2f}")

def make_video_prediction(model_name: str, record: bool, webcam: bool, video_input: str, output_file: str, show_processing: bool):
    """Make video prediction using the model with the given name"""
    if not os.path.isfile(FACE_CASCADE_PATH):
        raise IOError(
            "Please download the haar cascade file from https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml and put it into /cascades")

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    model, preprocessing, device = initialize_model(model_name)

    cap = initialize_cap(webcam, video_input)
    
    if record:
        out = initialize_out(cap, output_file)

    try:
        main_loop(cap, face_cascade, model, device, preprocessing, show_processing)
    except Exception as e:
        print(e)
    finally:
        cap.release()
        if record:
            out.release()
        cv2.destroyAllWindows()

    print(f"Video successfully saved as {output_file}")
