import cv2
import dlib
import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.capi.onnxruntime_pybind11_state import NoSuchFile

from gradcam import overlay
from inference import load_model_and_preprocessing
from utils import LABEL_TO_STR

COLOR_BLUE = (255, 0, 0)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_WHITE = (255, 255, 255)
INFO_TEXT_SIZE = 0.7
DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX

try:
    SHAPE_PREDICTOR = dlib.shape_predictor(
        "models/shape_predictor_68_face_landmarks.dat")
except RuntimeError:
    SHAPE_PREDICTOR = None

"""
Disclaimer: Parts of the face detection code are based on the following repository:
https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/

It is a ultra light face detection model, which is based on the RFBNet architecture.

The model can be downloaded from:
https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/models/onnx/version-RFB-320.onnx
"""
ONNX_PATH = "models/version-RFB-320.onnx"
try:
    ORT_SESSION = ort.InferenceSession(ONNX_PATH)
    INPUT_NAME = ORT_SESSION.get_inputs()[0].name
except NoSuchFile:
    ORT_SESSION = None
    INPUT_NAME = None


def initialize_model(model_name: str):
    """Initialize the model with the given name

    Args:
        model_name (str) : the name of the model to be used

    Returns (tuple) : the model, the preprocessing function, and the device
    """

    _, model, preprocessing = load_model_and_preprocessing(model_name)
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, preprocessing, device


def initialize_cap(webcam: bool, cam_id: int, video_input: str):
    """Initialize the camera or video input.

    Args:
        webcam (bool) : whether to use the webcam.
        cam_id (int) : the camera id.
        video_input (str) : the video input file, if not using the webcam.

    Returns (cv2.VideoCapture) : the camera or video input.
    """

    if webcam:
        cap = cv2.VideoCapture(cam_id)
    else:
        cap = cv2.VideoCapture(video_input)

    if not cap.isOpened():
        raise IOError("Cannot open camera or video file")

    return cap


def initialize_out(cap: cv2.VideoCapture, file: str, codec: str):
    """Initialize the video output.

    Args:
        cap (cv2.VideoCapture) : the camera or video input.
        file (str) : the output file.
        codec (str) : the codec to be used.

    Returns (cv2.VideoWriter) : the video output.
    """

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter(file, fourcc, fps, (width, height))


def top_prediction_with_label(predictions: torch.Tensor):
    """Get the top prediction with its label and score.

    Args:
        predictions (torch.Tensor) : the predictions.

    Returns (tuple) : the label and the score.
    """
    top_prediction = torch.argmax(predictions, 1)
    label = LABEL_TO_STR[top_prediction.item()]

    score = predictions[0][top_prediction].item()
    return label, score


def predict_expressions(
        image: np.ndarray,
        model: torch.nn.Module,
        preprocessing: torch.nn.Module,
        device: str,
):
    """Predict the expressions of the given image.

    Args:
        image (np.ndarray) : the image to be predicted.
        model (torch.nn.Module) : the model to be used.
        preprocessing (torch.nn.Module) : the preprocessing function.
        device (str) : the device to be used.

    Returns (torch.Tensor) : the predictions.
    """
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


def draw_landmarks(
        frame: np.ndarray,
        box: np.ndarray,
):
    """Draw the landmarks on the frame using dlib.

    Args:
        frame (np.ndarray) : the frame to be drawn on.
        box (np.ndarray) : the bounding box of the face.
    """
    landmarks = SHAPE_PREDICTOR(
        frame, dlib.rectangle(box[0], box[1], box[2], box[3]))

    for i in range(0, 68):
        x, y = landmarks.part(i).x, landmarks.part(i).y
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)


def draw_explainability_overlay(
        frame: np.ndarray,
        id: int,
        picture_with_overlay: np.ndarray,
):
    """Draw the explainability overlay on the frame.

    Args:
        frame (np.ndarray) : the frame to be drawn on.
        id (int) : the id of the face.
        picture_with_overlay (np.ndarray) : the picture with the overlay.
    """

    OVERLAY_SIZE = 128
    picture_with_overlay = cv2.resize(
        picture_with_overlay, (OVERLAY_SIZE, OVERLAY_SIZE)
    )
    cv2.putText(
        frame,
        f"Face #{id + 1}",
        (frame.shape[1] - 129 - id * 128 + 10, frame.shape[0] - 138),
        DEFAULT_FONT,
        0.5,
        COLOR_BLUE,
        2,
    )
    frame[-129: -1, -129 - id * 129: -1 - id * 129] = picture_with_overlay


def draw_info_box(
        frame: np.ndarray,
        id: int,
        predictions: torch.Tensor,
):
    """Draw the info box with the predictions.

    Args:
        frame (np.ndarray) : the frame to be drawn on.
        id (int) : the id of the face.
        predictions (torch.Tensor) : the predictions.
    """
    offset = id * 150 + 60
    cv2.rectangle(frame, (10, offset), (220, offset + 150), COLOR_WHITE, 2)
    cv2.putText(
        frame,
        f"Face #{id + 1}",
        (15, offset + 20),
        DEFAULT_FONT,
        INFO_TEXT_SIZE,
        COLOR_WHITE,
        2,
    )

    for i, (expression, score) in enumerate(
            zip(LABEL_TO_STR.values(), predictions[0])
    ):
        cv2.putText(
            frame,
            f"{expression}: {score:.2f}",
            (15, offset + 40 + i * 20),
            DEFAULT_FONT,
            INFO_TEXT_SIZE,
            COLOR_WHITE,
            2,
        )


def get_expression_color(expression: str):
    """Return the color for the given expression.

    Args:
        expression (str): the expression.

    Returns (tuple): the color.
    """
    return COLOR_GREEN if expression in ["happiness", "surprise"] else COLOR_RED


def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.

    Code from: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/vision/utils/box_utils.py

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Code from: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/vision/utils/box_utils.py

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.

    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])

    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """Apply Hard Non-Maximum Suppression to filter relevant boxes based on scores.

    Code from: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/vision/utils/box_utils.py

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.

    Returns:
            picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]

    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break

        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def predict_face(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    """Perform post-processing for face detection.

    Code from: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/run_video_face_detect_onnx.py

    Args:
        width: Width of the image.
        height: Height of the image.
        confidences: Confidence scores for each class.
        boxes: Bounding boxes.
        prob_threshold: Probability threshold for filtering detections.
        iou_threshold: Intersection over union threshold for Hard NMS.
        top_k: Keep top_k detections.

    Returns:
        Picked bounding box probabilities and labels.
    """
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []

    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]

        if probs.shape[0] == 0:
            continue

        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate(
            [subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(
            box_probs, iou_threshold=iou_threshold, top_k=top_k)
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])

    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])

    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height

    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


def process_frame(
        frame: np.ndarray,
        model: torch.nn.Module,
        device: str,
        preprocessing: str,
        show_explainability: bool,
        show_landmarks: bool,
        show_info: bool,
):
    """Process the frame and return the frame with the predicted expressions, landmarks, and explainability.

    Args:
        frame (np.ndarray) : the frame to be processed.
        model (torch.nn.Module) : the model to be used.
        device (str) : the device to be used.
        preprocessing (str) : the preprocessing function.
        show_explainability (bool) : whether to show the explainability.
        show_landmarks (bool) : whether to show the landmarks.
        show_info (bool) : whether to show the info.
    """
    num_faces = 0

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 240))
    image = (image - np.array([127, 127, 127])) / 128
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)
    image = image.astype(np.float32)

    threshold = 0.7
    confidences, boxes = ORT_SESSION.run(None, {INPUT_NAME: image})
    boxes, _, _ = predict_face(
        frame.shape[1], frame.shape[0], confidences, boxes, threshold)

    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        mean_height = (box[3] - box[1]) / 2
        mean_width = (box[2] - box[0]) / 2
        center = (box[0] + mean_width, box[1] + mean_height)
        size = int(max(mean_height, mean_width))
        squared_box = [int(center[0] - size), int(center[1] - size),
                       int(center[0] + size), int(center[1] + size)]
        roi = frame[squared_box[1]: squared_box[3],
              squared_box[0]: squared_box[2]]
        if roi.size == 0:
            continue

        num_faces += 1

        if show_explainability:
            predictions, picture_with_overlay = overlay(roi, model)
            predictions = torch.nn.functional.softmax(predictions, dim=1)
        else:
            predictions = predict_expressions(
                roi, model, preprocessing, device)

        expression, _ = top_prediction_with_label(predictions)

        if show_info and i < 3:
            draw_info_box(frame, i, predictions)

        if show_explainability and i < 3:
            draw_explainability_overlay(frame, i, picture_with_overlay)

        if show_landmarks:
            draw_landmarks(frame, squared_box)

        cv2.putText(
            frame,
            f"Expression: {expression}",
            (squared_box[0], squared_box[1] - 10),
            DEFAULT_FONT,
            INFO_TEXT_SIZE,
            get_expression_color(expression),
            2,
        )
        cv2.rectangle(
            frame,
            (squared_box[0], squared_box[1]),
            (squared_box[2], squared_box[3]),
            COLOR_BLUE,
            2,
        )

    if show_info:
        cv2.putText(
            frame,
            f"Number of faces: {num_faces}",
            (10, 30),
            DEFAULT_FONT,
            1.1,
            COLOR_WHITE,
            2,
        )

    return frame


def main_loop(
        cap: cv2.VideoCapture,
        model: torch.nn.Module,
        device: str,
        preprocessing: str,
        webcam: bool,
        show_processing: bool,
        show_explainability: bool,
        show_landmarks: bool,
        show_info: bool,
        out: cv2.VideoWriter,
):
    """Main loop for video prediction.

    Args:
        cap (cv2.VideoCapture) : the camera or video input.
        model (torch.nn.Module) : the model to be used.
        device (str) : the device to be used.
        preprocessing (str) : the preprocessing function.
        webcam (bool) : whether to use the webcam.
        show_processing (bool) : whether to show the processing.
        show_explainability (bool) : whether to show the explainability.
        show_landmarks (bool) : whether to show the landmarks.
        show_info (bool) : whether to show the info.
        out (cv2.VideoWriter) : the video output.
    """
    print("Starting video prediction...")

    while True:
        has_frame, frame = cap.read()

        if not has_frame:
            break

        if webcam:
            frame = cv2.flip(frame, 1)

        processed_frame = process_frame(
            frame,
            model,
            device,
            preprocessing,
            show_explainability,
            show_landmarks,
            show_info,
        )

        if show_processing:
            cv2.namedWindow("Facial Expression Recognition", cv2.WINDOW_NORMAL)
            cv2.imshow("Facial Expression Recognition", processed_frame)

        if not webcam:
            out.write(processed_frame)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame % fps == 0:
                print(
                    f"Processed {current_frame} / {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}"
                )

        q_pressed = cv2.waitKey(1) == ord("q")
        window_closed = (
                show_processing
                and cv2.getWindowProperty(
            "Facial Expression Recognition", cv2.WND_PROP_VISIBLE
        )
                < 1
        )
        if q_pressed or window_closed:
            print("Video prediction interrupted.")
            break

    print("Video prediction finished.")


def make_video_prediction(
        model_name: str,
        webcam: bool,
        cam_id: int,
        video_input: str,
        output_file: str,
        show_processing: bool,
        show_explainability: bool,
        show_landmarks: bool,
        show_info: bool,
        codec: str,
):
    """Make video prediction using the model with the given name.

    Args:
        model_name (str) : the name of the model to be used.
        webcam (bool) : whether to use the webcam.
        cam_id (int) : the camera id.
        video_input (str) : the video input file, if not using the webcam.
        output_file (str) : the output file.
        show_processing (bool) : whether to show the processing.
        show_explainability (bool) : whether to show the explainability.
        show_landmarks (bool) : whether to show the landmarks.
        show_info (bool) : whether to show the info.
        codec (str) : the fourcc codec to be used for the output video.
    """
    model, preprocessing, device = initialize_model(model_name)
    cap = initialize_cap(webcam, cam_id, video_input)
    out = initialize_out(cap, output_file, codec) if not webcam else None

    main_loop(
        cap,
        model,
        device,
        preprocessing,
        webcam,
        show_processing,
        show_explainability,
        show_landmarks,
        show_info,
        out,
    )
    cap.release()
    if not webcam:
        out.release()
        print(f"Video successfully saved as {output_file}")
    cv2.destroyAllWindows()
