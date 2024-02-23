import os

import cv2

FACE_CASCADE = None


def clip_face(img):
    global FACE_CASCADE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if FACE_CASCADE is None:
        if os.path.exists("models/haarcascade_frontalface_default.xml"):
            FACE_CASCADE = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
        else:
            raise FileNotFoundError("Could not find haarcascade_frontalface_default.xml")

    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    was_clipped = False
    if len(faces) > 0:
        x, y, w, h = faces[0]
        img = img[y:y + h, x:x + w]
        was_clipped = True
    return cv2.resize(img, (64, 64)), was_clipped


def clip_affect_net_faces(input_path, output_dir, use_rafdb_format):
    clipped_files = 0
    total_files = 0
    for root, dirs, files in os.walk(input_path):
        print(f"Processing {root}")

        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                img = cv2.imread(os.path.join(root, file))
                img, was_clipped = clip_face(img)
                if was_clipped:
                    clipped_files += 1
                total_files += 1

                if use_rafdb_format:
                    emotion_label = os.path.basename(root)
                    filename, ext = os.path.splitext(file)
                    new_filename = f"{filename}_{emotion_label}{ext}"
                    output_path = os.path.join(output_dir, new_filename)
                else:
                    relative_path = os.path.relpath(root, input_path)
                    output_subdir = os.path.join(output_dir, relative_path)

                    if not os.path.exists(output_subdir):
                        os.makedirs(output_subdir)

                    output_path = os.path.join(output_subdir, file)

                cv2.imwrite(output_path, img)

    print(f"Clipped {clipped_files} out of {total_files} files.")
