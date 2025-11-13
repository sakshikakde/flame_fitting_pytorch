import mediapipe as mp
import cv2
import numpy as np
from matplotlib import pyplot as plt

def extract_mediapipe_lmk3d(image):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            print("No face landmarks detected.")
            return None

        face_landmarks = results.multi_face_landmarks[0]
        lmk3d = []
        for landmark in face_landmarks.landmark:
            lmk3d.append([landmark.x * image.shape[1], landmark.y * image.shape[0], landmark.z * image.shape[1]])
        lmk3d = np.array(lmk3d)
        return lmk3d
    
def align_with_flame_axes(lmk3d):
    # roatate along x by -90
    lmk3d = lmk3d - np.mean(lmk3d, axis=0)  # center the landmarks
    rotation_matrix = np.array([[1, 0, 0],
                                [0, 0, 1],
                                [0, -1, 0]])
    lmk3d = lmk3d.dot(rotation_matrix.T)
    lmk3d = lmk3d / 1000.0  # convert to meters
    return lmk3d
    
def plot_landmarks(image, lmk3d, save_file="landmarks.png"):
    annotated_image = image.copy()
    h, w = image.shape[:2]
    for id, lm in enumerate(lmk3d):
        x, y = int(lm[0]), int(lm[1])
        # Draw a small circle for each landmark
        cv2.circle(annotated_image, (x, y), 1, (255, 0, 0), thickness=2)

    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig(save_file, bbox_inches='tight', pad_inches=0)
    plt.close()