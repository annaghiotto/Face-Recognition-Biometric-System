import cv2
import pickle
from encoder import FaceEncoder


def load_face_encoder() -> FaceEncoder:
    with open("encoder.pkl", "rb") as f:
        enc = pickle.load(f)  # type: FaceEncoder
    return enc


image = cv2.imread("att_faces/s1/1.pgm", cv2.IMREAD_GRAYSCALE)
encoder = load_face_encoder()

encoded_face = encoder.encode(image)
