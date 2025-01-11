import pickle
from dataclasses import dataclass
import numpy as np


@dataclass
class FaceEncoder:
    mean_face: np.ndarray
    eigenfaces: np.ndarray

    def encode(self, face):
        face = face - self.mean_face
        face = face.reshape(-1)
        return np.dot(face, self.eigenfaces.T)

    def decode(self, encoded_face):
        face = np.dot(encoded_face, self.eigenfaces)
        face = face.reshape(self.mean_face.shape)
        face += self.mean_face
        return face


def load_face_encoder() -> FaceEncoder:
    with open("encoder.pkl", "rb") as f:
        enc = pickle.load(f)  # type: FaceEncoder
    return enc


encoder = load_face_encoder()
