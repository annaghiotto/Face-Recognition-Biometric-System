from abc import ABC
from dataclasses import dataclass
from typing import List
from custom_types import Image, Template
from encoder import encoder

@dataclass
class FeatureExtractor(ABC):
    """
    Abstract base class for feature extractors.
    """

    def __post_init__(self):
        super().__init__()

    def __call__(self, images: List[Image]) -> List[Template]:
        return [self.extract(image) for image in images]

    def extract(self, image: Image) -> Template:
        pass


class EignenFacesExtractor(FeatureExtractor):
    """
    Extracts Eigenfaces from images.
    """

    def extract(self, image: Image) -> Template:
        return encoder.encode(image)
