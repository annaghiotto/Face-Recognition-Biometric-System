from abc import ABC
from typing import List
from custom_types import Image


class Preprocessor(ABC):
    """
    Abstract base class for signal preprocessors.
    Provides a callable interface to apply preprocessing on a list of signals.
    """

    def __call__(self, signals: List[Image]) -> List[Image]:
        # Apply preprocessing to each signal in the list
        return [self.preprocess(signal) for signal in signals]

    def preprocess(self, signal: Image) -> Image:
        # Abstract method for preprocessing a single signal
        pass


class SimplePreprocessor(Preprocessor):
    """
    Simple preprocessor that returns the signal unchanged.
    """

    def preprocess(self, image: Image) -> Image:
        return image