from dataclasses import dataclass
from typing import List

from FeatureExtractor import FeatureExtractor
from custom_types import Signal, Template
from Preprocessor import Preprocessor


@dataclass
class TemplatesFactory:
    """
    Factory class for creating templates from raw signals using a specified
    preprocessor and feature extractor.
    """
    preprocessor: "Preprocessor"
    feature_extractor: "FeatureExtractor"

    def from_signals(self, signals: List[Signal]) -> List[Template]:
        """
        Preprocesses raw signals and extracts features to create templates.

        Parameters:
            signals (List[Signal]): List of raw signals.

        Returns:
            List[Template]: List of processed templates with extracted features.
        """
        # Apply preprocessor to signals, then extract features to create templates
        return self.feature_extractor(self.preprocessor(signals))
