import os
import cv2

from FeatureExtractor import FeatureExtractor
from Person import PersonFactory
from Preprocessor import Preprocessor
from Templates import TemplatesFactory
from TqdmIteratorBase import TqdmIteratorBase


class GetFaces(TqdmIteratorBase):
    """
    Data source for ECG-ID Data.
    Iterates over persons and yields Person objects with their records.
    """

    def __init__(self, dirname, preprocessor: Preprocessor, feature_extractor: FeatureExtractor, desc='GetFaces', raw=False,
                 **tqdm_kwargs):
        """
        Initializes the GetEcgIDData iterator.

        :param dirname: Directory containing ECG-ID data.
        :param preprocessor: Preprocessor instance.
        :param feature_extractor: FeatureExtractor instance.
        :param desc: Description for the tqdm progress bar.
        :param tqdm_kwargs: Additional keyword arguments for tqdm.
        """
        self.dirname = dirname
        self.raw = raw
        self.person_factory = PersonFactory(
            TemplatesFactory(preprocessor, feature_extractor)
        )

        # Determine total number of persons
        self.person_dirs = [
            d for d in os.listdir(self.dirname)
            if os.path.isdir(os.path.join(self.dirname, d))
        ]
        total_persons = len(self.person_dirs)

        super().__init__(desc=desc, total=total_persons, **tqdm_kwargs)

    def generator(self):
        """
        Generator that yields Person objects or raw signals for all files in the directory.
        """
        for person_dir in sorted(os.listdir(self.dirname)):
            person_path = os.path.join(self.dirname, person_dir)
            if not os.path.isdir(person_path):
                continue  # Skip if it's not a directory

            try:
                person_number = int(person_dir.split('s')[-1])  # Extract person number
            except ValueError:
                print(f"Invalid person directory name: {person_dir}. Skipping.")
                continue

            person_signals = []
            for filename in sorted(os.listdir(person_path)):
                record_filename = os.path.join(person_path, filename)
                try:
                    img = cv2.imread(record_filename, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        raise ValueError(f"Could not read image file {record_filename}.")
                    img = 255 - img  # Invert image
                    person_signals.append(img)
                except Exception as e:
                    print(f"Error processing file {record_filename}: {e}. Skipping.")
                    continue

            if person_signals:
                if self.raw:
                    yield person_signals[0]
                else:
                    yield self.person_factory.create(person_signals, person_number - 1)
            else:
                print(f"No valid records found for {person_dir}. Skipping.")