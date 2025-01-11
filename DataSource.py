import os
import numpy as np
import cv2

from FeatureExtractor import FeatureExtractor
from Person import PersonFactory
from Preprocessor import Preprocessor
from Templates import TemplatesFactory
from TqdmIteratorBase import TqdmIteratorBase


class GetEcgIDData(TqdmIteratorBase):
    """
    Data source for ECG-ID Data.
    Iterates over persons and yields Person objects with their records.
    """

    def __init__(self, filename, preprocessor: Preprocessor, feature_extractor: FeatureExtractor, desc='GetEcgIDData', raw=False,
                 **tqdm_kwargs):
        """
        Initializes the GetEcgIDData iterator.

        :param filename: Directory containing ECG-ID data.
        :param preprocessor: Preprocessor instance.
        :param feature_extractor: FeatureExtractor instance.
        :param desc: Description for the tqdm progress bar.
        :param tqdm_kwargs: Additional keyword arguments for tqdm.
        """
        self.filename = filename
        self.raw = raw
        self.person_factory = PersonFactory(
            TemplatesFactory(preprocessor, feature_extractor)
        )

        # Determine total number of persons
        self.person_dirs = [
            d for d in os.listdir(self.filename)
            if os.path.isdir(os.path.join(self.filename, d))
        ]
        total_persons = len(self.person_dirs)

        super().__init__(desc=desc, total=total_persons, **tqdm_kwargs)

    def generator(self):
        """
        Generator that yields Person objects.
        """
        for person_dir in sorted(self.person_dirs):
            person_path = os.path.join(self.filename, person_dir)
            person_number_str = person_dir.split('s')[-1]
            try:
                person_number = int(person_number_str)
            except ValueError:
                print(f"Invalid person directory name: {person_dir}. Skipping.")
                continue

            person_signals = []
            record = 1
            while True:
                record_filename = os.path.join(person_path, f"{record}.pgm")
                try:
                    img = cv2.imread(record_filename, cv2.IMREAD_GRAYSCALE)
                    img = 255 - img  # Invert image
                    # Assuming the first column contains the desired signal
                    person_signals.append(img)
                    record += 1
                except FileNotFoundError:
                    if person_signals:
                        if self.raw:
                            yield person_signals[0]
                        else:
                            yield self.person_factory.create(person_signals, person_number - 1)
                    else:
                        print(f"No records found for {person_dir}. Skipping.")
                    break
                except Exception as e:
                    print(f"Error reading {record_filename}: {e}. Skipping this record.")
                    record += 1
