import math
from dataclasses import dataclass

from sklearn.model_selection import KFold
from custom_types import Image, Template
from Templates import TemplatesFactory
from typing import List, Tuple
from itertools import chain


@dataclass
class Person:
    """
    Represents a person with a list of templates (signal features) and a unique identifier.
    """
    templates: List["Template"]
    uid: int

    def train_test_split(self, test_size: float) -> ("Person", "Person"):
        """
        Splits the templates into training and testing sets based on the test_size ratio.

        Parameters:
            test_size (float): Ratio of templates to be used in the test set.

        Returns:
            Tuple[Person, Person]: The training and testing Person objects.
        """
        n_templates = len(self.templates)

        # Handle case where only one template is available
        if n_templates == 1:
            n_templates = len(self.templates[0])
            n_test = math.ceil(n_templates * test_size)
            return Person([self.templates[0][n_test:]], self.uid), Person([self.templates[0][:n_test]], self.uid)

        n_test = math.ceil(n_templates * test_size)
        return Person(self.templates[n_test:], self.uid), Person(self.templates[:n_test], self.uid)

    def k_fold_split(self, k: int) -> List[Tuple["Person", "Person"]]:
        """
        Performs K-Fold split on templates and returns a list of train/test splits for each fold.

        Parameters:
            k (int): Number of folds for cross-validation.

        Returns:
            List[Tuple[Person, Person]]: List of train/test Person objects for each fold.
        """
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        folds = []
        n_templates = len(self.templates)

        if n_templates < k:
            # Case where templates are fewer than the number of folds
            train_fold = [[] for _ in range(k)]
            test_fold = [[] for _ in range(k)]
            for j in range(n_templates):
                i = 0
                for train_idx, test_idx in kf.split(self.templates[j]):
                    train_fold[i].append([self.templates[j][i] for i in train_idx])
                    test_fold[i].append([self.templates[j][i] for i in test_idx])
                    i += 1

            for i in range(k):
                folds.append(
                    (Person(train_fold[i], self.uid), Person(test_fold[i], self.uid))
                )

            return folds
        else:
            # Standard case with enough templates for the specified folds
            for train_idx, test_idx in kf.split(self.templates):
                train_templates = [self.templates[i] for i in train_idx]
                test_templates = [self.templates[i] for i in test_idx]
                folds.append(
                    (Person(train_templates, self.uid), Person(test_templates, self.uid))
                )

            return folds


@dataclass
class PersonFactory:
    """
    Factory class to create a Person object from raw signals using a TemplatesFactory instance.
    """
    templates_factory: "TemplatesFactory"

    def create(self, signals: List[Image], uid: int) -> Person:
        # Generates Person instance by creating templates from signals
        return Person(self.templates_factory.from_signals(signals), uid)
