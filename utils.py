from typing import List, Tuple
from Person import Person


def train_test_split(data: List[Person], test_size: float) -> (List[Person], List[Person]):
    """
    Splits the data into training and testing sets for each Person.

    :param data: List of Person objects.
    :param test_size: Proportion of each Person's data to include in the test set.
    :return: Tuple containing lists of Person objects for training and testing.
    """
    # Perform train-test split on each Person object and aggregate results
    return list(zip(*[person.train_test_split(test_size) for person in data]))


def k_fold_split(data: List[Person], k: int) -> List[Tuple[List[Person], List[Person]]]:
    """
    Perform K-Fold split on a list of Person objects.

    :param data: List of Person objects.
    :param k: Number of folds.
    :return: A list where each item is a tuple containing lists of Person objects 
             for the train and test sets in each fold.
    """
    all_folds = []
    for _ in range(k):
        all_folds.append(([], []))  # Initialize each fold as a tuple of train and test lists

    # Distribute each Person's K-Fold split results across the folds
    for person in data:
        person_fold = person.k_fold_split(k)
        i = 0
        for fold in person_fold:
            all_folds[i][0].append(fold[0])  # Append training data for fold i
            all_folds[i][1].append(fold[1])  # Append testing data for fold i
            i += 1

    return all_folds
