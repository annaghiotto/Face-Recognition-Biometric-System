from abc import ABC
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from Person import Person
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb


@dataclass
class Classifier(ABC):
    threshold: float

    def fit(self, train: List[Person], eval_set: List[Person]):
        pass

    def identify(self, person: Person) -> Optional[int]:
        pass

    def authenticate(self, person: Person) -> bool:
        pass


@dataclass
class XGBoostClassifier(Classifier):
    scaler: StandardScaler = field(default_factory=StandardScaler, init=False)
    label_encoder: LabelEncoder = field(default_factory=LabelEncoder, init=False)
    model: xgb.XGBClassifier = field(default=None, init=False)

    def __post_init__(self):
        # Initialize XGBoost model with specific parameters
        self.model = xgb.XGBClassifier(
            objective="multi:softmax",
            verbosity=2,
            eval_metric='mlogloss'
        )

    def preprocess_data(self, data: List[Person], fit_scaler: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data by extracting features and labels, handling missing values,
        scaling features, and encoding labels.

        Parameters:
            data (List[Person]): The input data to preprocess.
            fit_scaler (bool): Whether to fit the scaler on the data.

        Returns:
            X (np.ndarray): The preprocessed feature matrix.
            y (np.ndarray): The encoded labels.
        """
        # Extract features and labels
        X = [
            template
            for person in data
            for template in person.templates_flat
        ]
        y = [
            person.uid
            for person in data
            for _ in person.templates_flat
        ]

        X = np.array(X)
        y = np.array(y)

        # Handle missing values if any
        if np.isnan(X).any():
            # Replace NaNs with the mean of each feature
            col_means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_means, inds[1])

        # Scale features
        if fit_scaler:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        # Encode labels
        if fit_scaler and not hasattr(self.label_encoder, 'classes_'):
            y = self.label_encoder.fit_transform(y)
        else:
            y = self.label_encoder.transform(y)

        return X, y

    def fit(self, train: List[Person], eval_set: List[Person]):
        """
        Fit the XGBoost classifier on the preprocessed training data.

        Parameters:
            train (List[Person]): The training data.
            eval_set (List[Person]): The evaluation data for monitoring.
        """
        X_train, y_train = self.preprocess_data(train, fit_scaler=True)
        X_eval, y_eval = self.preprocess_data(eval_set, fit_scaler=False)

        # Train the model, monitoring performance on the evaluation set
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_eval, y_eval)],
            verbose=True
        )

    def evaluate(self, eval_set: List[Person]):
        """
        Evaluate the model's performance, calculating accuracy, EER, and AUC.

        Parameters:
            eval_set (List[Person]): The evaluation dataset.

        Returns:
            accuracy (float): Accuracy score.
            eer (float | None): Equal error rate (EER).
            eer_threshold (float | None): Threshold at EER.
            auc (float | None): Area under the ROC curve.
        """
        X_eval, y_eval = self.preprocess_data(eval_set, fit_scaler=False)
        y_pred = self.model.predict(X_eval)
        accuracy = accuracy_score(y_eval, y_pred)

        y_true = []
        y_scores = []

        # Calculate scores for ROC analysis
        for person in eval_set:
            for template in person.templates_flat:
                prediction_proba = self.model.predict_proba([template])[0]
                if prediction_proba[person.uid] >= self.threshold:
                    y_true.append(1)
                    y_scores.append(prediction_proba[person.uid])
                elif any(element >= self.threshold for element in prediction_proba):
                    for element in prediction_proba:
                        if element >= self.threshold:
                            y_true.append(0)
                            y_scores.append(element)

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # Filter out zero or near-zero differences in FPR for stability
        fpr, tpr = np.array(fpr), np.array(tpr)
        unique_fpr_indices = np.where(np.diff(fpr) > 1e-6)[0]
        if len(unique_fpr_indices) < 2:
            print("Error: Insufficient unique FPR values to compute ROC curve.")
            eer = None
            eer_threshold = None
        else:
            fnr = 1 - tpr
            # Find the threshold where the difference between FPR and FNR is smallest
            eer_index = np.nanargmin(np.abs(fpr - fnr))
            eer_threshold = thresholds[eer_index]
            eer = fpr[eer_index]

        try:
            auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            print("Error: Insufficient unique FPR values to compute ROC curve.")
            auc = None

        return accuracy, eer, eer_threshold, auc

    def identify(self, person: Person) -> Optional[int]:
        """
        Identify the user based on the given person's ECG templates.

        Parameters:
            person (Person): The person to identify.

        Returns:
            int | None: The predicted user ID or None if no prediction meets the threshold.
        """
        if not person.templates_flat:
            return None

        X = np.array(person.templates_flat)

        # Handle missing values
        if np.isnan(X).any():
            col_means = self.scaler.mean_
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_means, inds[1])

        X_scaled = self.scaler.transform(X)
        prediction_proba = self.model.predict_proba(X_scaled)

        predicted_classes = []
        for proba in prediction_proba:
            predicted_class = np.argmax(proba)
            max_proba = proba[predicted_class]
            if max_proba >= self.threshold:
                predicted_classes.append(predicted_class)

        if not predicted_classes:
            return None

        return int(np.bincount(predicted_classes).argmax())

    def authenticate(self, person: Person) -> bool:
        """
        Authenticate the user based on a random template from the person's ECG.

        Parameters:
            person (Person): The person to authenticate.

        Returns:
            bool: True if authenticated successfully, False otherwise.
        """
        if not person.templates_flat:
            return False

        random_template = person.templates_flat[np.random.randint(len(person.templates_flat))]

        X = np.array([random_template])

        # Handle missing values
        if np.isnan(X).any():
            col_means = self.scaler.mean_
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_means, inds[1])

        X_scaled = self.scaler.transform(X)

        prediction_proba = self.model.predict_proba(X_scaled)[0]
        true_label_encoded = self.label_encoder.transform([person.uid])[0]

        return prediction_proba[true_label_encoded] >= self.threshold