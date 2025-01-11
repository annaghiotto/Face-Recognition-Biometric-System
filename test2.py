from Classifier import XGBoostClassifier
from DataSource import GetFaces
from FeatureExtractor import EignenFacesExtractor
from Preprocessor import SimplePreprocessor
from utils import train_test_split
import numpy as np
faces = [face for face in GetFaces('att_faces', SimplePreprocessor(), EignenFacesExtractor())]


for face in faces:
    print(np.array(face.templates).shape)


train, test = train_test_split(faces, 0.2)

classifier = XGBoostClassifier(threshold=0.5)

classifier.fit(train, train)

# Evaluate the classifier on the test set
accuracy, eer, eer_threshold, auc = classifier.evaluate(test)
print(f"Accuracy: {accuracy}")
if eer is not None:
    print(f"EER: {eer}")
    print(f"EER Threshold: {eer_threshold}")
if auc is not None:
    print(f"AUC: {auc}")