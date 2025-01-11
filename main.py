from experiment import run_experiments, run_experiments_signals
from Preprocessor import SimplePreprocessor
from FeatureExtractor import EignenFacesExtractor

if __name__ == "__main__":
    run_experiments_signals(
        preprocessor_feature_combinations=[
            (SimplePreprocessor(), EignenFacesExtractor())
        ]
    )