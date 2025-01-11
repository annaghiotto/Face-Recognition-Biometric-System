from experiment import run_experiments, run_experiments_signals
from Preprocessor import BasicPreprocessor, SARModelPreprocessor
from FeatureExtractor import StatisticalTimeExtractor, DCTExtractor, PCAExtractor, SARModelExtractor, StatisticalTimeFreqExtractor, HMMExtractor

if __name__ == "__main__":
    run_experiments_signals(
        preprocessor_feature_combinations=[
            (BasicPreprocessor(), StatisticalTimeExtractor()),
            (BasicPreprocessor(), DCTExtractor()),
            (BasicPreprocessor(), PCAExtractor(5)),
            (SARModelPreprocessor(), SARModelExtractor()),
            (BasicPreprocessor(), StatisticalTimeFreqExtractor()),
            (BasicPreprocessor(), HMMExtractor())
        ]
    )