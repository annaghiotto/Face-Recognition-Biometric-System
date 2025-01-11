import os
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bars
from Classifier import XGBoostClassifier
from DataSource import GetFaces
from utils import train_test_split, k_fold_split
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_results(results_df: pd.DataFrame, filename="experiment_results.png"):
    """
    Plot the experiment results as bar charts for multiple metrics on a single plot.

    Parameters:
        results_df (pd.DataFrame): DataFrame containing experiment results.
        filename (str): The file path where the plot image will be saved.
    """
    sns.set(style="whitegrid")

    metrics = ['accuracy', 'eer', 'auc']
    metric_titles = {
        'accuracy': 'Accuracy',
        'eer': 'Equal Error Rate (EER)',
        'auc': 'Area Under ROC Curve (AUC)'
    }

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharey=False)

    for ax, metric in zip(axes, metrics):
        sns.barplot(
            data=results_df,
            x='feature_extractor',
            y=metric,
            hue='data_source',  # Use data source as hue
            ax=ax
        )
        ax.set_title(f'Comparison of {metric_titles[metric]}', fontsize=16)
        ax.set_xlabel('Feature Extractor', fontsize=14)
        ax.set_ylabel(metric_titles[metric], fontsize=14)
        ax.legend(title='Data Source', fontsize=10, title_fontsize=12)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


# Usage example:
# plot_results(results_df, filename="experiment_results.png")


@dataclass
class ExperimentResult:
    data_source: str
    preprocessor: str
    feature_extractor: str
    accuracy: Optional[float]
    eer: Optional[float]
    auc: Optional[float]


def process_combination(args):
    """
    Processes a single combination of preprocessor and feature extractor.
    Returns the ExperimentResult for each data source.
    """
    preprocessor, feature_extractor, n_folds = args
    preprocessor_name = type(preprocessor).__name__
    feature_extractor_name = type(feature_extractor).__name__

    results = []

    # Load data for both ECGID and SB sources
    ecg_id = GetFaces('ecg-id-database-1.0.0', preprocessor, feature_extractor)
    data_ecg_id = list(ecg_id.generator())
    data_sb = list(GetFaces('SB_ECGDatabase_01', preprocessor, feature_extractor).generator())

    for data_source_name, data in [("ECGID", data_ecg_id), ("SB", data_sb)]:
        # Handle k-fold cross-validation
        if n_folds == 1:
            try:
                train_data, test_data = train_test_split(data, test_size=0.2)
                folds = [(train_data, test_data)]
            except Exception as e:
                print(f"Error during train-test split for {data_source_name}: {e}")
                continue
        else:
            try:
                folds = k_fold_split(data, n_folds)
            except Exception as e:
                print(f"Error during k-fold split for {data_source_name}: {e}")
                continue

        accuracy_list = []
        eer_list = []
        auc_list = []

        for fold_num, (train, test) in enumerate(folds, start=1):
            try:
                classifier = XGBoostClassifier(threshold=0.5)
                classifier.fit(train, test)
                accuracy, eer, eer_threshold, auc_score = classifier.evaluate(test)
                accuracy_list.append(accuracy)
                eer_list.append(eer)
                auc_list.append(auc_score)
            except Exception as e:
                print(f"Error during evaluation in fold {fold_num} for {data_source_name}: {e}")
                continue

        avg_accuracy = np.mean(accuracy_list) if accuracy_list else None
        avg_eer = np.mean(eer_list) if eer_list else None
        avg_auc = np.mean(auc_list) if auc_list else None

        result = ExperimentResult(
            data_source=data_source_name,
            preprocessor=preprocessor_name,
            feature_extractor=feature_extractor_name,
            accuracy=avg_accuracy,
            eer=avg_eer,
            auc=avg_auc
        )
        results.append(result)


    return results


def run_experiments(preprocessor_feature_combinations: List[Tuple], n_folds: int = 1) -> pd.DataFrame:
    """
    Run experiments for each combination of Preprocessor and FeatureExtractor across
    SB and ECG-ID data sources using multiprocessing.

    Parameters:
        preprocessor_feature_combinations (List[Tuple]): List of (Preprocessor, FeatureExtractor) tuples.
        n_folds (int): Number of folds for cross-validation. Default is 1 (train-test split).

    Returns:
        pd.DataFrame: A DataFrame containing the results of all experiments.
    """
    with Pool(cpu_count()) as pool:
        all_results = []

        # Run experiments in parallel
        for result_set in tqdm(pool.imap_unordered(process_combination,
                                                   [(preprocessor, feature_extractor, n_folds)
                                                    for preprocessor, feature_extractor in
                                                    preprocessor_feature_combinations]),
                               total=len(preprocessor_feature_combinations), desc="Running experiments"):
            all_results.extend(result_set)

    # Convert results to DataFrame
    results_df = pd.DataFrame([r.__dict__ for r in all_results])

    # Plot the results
    plot_results(results_df)

    return results_df


def plot_signals_two_samples(
    example_samples, preprocessed_samples, extracted_samples,
    output_dir, preprocessor_name, extractor_name
):
    """
    Plot and save each signal separately for two samples with a dark background and custom colors.

    Parameters:
        example_samples (list of array-like): List of original example signals (two samples expected).
        preprocessed_samples (list of array-like): List of preprocessed signals (two samples expected).
        extracted_samples (list of array-like): List of feature-extracted signals (two samples expected, can be 3D).
        output_dir (str): Directory to save the plots.
        preprocessor_name (str): Name of the preprocessor used.
        extractor_name (str): Name of the feature extractor used.
    """
    # Ensure inputs are NumPy arrays
    example_samples = [np.array(s) for s in example_samples]
    preprocessed_samples = [np.array(s) for s in preprocessed_samples]
    extracted_samples = [np.array(s) for s in extracted_samples]

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set a dark background style
    plt.style.use('dark_background')

    colors = ['#FB47DE', '#8C2BBF']  # Palette for extracted features

    for i in range(2):  # Loop through two samples
        # Plot the original example sample
        plt.figure(figsize=(8, 4))
        plt.plot(example_samples[i][0], color='#FB47DE')
        plt.title(f"Original Example Signal {i + 1}", color='white')
        plt.savefig(os.path.join(output_dir, f"example_sample_{i + 1}_{preprocessor_name}.png"))
        plt.close()

        # Plot the preprocessed sample
        plt.figure(figsize=(8, 4))
        plt.plot(preprocessed_samples[i][0], color='#8C2BBF')
        plt.title(f"Preprocessed Signal {i + 1}", color='white')
        plt.savefig(os.path.join(output_dir, f"preprocessed_sample_{i + 1}_{preprocessor_name}.png"))
        plt.close()

        # Plot scatter plots for extracted features
        plt.figure(figsize=(8, 4))
        if extracted_samples[i].ndim == 3:  # Handle 3D data
            for j in range(extracted_samples[i].shape[1]):  # Iterate over the second dimension
                plt.scatter(
                    np.arange(extracted_samples[i][0, j, :].shape[0]),  # X-axis: sample indices
                    extracted_samples[i][0, j, :],  # Y-axis: feature values
                    color=colors[j % len(colors)], alpha=0.6
                )
        elif extracted_samples[i].ndim == 2:  # Handle 2D data
            for j in range(extracted_samples[i].shape[0]):
                plt.scatter(
                    np.arange(extracted_samples[i][j, :].shape[0]),  # X-axis: sample indices
                    extracted_samples[i][j, :],  # Y-axis: feature values
                    color=colors[j % len(colors)], alpha=0.6
                )
        else:  # Handle 1D data
            plt.scatter(
                np.arange(extracted_samples[i].shape[0]),  # X-axis: sample indices
                extracted_samples[i],  # Y-axis: feature values
                color=colors[0], alpha=0.6
            )
        plt.title(f"Extracted Features Signal {i + 1}", color='white')
        plt.savefig(os.path.join(output_dir, f"extracted_features_{i + 1}_{extractor_name}.png"))
        plt.close()



def process_combination_signal(preprocessor, feature_extractor):
    preprocessor_name = type(preprocessor).__name__
    feature_extractor_name = type(feature_extractor).__name__

    example_sample_generator = GetFaces('att_faces', preprocessor, feature_extractor, raw=True).generator()
    sample1 = [next(example_sample_generator)]
    sample2 = [next(example_sample_generator)]

    sample1_preprocessed = preprocessor(sample1)
    sample2_preprocessed = preprocessor(sample2)

    sample1_extracted = feature_extractor(sample1_preprocessed)
    sample2_extracted = feature_extractor(sample2_preprocessed)

    plot_signals_two_samples([sample1, sample2], [sample1_preprocessed, sample2_preprocessed], [sample1_extracted, sample2_extracted], "plots",  preprocessor_name, feature_extractor_name)


def run_experiments_signals(preprocessor_feature_combinations: List[Tuple]):

    for idx, (preprocessor, feature_extractor) in enumerate(
        tqdm(preprocessor_feature_combinations, desc="Running experiments")
    ):
        process_combination_signal(preprocessor, feature_extractor)
