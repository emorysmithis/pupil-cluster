# Machine Learning to Detect Working Memory Utilization from Pupillometry
FA25-EE-67004-01 Brain-Computer Interface class project


## Overview

This project processes pupillometry data from “Pupillometry tracks fluctuations in working memory performance” to achieve the below goals:
- Process raw psychology data to get pupil dilations, remove blinks, remove bad trials, etc
- Try an unsupervised machine learning model to see how well it can separate successful and unsuccessful trials
- Try a supervised machine learning model to see if it can classify the trials
- Analyze which features in the supervised machine learning model are helping versus hurting the model.

## Project Structure

### **Main files**

#### Exploring the dataset: `BCI_dilation.ipynb`
**Purpose**: Jupyter notebook for initial exploratory data analysis
- Conduct initial preprocessing
- Visualize whether data is separable or not to see if anomaly detection can work

#### Preprocessing the dataset: `preprocess.py`
**Purpose**: Core preprocessing pipeline for pupil diameter data
- **`call_all_filtered_trial_dataframes()`**: Loads raw trial data, splits by trial ID, removes metadata columns, filters out invalid trials (empty, low validity, missing pupil data)
- **`interpolate_diameter_pupil()`**: Interpolates missing pupil diameter values using PCHIP (piecewise cubic Hermite interpolation), with linear fallback. Filters out physiologically implausible values (< 1mm or > 6mm)
- **`dilation_preprocessing()`**: 
  - Extracts baseline pupil diameter from "Fixation" periods
  - Bins data into 100ms time windows (40 bins total)
  - Calculates pupil dilation as difference from baseline during "Mask" periods
- **`apply_low_pass_filter()`**: Applies 5Hz low-pass Butterworth filter to smooth dilation signals
- **`group_by_subject()`**: Groups preprocessed trials by subject ID
- **`preprocess_data()`**: Main entry point that orchestrates the entire preprocessing pipeline

#### Extracting Features and ANOVA analysis: `feature_engineering.py`
**Purpose**: Comprehensive feature extraction and statistical analysis; We found out there are severe overlapping in different attention levels
- **`categorize_attention()`**: Maps `NumCorrect` scores to attention levels (low: 0-2, medium: 3, high: 4-6)
- **`extract_comprehensive_features()`**: Extracts 50+ features per trial:
  - **Statistical**: mean, std, median, min, max, range, skewness, kurtosis, quartiles, IQR
  - **Time series**: first/second derivatives, peak indices, post-peak slopes
  - **Temporal sections**: early/middle/late period means and transitions
  - **Bilateral features**: inter-eye differences and correlations
  - **Variability**: coefficient of variation, autocorrelation
- **`compare_features_by_group()`**: Creates box plots comparing feature distributions across attention groups with ANOVA significance testing
- **`feature_importance_analysis()`**: Calculates ANOVA F-scores and p-values to identify most discriminative features, generates importance visualizations
- **`plot_group_time_series()`**: Visualizes average time series patterns for each attention group
- **`generate_feature_report()`**: Creates text report summarizing feature statistics and group differences

#### Unsupervised clustering: `cluster.py`
**Purpose**: Unsupervised clustering methods to discover patterns in pupil dilation data
- **`feature_kmeans_clustering()`**: 
  - Extracts hand-crafted features (early period means, peak indices, peak time ratios)
  - Applies K-Means clustering on standardized features
- **`time_series_kmeans_clustering()`**: 
  - Uses full time series as input
  - Applies time series-specific scaling (min-max normalization)
  - Performs K-Means clustering using Euclidean distance on time series
- **`time_series_clustering_wavelet()`**: 
  - Applies Discrete Wavelet Transform (Daubechies 4, level 4) to decompose time series
  - Uses PCA for dimensionality reduction
  - Performs K-Means clustering on wavelet coefficients

#### Pipieline for CNN (time-series features) + other crafted features: `ml_cnn_binary.py`
**Purpose**: 1D CNN for binary classification of attention levels
- **`CNN1DHybrid`**: PyTorch model combining:
  - 1D convolutional layers for time series processing
  - Fully connected layers for hand-crafted features
  - Concatenation of two values and classification
- **`extract_pupil_features()`**: Extracts velocity, acceleration, and other temporal features
- **`train_model()`**: Training loop with validation monitoring
- **`evaluate_accuracy()`**: Model evaluation with Testset


### **Other files**

#### `scripts/load_data.py`
**Purpose**: Initial data loading and validation
- Recursively finds all `.gazedata` files in a directory structure
- Validates that all files have consistent headers
- Combines multiple tab-separated files into a single DataFrame
- Drops gaze position and camera position columns
- Generates summary statistics (number of subjects, sessions, trials)
- Creates sample visualization plots for individual trials
- **Usage**: `python scripts/load_data.py <data_directory> <file_extension>`

#### `ml_cnn_regression.py`
**Purpose**: 1D CNN for regression task (predicting `NumCorrect` as continuous value)
- Similar architecture to `ml_cnn_binary.py` but with regression output
- Uses MSE loss instead of cross-entropy
- Predicts continuous attention scores rather than binary classes

#### `ml_es.py`
**Purpose**: Early stopping CNN implementation for regression
- Similar to `ml_cnn_regression.py` but includes early stopping to prevent overfitting
- Monitors validation loss and stops training when no improvement is detected

#### `ml_es_class.py`
**Purpose**: Early stopping CNN implementation for classification
- Similar to `ml_cnn_binary.py` but includes early stopping
- Binary classification with early stopping based on validation accuracy

#### `vis.py`
**Purpose**: Comprehensive visualization tools for clustering and analysis results
- **`pca_features()`**: Reduces high-dimensional features to 2D using PCA for visualization
- **`plot_clusters()`**: Creates 2D scatter plots of cluster assignments with PCA projection
- **`count_performance()`**: Visualizes distribution of `NumCorrect` scores within each cluster
- **`plot_binned_data()`**: Compares original vs. binned time series data side-by-side

#### `vis/` directory
Contains generated visualization outputs:
- Cluster visualizations (feature-based and time series-based)
- Feature importance plots
- Group time series comparisons
- Binned data comparisons

### **Output**

#### `feature_importance.csv`
CSV file containing ANOVA F-scores and p-values for all extracted features, sorted by importance.

#### `feature_report.txt`
Text report summarizing:
- Group distribution statistics
- Top discriminative features
- Group-wise feature statistics

#### `combined.csv`
Output from `load_data.py` containing all concatenated raw data from all `.gazedata` files.

## Dependencies

- **Data Processing**: pandas, numpy
- **Signal Processing**: scipy
- **Machine Learning**: scikit-learn, tslearn, torch
- **Visualization**: matplotlib
- **Statistical Analysis**: scipy.stats

## Data Format

Input files are tab-separated `.gazedata` files with columns including:
- `Subject`, `Session`, `TrialId`: Trial identifiers
- `TETTime`: Timestamp
- `DiameterPupilLeftEye`, `DiameterPupilRightEye`: Pupil diameters
- `ValidityLeftEye`, `ValidityRightEye`: Data validity flags
- `CurrentObject`: Task phase (e.g., "Fixation", "Mask")
- `NumCorrect`: Performance metric (0-6)

