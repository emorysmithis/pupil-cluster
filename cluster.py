import numpy as np
import pandas as pd
import pywt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from preprocess import preprocess_data
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax

def feature_kmeans_clustering(mask_dataframes, n_clusters=3):
    # use mean and median of smoothed dilation pupil diameter as features
    features = []
    num_correct = []
    for idx, trial_df in enumerate(mask_dataframes):
        data_left = trial_df['DilationPupilLeftEye'].values
        data_right = trial_df['DilationPupilRightEye'].values
        if np.isnan(data_left).any() or np.isnan(data_right).any():
            print(f"Trial {idx} has NaN values")
        n = len(data_left)
        early_left = data_left[:n//3]
        early_right = data_right[:n//3]
        
        num_correct.append(np.unique(trial_df['NumCorrect']).item())
        feature_vector = [
            np.mean(early_left),           # mean
            np.mean(early_right),           # mean
            np.argmax(data_left),  # peak index
            np.argmax(data_left) / len(data_left), # peak time ratio
            np.argmax(data_right),
            np.argmax(data_right) / len(data_right),
        ]
        features.append(feature_vector)
    
    features = np.array(features)
    
    # normalize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    return cluster_labels, features_scaled, num_correct

def time_series_kmeans_clustering(mask_dataframes, max_iter=10, n_clusters=3):
    # use the smoothed dilation pupil diameter as a time series
    features = []
    num_correct = []
    for idx, trial_df in enumerate(mask_dataframes):
        data = trial_df['DilationPupilLeftEye'].values
        if np.isnan(data).any().item():
            print(f"Trial {idx} has NaN values")
        num_correct.append(np.unique(trial_df['NumCorrect']).item())
        features.append(data)
    
    features = np.array(features)
    scaler = TimeSeriesScalerMinMax()
    features_scaled = scaler.fit_transform(features)
    kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", max_iter=max_iter, random_state=0)
    cluster_labels = kmeans.fit_predict(features_scaled)
    return cluster_labels, features_scaled, num_correct


def time_series_clustering_wavelet(mask_dataframes):
    target_len = 463
    wavelet = 'db4'
    level = 4
    n_clusters = 2
    pca_dim = 20

    series_list = []
    labels = [] # num correct label

    # 1) clean and align the length
    for idx, trial_df in enumerate(mask_dataframes):
        s = trial_df['SmoothedDilationPupilLeftEye'].values.astype(float)
        # crop/pad
        if len(s) > target_len:
            start = (len(s) - target_len) // 2
            s = s[start:start+target_len]
        series_list.append(s)
        labels.append(int(np.unique(trial_df['NumCorrect']).item()))

    X = np.vstack(series_list)  # (n, 463)

    # 2) Wavelet transform
    wavelet_features = []
    for s in X:
        coeffs = pywt.wavedec(s, wavelet=wavelet, level=level)
        vec = np.concatenate(coeffs)  # concatenate all scale coefficients into 1D
        wavelet_features.append(vec)

    wavelet_features = np.vstack(wavelet_features)

    # 3) normalize
    scaler = StandardScaler()
    wavelet_features = scaler.fit_transform(wavelet_features)

    # 4) PCA (recommended because the length of the wavelet vector is long)
    if pca_dim is not None:
        pca = PCA(n_components=pca_dim)
        wavelet_pca = pca.fit_transform(wavelet_features)
    else:
        wavelet_pca = wavelet_features

    # 5) K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(wavelet_pca)

    return cluster_labels, wavelet_pca, labels


if __name__ == "__main__":
    mask_dataframes = preprocess_data()
    # cluster_labels, features_scaled = feature_kmeans_clustering(mask_dataframes)
    cluster_labels, features_scaled = time_series_kmeans_clustering(mask_dataframes)
    print(np.unique(cluster_labels, return_counts=True))