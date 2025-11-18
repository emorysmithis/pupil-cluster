from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from preprocess import preprocess_data
from cluster import time_series_kmeans_clustering, feature_kmeans_clustering, time_series_clustering_wavelet

def pca_features(features_scaled):
    # reduce the time series data to 2D
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_scaled.reshape(features_scaled.shape[0], -1))
    return pca, features_2d

def plot_clusters(cluster_labels, features_scaled, outpath):
    if features_scaled.shape[1] != 2:
        pca, features_2d = pca_features(features_scaled)
    else:
        pca = None
    
    # 2D scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Cluster')
    if pca is not None:
        plt.xlabel(f'PC1 (explained variance: {pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 (explained variance: {pca.explained_variance_ratio_[1]:.2%})')
    plt.title('Time Series Clustering Visualization (2D PCA)')
    plt.grid(True, alpha=0.3)
    plt.savefig(outpath)

def count_performance(cluster_labels, num_correct, outpath):
    concat = np.stack((cluster_labels, num_correct)).T
    df = pd.DataFrame(concat, columns=['cluster_label', 'num_correct'])
    count_0 = df[df['cluster_label'] == 0]['num_correct'].value_counts().sort_index()
    count_1 = df[df['cluster_label'] == 1]['num_correct'].value_counts().sort_index()

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    bars_0 = plt.bar(count_0.index, count_0.values, label='Cluster 0', alpha=0.7)
    len_0 = len(count_0)
    # 각 바 위에 count 값 표시
    for x, y in zip(count_0.index, count_0.values):
        plt.text(x, y, str(y), ha='center', va='bottom', fontsize=9)
    plt.xlabel('Number of Correct')
    plt.ylabel('Count')
    plt.title('Cluster 0')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.subplot(1, 2, 2)
    bars_1 = plt.bar(count_1.index, count_1.values, label='Cluster 1', alpha=0.7, color='orange')
    len_1 = len(count_1)
    # 각 바 위에 count 값 표시
    for x, y in zip(count_1.index, count_1.values):
        plt.text(x, y, str(y), ha='center', va='bottom', fontsize=9)
    plt.xlabel('Number of Correct')
    plt.ylabel('Count')
    plt.title('Cluster 1')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(outpath)

    

if __name__ == "__main__":
    mask_dataframes = preprocess_data()
    cluster_labels, features_scaled, num_correct = time_series_clustering_wavelet(mask_dataframes)
    # cluster_labels, features_scaled, num_correct = feature_kmeans_clustering(mask_dataframes)
    # count_performance(cluster_labels, num_correct)
    plot_clusters(cluster_labels, features_scaled, 'time_series_wavelet_2_pca_20.png')
    count_performance(cluster_labels, num_correct, 'time_series_wavelet_2_pca_20_c.png')
    # cluster_labels, features_scaled = feature_kmeans_clustering(mask_dataframes)
    # plot_clusters(cluster_labels, features_scaled, 'feature_clusters.png')