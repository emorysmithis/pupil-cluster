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

def plot_clusters(cluster_labels, features_scaled, outpath, n_clusters=3):
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

def count_performance(cluster_labels, num_correct, outpath, n_clusters=3):
    concat = np.stack((cluster_labels, num_correct)).T
    df = pd.DataFrame(concat, columns=['cluster_label', 'num_correct'])

    plt.figure(figsize=(12, 5))
    
    for i in range(n_clusters):
        plt.subplot(1, n_clusters, i+1)
        count_i = df[df['cluster_label'] == i]['num_correct'].value_counts().sort_index()
        plt.bar(count_i.index, count_i.values, label=f'Cluster {i}', alpha=0.7)
        for x, y in zip(count_i.index, count_i.values):
            plt.text(x, y, str(y), ha='center', va='bottom', fontsize=9)
        plt.xlabel('Number of Correct')
        plt.ylabel('Count')
        plt.title(f'Cluster {i}')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(outpath)


def plot_binned_data(filtered_all_trial_dataframes, mask_dataframes, outpath, n_trials=3):
    """
    Binning 전후 데이터를 비교 시각화
    
    Args:
        filtered_all_trial_dataframes: 원본 trial 데이터프레임 리스트
        mask_dataframes: Binned된 mask 데이터프레임 리스트
        outpath: 저장할 파일 경로
        n_trials: 시각화할 trial 개수
    """
    # mask_dataframes에 해당하는 원본 데이터 찾기
    fig, axes = plt.subplots(n_trials, 2, figsize=(14, 4*n_trials))
    if n_trials == 1:
        axes = axes.reshape(1, -1)
    
    plotted = 0
    random_indices = np.random.choice(len(mask_dataframes), n_trials, replace=False)
    # Extract the binned dataframes on the selected random indices,
    # then iterate with enumerate so idx reflects 0..n_trials-1 and
    # binned_df is the actual DataFrame from mask_dataframes.
    selected_binned_dfs = [mask_dataframes[i] for i in random_indices]
    for idx, binned_df in enumerate(selected_binned_dfs):
        if plotted >= n_trials:
            break
            
        # 원본 데이터 찾기 (TrialId, Subject, Session으로 매칭)
        trial_id = binned_df['TrialId'].iloc[0]
        subject = binned_df['Subject'].iloc[0]
        session = binned_df['Session'].iloc[0]
        num_correct = binned_df['NumCorrect'].iloc[0]
        
        original_trial = None
        for trial_df in filtered_all_trial_dataframes:
            if (trial_df['TrialId'].iloc[0] == trial_id and 
                trial_df['Subject'].iloc[0] == subject and 
                trial_df['Session'].iloc[0] == session and
                trial_df['NumCorrect'].iloc[0] == num_correct):
                original_trial = trial_df
                break
        
        if original_trial is None:
            continue
        
        # Mask 부분만 추출
        original_mask = original_trial[original_trial["CurrentObject"] == "Mask"].copy()
        
        # 원본 데이터 플롯 (왼쪽)
        ax_orig = axes[plotted, 0]
        ax_orig.plot(original_mask['ElapsedTime'], original_mask['DiameterPupilLeftEye'], 
                     'b-', alpha=0.6, linewidth=0.5, label='Original (Left)')
        ax_orig.plot(original_mask['ElapsedTime'], original_mask['DiameterPupilRightEye'], 
                     'r-', alpha=0.6, linewidth=0.5, label='Original (Right)')
        ax_orig.set_xlabel('Elapsed Time (ms)')
        ax_orig.set_ylabel('Pupil Diameter')
        ax_orig.set_title(f'Trial {idx+1} - Before Binning\n(Subject: {subject}, Session: {session}, NumCorrect: {num_correct})')
        ax_orig.legend()
        ax_orig.grid(True, alpha=0.3)
        
        # Binned 데이터 플롯 (오른쪽)
        ax_binned = axes[plotted, 1]
        # time_bin을 ms로 변환 (원래는 100ms 단위)
        time_bin_ms = binned_df['time_bin'].values
        ax_binned.plot(time_bin_ms, binned_df['DiameterPupilLeftEye'], 
                      'b-o', alpha=0.7, markersize=4, label='Binned (Left)')
        ax_binned.plot(time_bin_ms, binned_df['DiameterPupilRightEye'], 
                      'r-o', alpha=0.7, markersize=4, label='Binned (Right)')
        ax_binned.set_xlabel('Time Bin (ms)')
        ax_binned.set_ylabel('Pupil Diameter (Mean per 100ms)')
        ax_binned.set_title(f'Trial {idx+1} - After Binning (40 bins)')
        ax_binned.legend()
        ax_binned.grid(True, alpha=0.3)
        
        plotted += 1
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


if __name__ == "__main__":
    mask_dataframes = list(preprocess_data(is_group_by_subject=True).values())[0]
    cluster_labels, features_scaled, num_correct = time_series_kmeans_clustering(mask_dataframes, n_clusters=3)
    # cluster_labels, features_scaled, num_correct = feature_kmeans_clustering(mask_dataframes, n_clusters=3)
    plot_clusters(cluster_labels, features_scaled, 'ts_clusters_3.png', n_clusters=3)
    count_performance(cluster_labels, num_correct, 'ts_clusters_c_3.png', n_clusters=3)