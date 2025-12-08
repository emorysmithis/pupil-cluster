import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from preprocess import preprocess_data

def categorize_attention(num_correct):
    """Categorize attention level based on NumCorrect"""
    if num_correct <= 2:
        return 'low'  # low attention
    elif num_correct == 3:
        return 'medium'  # medium attention
    else:  # 4-6 attention
        return 'high'  # high attention

def extract_comprehensive_features(mask_dataframes):
    """
    Extract various features
    - Statistical features
    - Time series features
    - Frequency domain features
    - Change rate features
    """
    features_list = []
    attention_labels = []
    num_correct_list = []
    
    for idx, trial_df in enumerate(mask_dataframes):
        data_left = trial_df['DilationPupilLeftEye'].values
        data_right = trial_df['DilationPupilRightEye'].values
        
        # Check for NaN values
        if np.isnan(data_left).any() or np.isnan(data_right).any():
            continue
        
        num_correct = np.unique(trial_df['NumCorrect']).item()
        attention_label = categorize_attention(num_correct)
        
        feature_dict = {}
        
        # ===== Basic statistical features =====
        # Left eye
        feature_dict['left_mean'] = np.mean(data_left)
        feature_dict['left_std'] = np.std(data_left)
        feature_dict['left_median'] = np.median(data_left)
        feature_dict['left_max'] = np.max(data_left)
        feature_dict['left_min'] = np.min(data_left)
        feature_dict['left_range'] = np.max(data_left) - np.min(data_left)
        feature_dict['left_skew'] = stats.skew(data_left)
        feature_dict['left_kurtosis'] = stats.kurtosis(data_left)
        feature_dict['left_q25'] = np.percentile(data_left, 25)
        feature_dict['left_q75'] = np.percentile(data_left, 75)
        feature_dict['left_iqr'] = feature_dict['left_q75'] - feature_dict['left_q25']
        
        # Right eye features
        feature_dict['right_mean'] = np.mean(data_right)
        feature_dict['right_std'] = np.std(data_right)
        feature_dict['right_median'] = np.median(data_right)
        feature_dict['right_max'] = np.max(data_right)
        feature_dict['right_min'] = np.min(data_right)
        feature_dict['right_range'] = np.max(data_right) - np.min(data_right)
        feature_dict['right_skew'] = stats.skew(data_right)
        feature_dict['right_kurtosis'] = stats.kurtosis(data_right)
        feature_dict['right_q25'] = np.percentile(data_right, 25)
        feature_dict['right_q75'] = np.percentile(data_right, 75)
        feature_dict['right_iqr'] = feature_dict['right_q75'] - feature_dict['right_q25']
        
        # Bilateral difference
        feature_dict['bilateral_diff_mean'] = np.mean(np.abs(data_left - data_right))
        feature_dict['bilateral_diff_std'] = np.std(np.abs(data_left - data_right))
        feature_dict['bilateral_corr'] = np.corrcoef(data_left, data_right)[0, 1]
        
        # ===== Time series feature =====
        # Change rate (first derivative)
        diff_left = np.diff(data_left)
        diff_right = np.diff(data_right)
        
        feature_dict['left_diff_mean'] = np.mean(diff_left)
        feature_dict['left_diff_std'] = np.std(diff_left)
        feature_dict['left_diff_max'] = np.max(np.abs(diff_left))
        feature_dict['right_diff_mean'] = np.mean(diff_right)
        feature_dict['right_diff_std'] = np.std(diff_right)
        feature_dict['right_diff_max'] = np.max(np.abs(diff_right))
        
        # Second derivative of change rate - acceleration
        diff2_left = np.diff(diff_left)
        diff2_right = np.diff(diff_right)
        feature_dict['left_diff2_std'] = np.std(diff2_left)
        feature_dict['right_diff2_std'] = np.std(diff2_right)
        
        # Peak related features
        feature_dict['left_peak_idx'] = np.argmax(data_left)  # peak index
        feature_dict['left_peak_time_ratio'] = feature_dict['left_peak_idx'] / len(data_left)
        feature_dict['right_peak_idx'] = np.argmax(data_right)
        feature_dict['right_peak_time_ratio'] = feature_dict['right_peak_idx'] / len(data_right)
        
        # Post-peak decrease rate
        if feature_dict['left_peak_idx'] < len(data_left) - 1:
            post_peak_left = data_left[feature_dict['left_peak_idx']:]
            feature_dict['left_post_peak_slope'] = np.mean(np.diff(post_peak_left)) if len(post_peak_left) > 1 else 0
        else:
            feature_dict['left_post_peak_slope'] = 0
            
        if feature_dict['right_peak_idx'] < len(data_right) - 1:
            post_peak_right = data_right[feature_dict['right_peak_idx']:]
            feature_dict['right_post_peak_slope'] = np.mean(np.diff(post_peak_right)) if len(post_peak_right) > 1 else 0
        else:
            feature_dict['right_post_peak_slope'] = 0
        
        # ===== Section-wise feature =====
        # Early (0-33%), middle (33-66%), late (66-100%)
        n = len(data_left)
        early_left = data_left[:n//3]
        mid_left = data_left[n//3:2*n//3]
        late_left = data_left[2*n//3:]
        
        feature_dict['left_early_mean'] = np.mean(early_left)
        feature_dict['left_mid_mean'] = np.mean(mid_left)
        feature_dict['left_late_mean'] = np.mean(late_left)
        feature_dict['left_early_to_late'] = feature_dict['left_late_mean'] - feature_dict['left_early_mean']
        
        early_right = data_right[:n//3]
        mid_right = data_right[n//3:2*n//3]
        late_right = data_right[2*n//3:]
        
        feature_dict['right_early_mean'] = np.mean(early_right)
        feature_dict['right_mid_mean'] = np.mean(mid_right)
        feature_dict['right_late_mean'] = np.mean(late_right)
        feature_dict['right_early_to_late'] = feature_dict['right_late_mean'] - feature_dict['right_early_mean']
        
        # ===== Variability feature =====
        # Coefficient of Variation (CV) - coefficient of variation
        feature_dict['left_cv'] = feature_dict['left_std'] / (feature_dict['left_mean'] + 1e-10)
        feature_dict['right_cv'] = feature_dict['right_std'] / (feature_dict['right_mean'] + 1e-10)
        
        # Autocorrelation (lag=1) - autocorrelation
        if len(data_left) > 1:
            feature_dict['left_autocorr'] = np.corrcoef(data_left[:-1], data_left[1:])[0, 1]
            feature_dict['right_autocorr'] = np.corrcoef(data_right[:-1], data_right[1:])[0, 1]
        else:
            feature_dict['left_autocorr'] = 0
            feature_dict['right_autocorr'] = 0
        
        features_list.append(feature_dict)
        attention_labels.append(attention_label)
        num_correct_list.append(num_correct)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    features_df['attention_group'] = attention_labels
    features_df['num_correct'] = num_correct_list
    
    return features_df

def compare_features_by_group(features_df, outpath='feature_comparison.png'):
    """Group feature distribution comparison visualization"""
    feature_cols = [col for col in features_df.columns if col not in ['attention_group', 'num_correct']]
    n_features = len(feature_cols)
    
    # Select important features (top 20)
    if n_features > 20:
        # Calculate importance using ANOVA F-score
        X = features_df[feature_cols].values
        y = features_df['attention_group'].map({'low': 0, 'medium': 1, 'high': 2}).values
        f_scores, p_values = f_classif(X, y)
        
        # Select top 20
        top_indices = np.argsort(f_scores)[-20:][::-1]
        selected_features = [feature_cols[i] for i in top_indices]
    else:
        selected_features = feature_cols
    
    n_selected = len(selected_features)
    n_cols = 4
    n_rows = (n_selected + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_selected == 1 else axes
    
    for idx, feature in enumerate(selected_features):
        ax = axes[idx]
        
        # Group-wise data
        low_data = features_df[features_df['attention_group'] == 'low'][feature].dropna()
        medium_data = features_df[features_df['attention_group'] == 'medium'][feature].dropna()
        high_data = features_df[features_df['attention_group'] == 'high'][feature].dropna()
        
        # Box plot
        bp = ax.boxplot([low_data, medium_data, high_data], 
                        labels=['Low (0-2)', 'Medium (3)', 'High (4-6)'],
                        patch_artist=True)
        
        # Set colors
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title(feature, fontsize=10)
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Statistical significance display
        # ANOVA test
        groups = [low_data, medium_data, high_data]
        if all(len(g) > 0 for g in groups):
            f_stat, p_val = stats.f_oneway(*groups)
            if p_val < 0.001:
                ax.text(0.5, 0.95, '***', transform=ax.transAxes, 
                       ha='center', va='top', fontsize=12, color='red')
            elif p_val < 0.01:
                ax.text(0.5, 0.95, '**', transform=ax.transAxes, 
                       ha='center', va='top', fontsize=12, color='orange')
            elif p_val < 0.05:
                ax.text(0.5, 0.95, '*', transform=ax.transAxes, 
                       ha='center', va='top', fontsize=12, color='yellow')
    
    # Remove empty subplots
    for idx in range(n_selected, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Feature comparison saved to {outpath}")

def feature_importance_analysis(features_df, outpath='feature_importance.png'):
    """Feature importance analysis and visualization"""
    feature_cols = [col for col in features_df.columns if col not in ['attention_group', 'num_correct']]
    
    X = features_df[feature_cols].values
    y = features_df['attention_group'].map({'low': 0, 'medium': 1, 'high': 2}).values
    
    # Calculate ANOVA F-score
    f_scores, p_values = f_classif(X, y)
    
    # Result summary
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'f_score': f_scores,
        'p_value': p_values
    }).sort_values('f_score', ascending=False)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Top 20 F-score
    top_n = min(20, len(importance_df))
    top_features = importance_df.head(top_n)
    
    ax1.barh(range(len(top_features)), top_features['f_score'].values)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'].values, fontsize=9)
    ax1.set_xlabel('F-score')
    ax1.set_title(f'Top {top_n} Features by F-score')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # P-value visualization
    significant = importance_df[importance_df['p_value'] < 0.05]
    ax2.scatter(importance_df['f_score'], -np.log10(importance_df['p_value'] + 1e-10), 
               alpha=0.6, s=50)
    ax2.scatter(significant['f_score'], -np.log10(significant['p_value'] + 1e-10), 
               color='red', alpha=0.8, s=50, label='p < 0.05')
    ax2.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p = 0.05')
    ax2.set_xlabel('F-score')
    ax2.set_ylabel('-log10(p-value)')
    ax2.set_title('Feature Importance: F-score vs P-value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save to CSV
    csv_path = outpath.replace('.png', '.csv')
    importance_df.to_csv(csv_path, index=False)
    print(f"Feature importance saved to {outpath} and {csv_path}")
    
    return importance_df

def plot_group_time_series(mask_dataframes, outpath='group_time_series.png', n_samples_per_group=10):
    """Group time series pattern comparison"""
    # Group data by attention level
    groups = {'low': [], 'medium': [], 'high': []}
    
    for trial_df in mask_dataframes:
        num_correct = np.unique(trial_df['NumCorrect']).item()
        group = categorize_attention(num_correct)
        data = trial_df['DilationPupilLeftEye'].values
        if not np.isnan(data).any():
            groups[group].append(data)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'low': 'lightblue', 'medium': 'lightgreen', 'high': 'lightcoral'}
    
    for idx, (group_name, group_data) in enumerate(groups.items()):
        ax = axes[idx]
        
            # Sampling
        n_samples = min(n_samples_per_group, len(group_data))
        if n_samples > 0:
            sampled_indices = np.random.choice(len(group_data), n_samples, replace=False)
            sampled_data = [group_data[i] for i in sampled_indices]
            
            # Individual time series plot
            for ts in sampled_data:
                ax.plot(ts, alpha=0.3, color=colors[group_name], linewidth=0.8)
            
            # Mean time series
            mean_ts = np.mean(sampled_data, axis=0)
            std_ts = np.std(sampled_data, axis=0)
            ax.plot(mean_ts, color='black', linewidth=2, label='Mean')
            ax.fill_between(range(len(mean_ts)), 
                           mean_ts - std_ts, 
                           mean_ts + std_ts, 
                           alpha=0.2, color='black', label='±1 SD')
        
        ax.set_title(f'{group_name.upper()} Attention (n={len(group_data)})')
        ax.set_xlabel('Time Bin')
        ax.set_ylabel('Dilation Pupil Diameter')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Group time series comparison saved to {outpath}")

def generate_feature_report(features_df, outpath='feature_report.txt'):
    """Feature analysis report generation"""
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("FEATURE ENGINEERING REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Group distribution
        f.write("Group Distribution:\n")
        f.write("-" * 40 + "\n")
        group_counts = features_df['attention_group'].value_counts()
        for group, count in group_counts.items():
            f.write(f"{group.upper()}: {count} samples\n")
        f.write("\n")
        
        # Feature statistical comparison
        feature_cols = [col for col in features_df.columns if col not in ['attention_group', 'num_correct']]
        f.write(f"Total Features Extracted: {len(feature_cols)}\n\n")
        
        # Statistical feature comparison
        f.write("Group-wise Statistics (Top 10 Features):\n")
        f.write("-" * 40 + "\n")
        
        X = features_df[feature_cols].values
        y = features_df['attention_group'].map({'low': 0, 'medium': 1, 'high': 2}).values
        f_scores, p_values = f_classif(X, y)
        
        top_indices = np.argsort(f_scores)[-10:][::-1]
        for idx in top_indices:
            feature = feature_cols[idx]
            f.write(f"\n{feature}:\n")
            f.write(f"  F-score: {f_scores[idx]:.4f}, p-value: {p_values[idx]:.4e}\n")
            
            for group in ['low', 'medium', 'high']:
                group_data = features_df[features_df['attention_group'] == group][feature]
                f.write(f"  {group.upper()}: mean={group_data.mean():.4f}, std={group_data.std():.4f}\n")
        
    print(f"Feature report saved to {outpath}")

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    mask_dataframes = preprocess_data()
    
    # Extract features
    print("Extracting features...")
    features_df = extract_comprehensive_features(mask_dataframes)
    print(f"Extracted {len(features_df)} samples with {len(features_df.columns)-2} features")
    
    # Compare group time series
    print("Plotting group time series...")
    plot_group_time_series(mask_dataframes, 'group_time_series.png')
    
    # Compare features by group
    print("Comparing features by group...")
    compare_features_by_group(features_df, 'feature_comparison.png')
    
    # Analyze feature importance
    print("Analyzing feature importance...")
    importance_df = feature_importance_analysis(features_df, 'feature_importance.png')
    
    # Generate report
    print("Generating report...")
    generate_feature_report(features_df, 'feature_report.txt')
    
    print("\nFeature engineering complete!")
    print(f"Top 5 most important features:")
    print(importance_df.head(5)[['feature', 'f_score', 'p_value']])

