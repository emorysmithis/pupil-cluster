# SVM classifier pipeline for pupil time series (wavelet features)
import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)
from sklearn.pipeline import Pipeline
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from preprocess import preprocess_data

def _center_crop_or_pad(series, target_len=463):
    """center crop or symmetric pad (edge value)"""
    s = np.asarray(series, dtype=float)
    L = len(s)
    if L == target_len:
        return s.copy()
    if L > target_len:
        start = (L - target_len) // 2
        return s[start:start+target_len].copy()

def extract_wavelet_features(series_array, wavelet='db4', level=4):
    """extract wavelet features for each time series"""
    feats = []
    for s in series_array:
        coeffs = pywt.wavedec(s, wavelet=wavelet, level=level)
        vec = np.concatenate(coeffs)
        feats.append(vec)
    feats = np.vstack(feats)
    return feats

def prepare_dataset(mask_dataframes, target_len=463, wavelet='db4', level=4):
    """
    - Baseline processing is already done
    - NumCorrect -> binary (0-3 -> 0, 4-6 -> 1)
    """
    series_list = []
    labels = []
    ids = []
    for idx, df in enumerate(mask_dataframes):
        s = df['SmoothedDilationPupilLeftEye'].values
        s = _center_crop_or_pad(s, target_len)    # 길이 맞춤
        series_list.append(s)
        val = int(np.unique(df['NumCorrect']).item())
        group = 0 if val <= 3 else 1
        labels.append(group)
        ids.append((idx, val))
    X_series = np.vstack(series_list)   # (n, target_len)
    X_wave = extract_wavelet_features(X_series, wavelet=wavelet, level=level)
    return X_series, X_wave, np.array(labels), ids

# ---------- main training ----------
def train_svm_on_wavelet(mask_dataframes,
                         target_len=463,
                         wavelet='db4',
                         level=4,
                         pca_dim=20,
                         test_size=0.2,
                         random_state=42,
                         cv_folds=5,
                         scoring_metric='f1',
                         class_weight=None):
    """
    Returns dict with model, grid, test arrays, predictions, metrics.
    class_weight: None or 'balanced' (use if label imbalance)
    """
    X_series, X_wave, y, ids = prepare_dataset(mask_dataframes, target_len, wavelet, level)
    print("Wavelet feature shape:", X_wave.shape)

    # stratified split
    X_train, X_test, y_train, y_test, series_train, series_test = train_test_split(
        X_wave, y, X_series, test_size=test_size, stratify=y, random_state=random_state)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=pca_dim)),
        ('svm', SVC(kernel='rbf', probability=True, class_weight=class_weight))
    ])

    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.01, 0.001]
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring_metric, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except:
        auc = np.nan

    print("\nTest results:")
    print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ROC plot (optional)
    try:
        from sklearn.metrics import roc_curve, auc as aucfunc
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = aucfunc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC curve'); plt.legend()
        plt.show()
    except Exception:
        pass

    return {
        'model': best_model,
        'grid': grid,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'series_test': series_test,
        'ids': ids
    }

if __name__ == "__main__":
    mask_dataframes = preprocess_data()
    result = train_svm_on_wavelet(mask_dataframes, target_len=463, wavelet='db4', level=4, pca_dim=20)
    print(result)
