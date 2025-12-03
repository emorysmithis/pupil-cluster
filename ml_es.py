import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import argparse
import itertools
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn as nn


def call_all_filtered_trial_dataframes(file_path='WM study/Eye data/*/*.gazedata'):
    all_trial_dataframes = []
    count = 0
    
    for filename in glob.glob(file_path):
        raw_dataframe = pd.read_csv(filename, sep='\t', low_memory=False)
        # split by trial id
        for trial_id in raw_dataframe['TrialId'].unique():
            trial_dataframe = raw_dataframe[raw_dataframe['TrialId'] == trial_id]
            # drop metadata
            trial_dataframe = trial_dataframe.drop(columns=[
            'ID', 'RTTime', 'TimestampSec', 'TimestampMicrosec'])
            # drop specific movements since they will probably not be generalizable
            trial_dataframe = trial_dataframe.drop(columns=[
                'XGazePosLeftEye', 'YGazePosLeftEye', 'XCameraPosLeftEye', 'YCameraPosLeftEye',
                'XGazePosRightEye', 'YGazePosRightEye', 'XCameraPosRightEye', 'YCameraPosRightEye',
                'DistanceLeftEye', 'DistanceRightEye'
            ])
            try:
                trial_dataframe = trial_dataframe.drop(columns=['RESP'])
                count += 1
            except:
                pass
            all_trial_dataframes.append(trial_dataframe)

    print(f"Num trials before removing any: {len(all_trial_dataframes)}")
    

    filtered_all_trial_dataframes = []
    for index, trial in enumerate(all_trial_dataframes):
        # 1. remove empty trials
        if trial.empty:
            continue
        # 2. remove trials where left pupil diameter is 0 more than threshold
        zero_threshold = len(trial) / 2 # half the time, 50%
        if sum(a > 0 for a in trial['DiameterPupilLeftEye']) < zero_threshold:
            continue

        # 3. remove trials where right pupil diameter is 0 more than threshold
        if sum(a > 0 for a in trial['DiameterPupilRightEye']) < zero_threshold:
            continue

        # 4. remove trials where validity of right eye is not valid more than threshold
        if sum(a == 0 for a in trial['ValidityRightEye']) < zero_threshold:
            continue

        # 5. remove trials where validity of left eye is not valid more than threshold
        if sum(a == 0 for a in trial['ValidityLeftEye']) < zero_threshold:
            continue

        trial['ElapsedTime'] = trial['TETTime'] - trial['TETTime'].iloc[0]
        filtered_all_trial_dataframes.append(trial)

    print(f"Num trials remaining after removing empty and half-empty trials: {len(filtered_all_trial_dataframes)}")
    return filtered_all_trial_dataframes

def interpolate_diameter_pupil(filtered_all_trial_dataframes):
    MAX = 6
    MIN = 1
    num_linear_fallback = 0

    for index, trial in enumerate(filtered_all_trial_dataframes):

        # ------------------------------------
        # Step 1: Mark out-of-range values as NaN
        # ------------------------------------
        left_raw = trial["DiameterPupilLeftEye"].copy()
        right_raw = trial["DiameterPupilRightEye"].copy()

        # Mark invalid samples (baseline) so we know where interpolation happens
        left_invalid = (left_raw < MIN) | (left_raw > MAX)
        right_invalid = (right_raw < MIN) | (right_raw > MAX)

        trial.loc[left_invalid, "DiameterPupilLeftEye"] = np.nan
        trial.loc[right_invalid, "DiameterPupilRightEye"] = np.nan

        # Blink mask BEFORE interpolation 
        trial["LeftBlinkMask"] = trial["DiameterPupilLeftEye"].isna()
        trial["RightBlinkMask"] = trial["DiameterPupilRightEye"].isna()

        # ------------------------------------
        # Step 2: Interpolate pupil signals
        # ------------------------------------
        try:
            trial["DiameterPupilLeftEye"] = trial["DiameterPupilLeftEye"].interpolate(method="pchip")
            trial["DiameterPupilRightEye"] = trial["DiameterPupilRightEye"].interpolate(method="pchip")
        except:
            num_linear_fallback += 1
            trial["DiameterPupilLeftEye"] = trial["DiameterPupilLeftEye"].interpolate()
            trial["DiameterPupilRightEye"] = trial["DiameterPupilRightEye"].interpolate()

        # Save back
        filtered_all_trial_dataframes[index] = trial

    return filtered_all_trial_dataframes

def dilation_preprocessing(filtered_all_trial_dataframes):
    mask_dataframes = []
    for index, trial_df in enumerate(filtered_all_trial_dataframes):
        fixation_df = trial_df[trial_df["CurrentObject"] == "Fixation"]

        # TODO: I don't think the notna is working because still getting some nan baselines 
        baseline_left_pupil = fixation_df[fixation_df["DiameterPupilLeftEye"].notna()]["DiameterPupilLeftEye"].mean() # get mean of pupil diameter, but don't include diameter == -1
        baseline_right_pupil = fixation_df[fixation_df["DiameterPupilRightEye"].notna()]["DiameterPupilRightEye"].mean() # get mean of pupil diameter, but don't include diameter == -1
        if pd.isna(baseline_left_pupil) or pd.isna(baseline_right_pupil):
            print(f"Skipping trial {index} due to NaN baseline")
            continue

        trial_df.loc[:, "DilationPupilRightEye"] = np.where(
            (trial_df["CurrentObject"] == "Mask") & (trial_df["DiameterPupilRightEye"].notna()),
            trial_df["DiameterPupilRightEye"] - baseline_right_pupil,
            np.nan)

        trial_df.loc[:, "DilationPupilLeftEye"] = np.where(
            (trial_df["CurrentObject"] == "Mask") & (trial_df["DiameterPupilLeftEye"].notna()),
            trial_df["DiameterPupilLeftEye"] - baseline_left_pupil,
            np.nan)

        mask_dataframes.append(trial_df.loc[trial_df["CurrentObject"] == "Mask"].copy())
    # mask_dataframes = mask_dataframes[1:] # remove the first trial because it is not a mask trial
    print(f"Num trials after removing all parts of trial except 'delay period': {len(mask_dataframes)}")
    return mask_dataframes

def apply_low_pass_filter(mask_dataframes):
    # Filter
    def butter_lowpass(cutoff, fs, order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def lowpass_filter(data, cutoff, fs, order=4):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    # parameters
    fs = 120  # sampling rate in Hz (change to your eye tracker rate)
    cutoff = 5  # cutoff frequency in Hz

    for i, trial_df in enumerate(mask_dataframes):
        mask_dataframes[i].loc[:, 'SmoothedDilationPupilLeftEye'] = lowpass_filter(
            mask_dataframes[i]['DilationPupilLeftEye'], cutoff, fs)
        mask_dataframes[i].loc[:, 'SmoothedDilationPupilRightEye'] = lowpass_filter(
            mask_dataframes[i]['DilationPupilRightEye'], cutoff, fs)

    return mask_dataframes

def plot_trial(filtered_all_trial_dataframes, random_numbers,ax1):
    for i in random_numbers:
        trial_df = filtered_all_trial_dataframes[i]
        for column in trial_df.columns:
            if column == 'DiameterPupilLeftEye' or column == 'DiameterPupilRightEye': # Exclude the label from plotting
                ax1.plot(trial_df['ElapsedTime'], trial_df[column], label=column)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Pupil Diameter (mm)')
        plt.title(f'Random Trial {i+1}: Subject {trial_df['Subject'].iloc[0]}, Trial: {trial_df["TrialId"].iloc[0]} (NumCorrect: {trial_df["NumCorrect"].iloc[0]})')
        ax1.legend(loc='upper left')

def confirm_no_nans(mask_dataframes): 
    smoothed_mask_dataframes = [] 
    for idx, trial_df in enumerate(mask_dataframes):
        data = trial_df['SmoothedDilationPupilLeftEye'].values
        if np.isnan(data).any().item():
            print(f"Skipping Trial {idx} because has NaN values")
            continue 
        data = trial_df['SmoothedDilationPupilRightEye'].values
        if np.isnan(data).any().item():
            print(f"Skipping Trial {idx} because has NaN values")
            continue
        trial_df["subject_session_trial"] = f"{trial_df['Subject'].iloc[0]}_{trial_df['Session'].iloc[0]}_{trial_df['TrialId'].iloc[0]}"
        smoothed_mask_dataframes.append(trial_df)
    print(f"Num trials after removing trials with NaN in SmoothedDilationPupilLeftEye: {len(smoothed_mask_dataframes)}")
    return smoothed_mask_dataframes

def extract_pupil_features(all_trials, fs=120):

    feature_rows = []

    for trial_idx, df in enumerate(all_trials):

        row = {}

        # -------------------------------------------------------
        # Helper: compute run lengths (for blink durations)
        # -------------------------------------------------------
        def longest_run(mask):
            runs = [sum(group) for val, group in itertools.groupby(mask) if val == 1]
            return max(runs) if runs else 0

        def mean_run(mask):
            runs = [sum(group) for val, group in itertools.groupby(mask) if val == 1]
            return np.mean(runs) if runs else 0

        # -------------------------------------------------------
        # Loop over LEFT and RIGHT
        # -------------------------------------------------------
        for side in ["Left", "Right"]:

            sig = df[f"SmoothedDilationPupil{side}Eye"]
            interp_sig = df[f"DilationPupil{side}Eye"]
            mask = df[f"{side}BlinkMask"]

            # valid = True means ORIGINAL data was valid
            valid = ~mask

            # If too little valid data, skip trial
            if valid.sum() < 5:
                print(f"Warning: trial {trial_idx} has too few valid samples on {side}.")
                continue

            # ---------------------------------------------------
            # Temporal & value features (masked)
            # ---------------------------------------------------
            row[f"{side}_mean"] = sig[valid].mean()
            row[f"{side}_std"] = sig[valid].std()
            row[f"{side}_peak"] = sig[valid].max()
            row[f"{side}_min"] = sig[valid].min()
            row[f"{side}_range"] = sig[valid].max() - sig[valid].min()

            # time to peak (in seconds)
            peak_idx = sig[valid].idxmax()
            row[f"{side}_time_to_peak"] = peak_idx / fs

            # AUC (trapezoid)
            row[f"{side}_auc"] = np.trapezoid(sig[valid], dx=1/fs)

            # First derivative (slope)
            deriv = np.gradient(sig.values)
            row[f"{side}_deriv_max"] = deriv[valid].max()
            row[f"{side}_deriv_min"] = deriv[valid].min()

            # ---------------------------------------------------
            # Blink mask features
            # ---------------------------------------------------
            row[f"{side}_blink_fraction"] = mask.mean()
            row[f"{side}_blink_count"] = (mask.diff() == 1).sum()
            row[f"{side}_max_blink_len"] = longest_run(mask)
            row[f"{side}_mean_blink_len"] = mean_run(mask)

            # Derived blink features in seconds
            row[f"{side}_max_blink_dur_sec"] = longest_run(mask) / fs
            row[f"{side}_mean_blink_dur_sec"] = mean_run(mask) / fs

        # Add trial index so you can merge back later
        row["subject_session_trial"] = f"{df['Subject'].iloc[0]}_{df['Session'].iloc[0]}_{df['TrialId'].iloc[0]}"
        row["NumCorrect"] = df['NumCorrect'].iloc[0]
        feature_rows.append(row)

    # Return as a dataframe
    print(f"Extracted features from {len(feature_rows)} trials.")
    return pd.DataFrame(feature_rows)

def normalize_pupil_channels_by_name(df, pupil_col_names):
    ts = df.values.copy()
    for col_name in pupil_col_names:
        idx = df.columns.get_loc(col_name)
        mean = ts[:, idx].mean()
        std = ts[:, idx].std()
        ts[:, idx] = (ts[:, idx] - mean) / (std + 1e-8)
    return ts



class CNN1DHybrid(nn.Module):
    def __init__(self, seq_channels, hand_feat_dim, hidden_dim=64):
        super().__init__()

        # 1D CNN branch for time series
        self.cnn = nn.Sequential(
            nn.Conv1d(seq_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # You won't know final conv output size until runtime, so we infer it in forward()
        self.flatten = nn.Flatten()

        # Fully connected branch for hand-engineered features
        self.hand_net = nn.Sequential(
            nn.Linear(hand_feat_dim, hidden_dim),
            nn.ReLU(),
        )

        # Final fusion network (CNN + features → prediction)
        self.final = nn.Sequential(
            nn.Linear(hidden_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),     # predicts NumCorrect
        )

    def forward(self, X_seq, X_feat):
        # X_seq: [batch, seq_len, channels]
        X_seq = X_seq.permute(0, 2, 1)   # -> [batch, channels, seq_len]

        cnn_out = self.cnn(X_seq)
        cnn_out = cnn_out.mean(dim=2)    # simple global pooling → [batch, 64]

        feat_out = self.hand_net(X_feat)

        fused = torch.cat([cnn_out, feat_out], dim=1)

        out = self.final(fused)
        return out

def train_model(model, train_loader, val_loader, epochs=20):
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for X_seq_batch, X_feat_batch, y_batch in train_loader:
            optimizer.zero_grad()

            preds = model(X_seq_batch, X_feat_batch)

            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)

        train_loss /= len(train_loader.dataset)

        # ---- Validation ----
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_seq_batch, X_feat_batch, y_batch in val_loader:
                preds = model(X_seq_batch, X_feat_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * len(y_batch)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} Train RMSe: {np.sqrt(train_loss):.4f} | Val Loss: {val_loss:.4f} | Val RMSE: {np.sqrt(val_loss):.4f}")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--file_path', type=str, default='WM study/Eye data/*/*.gazedata', help='Path to the gaze data files')
    parsed_args = args.parse_args()

    filtered_all_trial_dataframes = call_all_filtered_trial_dataframes(parsed_args.file_path)
    filtered_all_trial_dataframes = interpolate_diameter_pupil(filtered_all_trial_dataframes)
    mask_dataframes = dilation_preprocessing(filtered_all_trial_dataframes)
    mask_dataframes = apply_low_pass_filter(mask_dataframes)
    mask_dataframes = confirm_no_nans(mask_dataframes)
    features_df = extract_pupil_features(mask_dataframes)
    # print(features_df.head())

    # Prepare features tensor 
    hand_feature_cols = [c for c in features_df.columns if c not in ["subject_session_trial", "NumCorrect"]] # Drop non-feature columns (subject_session_trial, NumCorrect)
    print(f"hand feature columns: {hand_feature_cols}")
    X_feat_all = features_df[hand_feature_cols].values
    y_all = features_df["NumCorrect"].values
    
    # ----------------------------
    # 2️⃣ Prepare X_seq (padded sequences)
    # ----------------------------
    pupil_cols = ["SmoothedDilationPupilLeftEye", "SmoothedDilationPupilRightEye"]
    mask_cols = ["LeftBlinkMask", "RightBlinkMask"]
    all_cols = pupil_cols + mask_cols


    trial_df_dict = {df["subject_session_trial"].iloc[0]: df for df in mask_dataframes}
    #print(f"trial_df: {trial_df_dict}")
    time_series_list = []

    for trial_id in features_df["subject_session_trial"]:
        df = trial_df_dict[trial_id]
        #print(f"Processing trial_id: {trial_id} with shape {df.shape}")

        # Only keep the pupil + mask columns, ensure numeric
        ts_values = df[all_cols].astype(np.float32).values  # [seq_len, n_channels]

        time_series_list.append(ts_values)

    # Pad sequences
    max_len = max(ts.shape[0] for ts in time_series_list)
    n_channels = time_series_list[0].shape[1]

    padded_sequences = []
    for ts in time_series_list:
        ts_numeric = ts.astype(np.float32)  # make sure numeric
        pad_len = max_len - ts_numeric.shape[0]
        if pad_len > 0:
            pad = np.zeros((pad_len, ts_numeric.shape[1]), dtype=np.float32)
            ts_padded = np.vstack([ts_numeric, pad])
        else:
            ts_padded = ts_numeric
        print(f"ts shape: {ts.shape}, padded shape: {ts_padded.shape}")
        padded_sequences.append(ts_padded)

    # Now stack
    X_seq = torch.tensor(np.stack(padded_sequences), dtype=torch.float32)


    # ----------------------------
    # 3️⃣ Split into train/val/test
    # ----------------------------
    X_feat_train, X_feat_temp, y_train, y_temp, X_seq_train, X_seq_temp = train_test_split(
        X_feat_all, y_all, X_seq, test_size=0.2, random_state=42
    )

    X_feat_val, X_feat_test, y_val, y_test, X_seq_val, X_seq_test = train_test_split(
        X_feat_temp, y_temp, X_seq_temp, test_size=0.5, random_state=42
    )

    # ----------------------------
    # 4️⃣ Normalize hand-engineered features (train stats)
    # ----------------------------
    scaler = StandardScaler()
    X_feat_train = torch.tensor(scaler.fit_transform(X_feat_train), dtype=torch.float32)
    X_feat_val = torch.tensor(scaler.transform(X_feat_val), dtype=torch.float32)
    X_feat_test = torch.tensor(scaler.transform(X_feat_test), dtype=torch.float32)

    # ----------------------------
    # 5️⃣ Normalize pupil channels (train stats only)
    # ----------------------------
    pupil_idx = [all_cols.index(c) for c in pupil_cols] # indices of pupil channels
    print(f"Pupil channel indices: {pupil_idx}")

    # Compute mean/std from training data
    mean_train = X_seq_train[:, :, pupil_idx].mean(dim=(0,1), keepdim=True)
    std_train = X_seq_train[:, :, pupil_idx].std(dim=(0,1), keepdim=True)

    # Normalize
    X_seq_train[:, :, pupil_idx] = (X_seq_train[:, :, pupil_idx] - mean_train) / std_train
    X_seq_val[:, :, pupil_idx]   = (X_seq_val[:, :, pupil_idx]   - mean_train) / std_train
    X_seq_test[:, :, pupil_idx]  = (X_seq_test[:, :, pupil_idx]  - mean_train) / std_train

    # ----------------------------
    # 6️⃣ Convert targets to tensors
    # ----------------------------
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # ----------------------------
    # 7️⃣ Create DataLoaders
    # ----------------------------
    train_dataset = TensorDataset(X_seq_train, X_feat_train, y_train)
    val_dataset = TensorDataset(X_seq_val, X_feat_val, y_val)
    test_dataset = TensorDataset(X_seq_test, X_feat_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"X_seq_train shape: {X_seq_train.shape}, X_feat_train shape: {X_feat_train.shape}, y_train shape: {y_train.shape}")

    seq_channels = X_seq_train.shape[2]
    hand_feat_dim = X_feat_train.shape[1]

    model = CNN1DHybrid(seq_channels, hand_feat_dim)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_model(model, train_loader, val_loader, epochs=30)

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X_seq_batch, X_feat_batch, y_batch in test_loader:
            preds = model(X_seq_batch, X_feat_batch)
            print(preds)
            loss = criterion(preds, y_batch)
            test_loss += loss.item() * len(y_batch)

    test_loss /= len(test_loader.dataset)
    print("Test MSE:", test_loss, "Test RMSE:", np.sqrt(test_loss))


