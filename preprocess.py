import pandas as pd
import numpy as np
import glob
from scipy.signal import butter, filtfilt
import random
import matplotlib.pyplot as plt

def call_all_filtered_trial_dataframes():
    all_trial_dataframes = []
    count = 0
    
    for filename in glob.glob(r'C:\Users\emory\Documents\git_projects\Research\WM study\WM study\Eye data\*\*.gazedata'):
        raw_dataframe = pd.read_csv(filename, sep='\t', low_memory=False)
        print(f"Processing file: {filename}")
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
    # NOTE: fixed this. You cannot pop inside the loop because that messes with the indexing.

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
    # interpolate DiameterPupilLeftEye and DiameterPupilRightEye using MNE
    MAX=6
    MIN=1
    num_linear_fallback = 0
    for index, trial in enumerate(filtered_all_trial_dataframes):
    # also cutting bad data (e.g. eye partially closed)
        trial.loc[:, "DiameterPupilRightEye"] = trial["DiameterPupilRightEye"].apply(
            lambda x: np.nan if x < MIN or x > MAX else x)
        trial.loc[:, "DiameterPupilLeftEye"] = trial["DiameterPupilLeftEye"].apply(
            lambda x: np.nan if x < MIN or x > MAX else x)

        # TODO: need to do something about false values (blink making the percieved pupil diameter look tiny then interpolation making it blow up)
        # dumb way would be to also make values around blinks NanN too
        try:
            filtered_all_trial_dataframes[index].loc[:, 'DiameterPupilLeftEye'] = trial['DiameterPupilLeftEye'].interpolate(method='pchip')
            filtered_all_trial_dataframes[index].loc[:, 'DiameterPupilRightEye'] = trial['DiameterPupilRightEye'].interpolate(method='pchip')

            #filtered_all_trial_dataframes[index] = trial.interpolate(method='pchip')
        except:
            num_linear_fallback+=1
            filtered_all_trial_dataframes[index].loc[:, 'DiameterPupilLeftEye'] = trial['DiameterPupilLeftEye'].interpolate()
            filtered_all_trial_dataframes[index].loc[:, 'DiameterPupilRightEye'] = trial['DiameterPupilRightEye'].interpolate()

        # print(f"failed interpolation and fell back to linear {num_linear_fallback} times")

    return filtered_all_trial_dataframes


def dilation_preprocessing(filtered_all_trial_dataframes):
    
    def avg_block_dataframes(trial_df_mask):

        trial_df_mask['time_bin'] = (trial_df_mask['ElapsedTime'] // 100).astype(int)

        time_bin_df = trial_df_mask.groupby('time_bin')[['DiameterPupilLeftEye', 'DiameterPupilRightEye']].mean().reset_index()

        time_bin_df['time_ms'] = time_bin_df['time_bin'] * 100

        adjusted_time_bin_df = time_bin_df.copy()
        adjusted_time_bin_df['time_bin'] = adjusted_time_bin_df['time_ms'] - adjusted_time_bin_df['time_ms'][0]
        #adjusted_time_bin_df = adjusted_time_bin_df.drop(columns=['time_ms'])
        adjusted_time_bin_df["ElapsedTime"] = adjusted_time_bin_df['time_bin']

        adjusted_time_bin_df['Subject'] = trial_df_mask['Subject'].iloc[0]
        adjusted_time_bin_df['Session'] = trial_df_mask['Session'].iloc[0]
        adjusted_time_bin_df['TrialId'] = trial_df_mask['TrialId'].iloc[0]
        adjusted_time_bin_df['CurrentObject'] = trial_df_mask['CurrentObject'].iloc[0]
        adjusted_time_bin_df['NumCorrect'] = trial_df_mask['NumCorrect'].iloc[0]

        return adjusted_time_bin_df
        
    mask_dataframes = []
    for index, trial_df in enumerate(filtered_all_trial_dataframes):
        fixation_df = trial_df[trial_df["CurrentObject"] == "Fixation"]

        baseline_left_pupil = fixation_df[fixation_df["DiameterPupilLeftEye"].notna()]["DiameterPupilLeftEye"].mean() # get mean of pupil diameter, but don't include diameter == -1
        baseline_right_pupil = fixation_df[fixation_df["DiameterPupilRightEye"].notna()]["DiameterPupilRightEye"].mean() # get mean of pupil diameter, but don't include diameter == -1
        if pd.isna(baseline_left_pupil) or pd.isna(baseline_right_pupil):
            print(f"Skipping trial {index} due to NaN baseline")
            continue
        
        trial_df = avg_block_dataframes(trial_df.loc[trial_df["CurrentObject"] == "Mask"].copy())

        if trial_df.shape[0] != 40:
            print(f"Skipping trial {index} due to not 40 bins")
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


# Low Pass Filter
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

def plot_trials(indices, mask_dataframes, columns_to_plot, plot_name): 
    for i in indices:
        trial_df = mask_dataframes[i]
        fig, ax1 = plt.subplots(figsize=(10, 6))

        for column in trial_df.columns:
            if column in columns_to_plot: # Exclude the label from plotting
                ax1.plot(trial_df['ElapsedTime'], trial_df[column], label=column)

        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Pupil Dilation (mm)')
        plt.title(f'Subject {trial_df['Subject'].iloc[0]}, Trial: {trial_df["TrialId"].iloc[0]} (NumCorrect: {trial_df["NumCorrect"].iloc[0]})')
        ax1.legend(loc='upper left')
        plt.savefig(f"{plot_name}_subject_{trial_df['Subject'].iloc[0]}_trial_{trial_df['TrialId'].iloc[0]}.png")

def plot_trial(subject, trial, mask_dataframes, columns_to_plot, plot_name): 
    for trial_df in mask_dataframes:
        if trial_df["Subject"].iloc[0] == subject and trial_df["TrialId"].iloc[0] == trial:
            fig, ax1 = plt.subplots(figsize=(10, 6))

            for column in trial_df.columns:
                if column in columns_to_plot: # Exclude the label from plotting
                    ax1.plot(trial_df['ElapsedTime'], trial_df[column], label=column)

            ax1.set_xlabel('Time (ms)')
            ax1.set_ylabel('Pupil Dilation (mm)')
            plt.title(f'Subject {trial_df['Subject'].iloc[0]}, Trial: {trial_df["TrialId"].iloc[0]} (NumCorrect: {trial_df["NumCorrect"].iloc[0]})')
            ax1.legend(loc='upper left')
            plt.savefig(f"{plot_name}_subject_{trial_df['Subject'].iloc[0]}_trial_{trial_df['TrialId'].iloc[0]}.png")

def preprocess_data():
    filtered_all_trial_dataframes = call_all_filtered_trial_dataframes()
    indices = [1]
    for i,trial in enumerate(filtered_all_trial_dataframes):
        if trial["Subject"].iloc[0] == 1005 and trial["TrialId"].iloc[0] == 17:
            print(f"Adding index {i} for subject: {trial['Subject'].iloc[0]}, trial: {trial['TrialId'].iloc[0]}")
            indices.extend([i])
            break
    print(f"Indices for plotting: {indices}")
    plot_trials(indices, filtered_all_trial_dataframes, ['DiameterPupilLeftEye', 'DiameterPupilRightEye'], 'raw_diameter_pupil')
    filtered_all_trial_dataframes = interpolate_diameter_pupil(filtered_all_trial_dataframes)
    for i,trial in enumerate(filtered_all_trial_dataframes):
         if trial["Subject"].iloc[0] == 1005 and trial["TrialId"].iloc[0] == 17:
            print(f"Adding index {i} for subject: {trial['Subject'].iloc[0]}, trial: {trial['TrialId'].iloc[0]}")
            indices.extend([i])
            break
    plot_trials(indices, filtered_all_trial_dataframes, ['DiameterPupilLeftEye', 'DiameterPupilRightEye'], 'interpolated_diameter_pupil')
    mask_dataframes = dilation_preprocessing(filtered_all_trial_dataframes)
    for i,trial in enumerate(filtered_all_trial_dataframes):
         if trial["Subject"].iloc[0] == 1005 and trial["TrialId"].iloc[0] == 17:
            print(f"Adding index {i} for subject: {trial['Subject'].iloc[0]}, trial: {trial['TrialId'].iloc[0]}")
            indices.extend([i])
            break
    plot_trials(indices, mask_dataframes, ['DilationPupilLeftEye', 'DilationPupilRightEye'], 'dilation_pupil')
    plot_trial(1005, 17, mask_dataframes, ['DilationPupilLeftEye', 'DilationPupilRightEye'], 'dilation_pupil_single_trial')
    # mask_dataframes = apply_low_pass_filter(mask_dataframes)
    return mask_dataframes

if __name__ == "__main__":
    mask_dataframes = preprocess_data()
    print(mask_dataframes[0])
