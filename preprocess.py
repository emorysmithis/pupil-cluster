import pandas as pd
import numpy as np
import glob
from scipy.signal import butter, filtfilt

def call_all_filtered_trial_dataframes():
    all_trial_dataframes = []
    count = 0
    
    for filename in glob.glob('WM study/Eye data/*/*.gazedata'):
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
        adjusted_time_bin_df = adjusted_time_bin_df.drop(columns=['time_ms'])

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

        mask_dataframes.append(trial_df)
        # if trial_df['NumCorrect'].min().item() == 0 or trial_df['NumCorrect'].min().item() == 6:
        #     mask_dataframes.append(trial_df)

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

<<<<<<< HEAD
def preprocess_data():
=======

def group_by_subject(dataframes):
    """
    Group dataframes by subject
    
    Args:
        dataframes: list of dataframes (each dataframe must contain 'Subject' column)
    
    Returns:
        dict: {subject: [dataframes for that subject]}
    """
    subject_groups = {}
    
    for df in dataframes:
        if 'Subject' not in df.columns:
            print(f"Warning: 'Subject' column not found in dataframe, skipping...")
            continue
        
        subject = df['Subject'].iloc[0].item()
        
        if subject not in subject_groups:
            subject_groups[subject] = []
        
        subject_groups[subject].append(df)
    
    print(f"Grouped data into {len(subject_groups)} subjects")
    for subject, dfs in subject_groups.items():
        print(f"  Subject {subject}: {len(dfs)} trials")
    
    return subject_groups

def preprocess_data(is_group_by_subject=False, return_original=False):
>>>>>>> d40b239 (feature engineering and anova analysis with report)
    filtered_all_trial_dataframes = call_all_filtered_trial_dataframes()
    filtered_all_trial_dataframes = interpolate_diameter_pupil(filtered_all_trial_dataframes)
    mask_dataframes = dilation_preprocessing(filtered_all_trial_dataframes)
    if is_group_by_subject:
        subject_groups = group_by_subject(mask_dataframes)
        return subject_groups
    # mask_dataframes = apply_low_pass_filter(mask_dataframes)
    if return_original:
        return mask_dataframes, filtered_all_trial_dataframes
    return mask_dataframes

if __name__ == "__main__":
<<<<<<< HEAD
    mask_dataframes = preprocess_data()
    print(mask_dataframes[0])
=======
    mask_dataframes = preprocess_data(is_group_by_subject=True)
    print(mask_dataframes[0])
>>>>>>> d40b239 (feature engineering and anova analysis with report)
