"""Recursively find files by extension, validate headers, and load into one DataFrame.

Usage examples:
    python scripts/load_data.py ./data .txt
    python scripts/load_data.py C:\\Data txt

The script accepts a directory (positional) and an extension (positional).
The extension may be given with or without a leading dot. Files must have the
same first-line header (exact match, line endings normalized); otherwise the
script errors out.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt



def find_files(directory, extension):
    """Return a sorted list of Path objects under `directory` matching `extension`.

    directory can be a string or a Path. extension may include a leading dot or not.
    """
    ext = extension if extension.startswith(".") else "." + extension
    p = Path(directory)
    if not p.exists():
        raise FileNotFoundError("Directory not found: %s" % directory)
    if not p.is_dir():
        raise NotADirectoryError("Not a directory: %s" % directory)

    matches = list(p.rglob("*" + ext))
    return sorted(matches)


def load_tabular_files_into_dataframe(file_paths):
    """Load tab-separated files into one DataFrame.

    If files have different headers, the returned DataFrame contains the union
    of all columns. Missing values in files that lacked a column are left as
    NaN.

    Args:
        file_paths: iterable of file paths (strings or Path objects).

    Returns:
        A pandas.DataFrame containing the concatenated data.

    Raises:
        ImportError: if pandas is not installed.
        ValueError: if no files provided.
    """
    paths = [Path(p) for p in file_paths]
    if not paths:
        raise ValueError("No files provided to load")

    dfs = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError("File not found: %s" % p)

        # Read the file as a tab-separated table
        df = pd.read_csv(p, sep="\t", header=0, low_memory=False)

        # Add filename column
        df["filename"] = p.name

        # Normalize column names (trim whitespace)
        df.columns = [c.strip() for c in df.columns]

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    # Concatenate with outer join (union of columns). ignore_index to reindex rows.
    return pd.concat(dfs, ignore_index=True, sort=False)


def plot_trial(df, subject, session, trial, outpath=None, show=True):
    """Plot pupil diameters for a given subject/session/trial.

    Parameters
    - df: pandas.DataFrame containing the combined data
    - subject, session, trial: values to match (compared as strings)
    - outpath: optional path to save the figure (png/pdf). If None, the figure is shown.
    - show: if True, call plt.show() after plotting.

    The function plots TETTime on x-axis and both pupil diameters on the right y-axis.
    Vertical dashed lines are drawn when `CurrentObject` changes.
    """
    # Use exact column names as provided
    subj_col = "Subject"
    sess_col = "Session"
    trial_col = "TrialId"
    time_col = "TETTime"
    right_col = "DiameterPupilRightEye"
    left_col = "DiameterPupilLeftEye"
    obj_col = "CurrentObject"

    # Filter rows: compare as strings for robustness
    mask = (
        df[subj_col].astype(str) == str(subject)
    ) & (
        df[sess_col].astype(str) == str(session)
    ) & (
        df[trial_col].astype(str) == str(trial)
    )
    dft = df[mask].copy()
    if dft.empty:
        raise ValueError(f"No rows found for subject={subject}, session={session}, trial={trial}")

    # Ensure time sorting and shift so first time is zero
    try:
        dft[time_col] = pd.to_numeric(dft[time_col], errors="coerce")
    except Exception:
        pass
    dft = dft.sort_values(by=time_col)

    # Compute change points (times and object values) before shifting time
    if obj_col in dft.columns:
        cur = dft[obj_col].astype(str).fillna("")
        changes = cur != cur.shift(1)
        change_times = list(dft.loc[changes, time_col])
        change_objects = list(dft.loc[changes, obj_col])
    else:
        change_times = []
        change_objects = []

    # shift time so first timestamp is zero
    if dft[time_col].notna().any():
        t0 = dft[time_col].iloc[0]
        dft[time_col] = dft[time_col] - t0
    else:
        t0 = 0

    x = dft[time_col]

    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot pupil diameters on the left axis (requested)
    # Apply validity masks: only plot when Validity==0 (valid)
    if right_col in dft.columns:
        y_right = pd.to_numeric(dft[right_col], errors="coerce")
        if "ValidityRightEye" in dft.columns:
            valid_mask_r = dft["ValidityRightEye"] == 0
            y_right = y_right.where(valid_mask_r, other=pd.NA)
        ax.plot(x, y_right, label="Right pupil", color="tab:blue")
    if left_col in dft.columns:
        y_left = pd.to_numeric(dft[left_col], errors="coerce")
        if "ValidityLeftEye" in dft.columns:
            valid_mask_l = dft["ValidityLeftEye"] == 0
            y_left = y_left.where(valid_mask_l, other=pd.NA)
        ax.plot(x, y_left, label="Left pupil", color="tab:orange")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Diameter (mm)")

    # Vertical lines when CurrentObject changes (use shifted times)
    if change_times:
        # shift change times by t0
        change_times_shifted = [ct - t0 for ct in change_times]
        ylim = ax.get_ylim()
        y_text = ylim[1] - 0.02 * (ylim[1] - ylim[0])
        for xi, objval in zip(change_times_shifted, change_objects):
            ax.axvline(x=xi, color="k", linestyle="--", alpha=0.6)
            # annotate the object slightly below the top of the axis in data coords
            ax.text(xi, y_text, str(objval), rotation=90, va="top", ha="right", fontsize=8)

    # Legend on the left axis
    try:
        ax.relim()
        ax.autoscale_view()
    except Exception:
        pass
    ax.legend(loc="upper right")

    fig.tight_layout()
    if outpath:
        fig.savefig(outpath)
    if show:
        plt.show()
    plt.close(fig)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Recursively list files in a directory matching the given extension.")
    parser.add_argument("directory", help="Directory to search (path)")
    parser.add_argument(
        "extension",
        help="File extension to match (with or without leading dot), e.g. .txt or txt",
    )
    parser.add_argument("--output", "-o", help="Output CSV path for combined dataframe", default="combined.csv")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    try:
        files = find_files(args.directory, args.extension)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if not files:
        print(f"No files found.", file=sys.stderr)
        return 1

    # Validate headers and load into a single DataFrame
    try:
        df = load_tabular_files_into_dataframe(files)
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 3
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 4
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 5

    # Print summary info
    print(f"Loaded DataFrame: rows={df.shape[0]}, cols={df.shape[1]}", file=sys.stderr)
    for i, file in enumerate(files): 
        print(f"{i}.{file}", file=sys.stderr)
    # Always print the number of files found
    print(f"Num files: {len(files)}", file=sys.stderr)
    
    # Drop gaze location columns
    df.drop(["XGazePosLeftEye","YGazePosLeftEye","XCameraPosLeftEye","YCameraPosLeftEye","XGazePosRightEye","YGazePosRightEye","XCameraPosRightEye","YCameraPosRightEye","DistanceRightEye","DistanceLeftEye"], axis=1, inplace=True)

    # For convenience, also print the DataFrame's first few rows to stdout
    try:
        # Avoid printing huge data — print first 10 rows as tab-separated
        if df.shape[0] > 0:
            print(f"{df.head(10)}")
        print(f"Dataframe columns: {', '.join(df.columns)}")
        print(f"Number of Subject-Sessions:")

        # Use exact column names for grouping
        subj_col = "Subject"
        sess_col = "Session"
        trial_col = "TrialId"

        num_subject_sessions = len(df.groupby([subj_col, sess_col]))
        print(num_subject_sessions)
        print(f"Trials per Subject-Session:")
        if trial_col in df.columns:
            trials_per_subject_session = df.groupby([subj_col, sess_col])[trial_col].nunique()
            print(trials_per_subject_session)
            total_trials = trials_per_subject_session.sum()
            print(f"Total trials: {total_trials}")
        else:
            print(f"Trial column not found (tried {trial_col}). Skipping trial counts.")
    except Exception:
        # If printing fails, ignore — we already loaded the data
        pass

    plot_trial(df, subject=1040, session=1, trial=1, outpath='S1040_E1_T1.png', show=False)
    # Save combined dataframe to CSV
    try:
        outpath = args.output
        df.to_csv(outpath, index=False)
        print(f"Saved combined dataframe to: {outpath}", file=sys.stderr)
    except Exception as exc:
        print(f"Warning: failed to save combined CSV: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
