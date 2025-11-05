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

        # Normalize column names (trim whitespace)
        df.columns = [c.strip() for c in df.columns]

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    # Concatenate with outer join (union of columns). ignore_index to reindex rows.
    return pd.concat(dfs, ignore_index=True, sort=False)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Recursively list files in a directory matching the given extension.")
    parser.add_argument("directory", help="Directory to search (path)")
    parser.add_argument(
        "extension",
        help="File extension to match (with or without leading dot), e.g. .txt or txt",
    )
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

    # Always print the number of files found
    print(f"Num files: {len(files)}", file=sys.stderr)
    
    # Drop gaze location columns
    df.drop(["XGazePosLeftEye","YGazePosLeftEye","XCameraPosLeftEye","YCameraPosLeftEye","XGazePosRightEye","YGazePosRightEye","XCameraPosRightEye","YCameraPosRightEye","DistanceRightEye","DistanceLeftEye"], axis=1, inplace=True)

    # For convenience, also print the DataFrame's first few rows to stdout
    try:
        # Avoid printing huge data — print first 10 rows as tab-separated
        if df.shape[0] > 0:
            print(f"{df.head(10)}")
        print(f"Datafame columns: {', '.join(df.columns)}")
        print(f"Number of Subject-Sessions:")
        num_subject_sessions = len(df.groupby(['Subject', 'Session']))
        print(num_subject_sessions)
        print(f"Trials per Subject-Session:")
        trials_per_subject_session = df.groupby(['Subject', 'Session'])['TrialId'].nunique()
        print(trials_per_subject_session)
        total_trials = trials_per_subject_session.sum()
        print(f"Total trials: {total_trials}")
    except Exception:
        # If printing fails, ignore — we already loaded the data
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
