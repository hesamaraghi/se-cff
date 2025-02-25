import os
import csv
from pathlib import Path

def process_sequence(seq_path: Path):
    """
    Process a sequence folder to extract timestamps from disparity/timestamps.txt
    and save them in a CSV file with the format:
    
    # timestamp_us, file_index
    53193302164, 20
    53194301052, 40
    ...
    """
    timestamps_txt = seq_path / "disparity" / "timestamps.txt"

    if timestamps_txt.exists():
        # Read timestamps
        timestamps = []
        with timestamps_txt.open("r") as f:
            for line in f:
                line = line.strip()
                if line.isdigit():  # Ensure the line is a valid number
                    timestamps.append(int(line))

        # Sort the timestamps in ascending order.
        timestamps.sort()
        
        # Define output CSV file path
        csv_filename = seq_path.name + ".csv"  # e.g., interlaken_00_c.csv
        csv_path = seq_path / csv_filename

        # Write CSV file with the desired format
        with csv_path.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Write the header
            writer.writerow(["# timestamp_us", "file_index"])

            # Write data: file_index starts at 20 and increments by 20
            for idx, ts in enumerate(timestamps):
                if idx == 0:
                    continue
                if idx % 1 == 0:
                    writer.writerow([ts, (idx *2)])

        print(f"✅ Created CSV file: {csv_path}")
    else:
        print(f"⚠️ No timestamps.txt found in {seq_path}, skipping.")

def process_dsec_folder(dsec_root: Path):
    """
    Walks through 'train' and 'test' directories in the DSEC dataset and processes each sequence.
    """
    for subset in ["test"]:
        subset_path = dsec_root / subset
        if not subset_path.exists():
            continue

        for seq in subset_path.iterdir():
            if seq.is_dir():
                process_sequence(seq)

if __name__ == "__main__":
    # Set the path to the DSEC folder
    dsec_root = Path("data") / "NSEK"

    if dsec_root.exists():
        process_dsec_folder(dsec_root)
    else:
        print("❌ DSEC folder not found. Please check the path.")
