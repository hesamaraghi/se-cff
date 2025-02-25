#!/usr/bin/env python3
"""
Script to convert NSEK event sequences to DSEC format with multiprocessing support.

This script processes event sequences from raw HDF5 files, checks for monotonicity in timestamps,
and saves the processed data in a structured DSEC format. It leverages multiprocessing to handle
multiple sequences in parallel, enhancing performance on multi-core systems.

Usage:
    python convert_nsek_to_dsec.py --kitchen-name <kitchen_name> [options]

Options:
    --kitchen-name   Name of the kitchen to convert (required).
    --raw-folder     Path to the Raw folder containing event data (default: data/Raw_events).
    --train          Flag to indicate that sequences should be saved in training folders.
    --save-to        Path to save the converted data (default: data/NSEK).
    --chunk-size     Number of events to process per chunk (default: 100000).
    --log-level      Logging level (default: INFO).
    --log-file       Path to the log file (optional).
"""
import argparse
import logging
import multiprocessing
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
from cv2 import log
import h5py
import numpy as np
from tqdm import tqdm  
from baselines.depth_estimation.rectify_event_depth import Pixel_Projector

# ----------------------------- Configuration -----------------------------

# Define the event locations and their corresponding event names
LOCATIONS = {
    'left': "EVENT0",
    'right': "EVENT1"
    # Add other locations and event names as needed
}

# Default chunk size for processing events
DEFAULT_CHUNK_SIZE = 10_000_000

HEIGHT = 720
WIDTH = 1280

# ----------------------------- Utility Functions -----------------------------

def check_monotonicity(arr: np.ndarray) -> Tuple[bool, np.ndarray]:
    """
    Check if the array is monotonically increasing.

    Parameters:
        arr (np.ndarray): Array to check.

    Returns:
        Tuple[bool, np.ndarray]: A tuple containing a boolean indicating if the array is non-monotonic,
                                  and an array of indices where the monotonicity is violated.
    """
    if arr.size < 2:
        return False, np.array([], dtype='uint32')
    arr = np.float64(arr)
    diff = np.diff(arr)
    non_mono = np.any(diff < 0)
    idx = np.where(diff < 0)[0] + 1  # +1 to get the index of the problematic element
    return non_mono, idx

# ----------------------------- Core Functions -----------------------------

def convert_event_sequences(raw_folder: Path, save_to: Path, chunk_size: int, location_mapping: dict):
    """
    Convert left and right event sequences to DSEC format.

    Parameters:
        raw_folder (Path): Path to the raw event folder.
        save_to (Path): Path to save the converted DSEC data.
        chunk_size (int): Number of events to process per chunk.
        location_mapping (dict): Mapping of location keys to event names.
    """
    for location, event_name in location_mapping.items():
        raw_event_file = raw_folder / event_name / f"{event_name}.hdf5"
        save_to_h5_folder = save_to / 'events' / location
        save_to_h5_folder.mkdir(parents=True, exist_ok=True)

        if not raw_event_file.exists():
            logging.error(f"File {raw_event_file} does not exist. Skipping location '{location}'.")
            continue

        logging.info(f"Converting '{location}' events from {raw_event_file.parent} ...")

        key_list = ['x', 'y', 'p', 't']
        save_to_h5_file = save_to_h5_folder / 'events.h5'

        with h5py.File(raw_event_file, 'r') as h5f:
            missing_keys = [key for key in key_list if key not in h5f]
            if missing_keys:
                logging.error(f"Keys {missing_keys} not found in {raw_event_file}. Skipping.")
                continue

            lengths = [h5f[key].shape[0] for key in key_list]
            if len(set(lengths)) != 1:
                logging.error(f"Lengths of x, y, p, t do not match in {raw_event_file}. Skipping.")
                continue
            num_events = lengths[0]

            with h5py.File(save_to_h5_file, 'w') as h5f_save:
                # Create datasets with initial shape and maxshape for unlimited growth
                dset_x = h5f_save.create_dataset('/events/x', shape=(num_events,), maxshape=(None,), dtype='uint16')
                dset_y = h5f_save.create_dataset('/events/y', shape=(num_events,), maxshape=(None,), dtype='uint16')
                dset_t = h5f_save.create_dataset('/events/t', shape=(num_events,), maxshape=(None,), dtype='uint32')
                dset_p = h5f_save.create_dataset('/events/p', shape=(num_events,), maxshape=(None,), dtype='uint8')

                # Handle timestamp offset
                t0_us = np.int64(h5f['t'][0] * 1e6)
                # logging.info(f"First event timestamp: {t0_sec} s, {t0_us} us")
                h5f_save.create_dataset('/t_offset', data=t0_us, dtype='int64')

                # Initialize datasets for monotonicity checks
                dset_non_mono = h5f_save.create_dataset('/non_mono', data=0, dtype='uint8')
                dset_non_mono_idx = h5f_save.create_dataset('/non_mono_idx', shape=(0,), maxshape=(None,), dtype='uint32')

                num_chunks = (num_events + chunk_size - 1) // chunk_size  # Ceiling division
                non_mono_idx_list = []

                # Optional: Progress bar
                for i in tqdm(range(num_chunks), desc=f"Processing {location} events", unit="chunk"):
                    start_idx = i * chunk_size
                    end_idx = min(start_idx + chunk_size, num_events)

                    # Read slices
                    x_chunk = h5f['x'][start_idx:end_idx]
                    y_chunk = h5f['y'][start_idx:end_idx]
                    t_chunk = h5f['t'][start_idx:end_idx]
                    p_chunk = h5f['p'][start_idx:end_idx]

                    # Convert timestamps
                    t_converted = np.uint32(np.int64(t_chunk * 1e6) - t0_us)

                    # Write to datasets
                    dset_x[start_idx:end_idx] = x_chunk
                    dset_y[start_idx:end_idx] = y_chunk
                    dset_t[start_idx:end_idx] = t_converted
                    dset_p[start_idx:end_idx] = p_chunk

                    # Check monotonicity
                    non_mono, non_mono_idx = check_monotonicity(t_converted)
                    if non_mono:
                        dset_non_mono[...] = 1
                        # Adjust indices to global
                        non_mono_idx_global = non_mono_idx + start_idx
                        non_mono_idx_list.extend(non_mono_idx_global.tolist())

                # Update non_mono_idx dataset if any non-monotonicity found
                if non_mono_idx_list:
                    non_mono_idx_array = np.array(non_mono_idx_list, dtype='uint32')
                    dset_non_mono_idx.resize((len(non_mono_idx_array),))
                    dset_non_mono_idx[:] = non_mono_idx_array
                    logging.warning(f"Non-monotonic timestamps found in {raw_event_file} events.")
                    logging.warning(f"Indices: {non_mono_idx_array}")

                dset_ms_to_idx = h5f_save.create_dataset('/ms_to_idx', shape=((dset_t[-1]// 1000) + 2,), maxshape=(None,), dtype='uint64')
                ms_to_idx_len = dset_ms_to_idx.shape[0]
                dset_ms_to_idx[...] = np.searchsorted(dset_t[:], np.arange(ms_to_idx_len)*1000)

                logging.info(f"Finished processing '{location}' events. Saved to {save_to_h5_folder}.")

def convert_depth_map(
        annotations_folder: Path, 
        calibration_folder: Path, 
        save_to: Path,
        ):
    """
    Convert depth maps to DSEC format.
    
    Parameters:
        annotations_folder (Path): Path to the annotations folder.
        calibration_folder (Path): Path to the calibration folder.
        save_to (Path): Path to save the converted data.
        location_mapping (dict): Mapping of location keys to event names.
    """
    annotations_folder = annotations_folder / 'DEPTH'
    save_to_depth_folder = save_to / 'disparity' / 'event'
    timestamps_file = save_to / 'disparity' / 'timestamps.txt'
    
    save_to_depth_folder.mkdir(parents=True, exist_ok=True)
    timestamps_file.parent.mkdir(parents=True, exist_ok=True)

    projector = Pixel_Projector(
        calibration_path=calibration_folder,
        data_path=annotations_folder,
        )
    
    timestamps = []
    for file in annotations_folder.glob('*.tif'):
        try:
            ts = np.float64(file.stem)
            timestamps.append(ts)
        except ValueError:
            logging.error(f"Failed to convert timestamp from file {file}. Skipping.")
            continue
    if len(timestamps ) == 0 and len(annotations_folder.glob('*.tif')) == 0:
        logging.error(f"There is no depth files in {str(annotations_folder)}")
        return
        
    # Sort the timestamps
    timestamps = np.sort(np.array(timestamps, dtype=np.float64))
    
    with open(timestamps_file, 'w') as f:
        i = 0
        for ts in tqdm(timestamps):
            try:
                rectified_projected_depth = projector.project_and_rectify(ts)
                rectified_projected_depth[rectified_projected_depth < 200 ] = 0 # depth < 200 is invalid
                rectified_projected_depth[rectified_projected_depth > 1500] = 0 # depth > 1500 is invalid
                rectified_projected_depth = rectified_projected_depth.astype(np.uint16)
                cv2.imwrite(str(save_to_depth_folder / f"{i:06d}.png"), rectified_projected_depth)
                ts_int = np.uint64(ts * 1e6)
                f.write(f"{ts_int}\n")
                i += 2 # !!! Why??
            except Exception as e:
                logging.error(f"Failed to process timestamp {ts}: {e}")
                continue
                
    logging.info(f"Saved depth maps to {save_to_depth_folder}.")

def create_rectify_map(
        calibration_folder: Path, 
        save_to: Path,
        location_mapping: dict,
        ):
    """
    Create rectification maps for each camera pair.
    
    Parameters:
        calibration_folder (Path): Path to the calibration folder.
        save_to (Path): Path to save the rectification maps.
        location_mapping (dict): Mapping of location keys to event names.
    """
    """
    Read intrinsic matrix
    """
    #### Read EVENT0 intrinsic matrix
    EVENT0_intrinsic_matrix, EVENT0_distortion_matrix = read_intrinsic_calibration(calibration_folder, 'EVENT0')

    #### Read EVENT1 intrinsic matrix
    EVENT1_intrinsic_matrix, EVENT1_distortion_matrix = read_intrinsic_calibration(calibration_folder, 'EVENT1')
            
    """
    Read extrinsic matrix
    """
    #### Read EVENT0 -- EVENT1 extrinsic matrix
    rotation_matrix_EVENT0_EVENT1, translation_matrix_EVENT0_EVENT1 = read_extrinsic_calibration(calibration_folder, camA='EVENT0', camB='EVENT1')

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
                                                        cameraMatrix1 = EVENT0_intrinsic_matrix,
                                                        distCoeffs1 = EVENT0_distortion_matrix,
                                                        cameraMatrix2 = EVENT1_intrinsic_matrix,
                                                        distCoeffs2 = EVENT1_distortion_matrix,
                                                        imageSize = (WIDTH, HEIGHT),
                                                        R = rotation_matrix_EVENT0_EVENT1,
                                                        T = translation_matrix_EVENT0_EVENT1,
                                                        newImageSize = (0, 0),
                                                        alpha= 1,
                                                        )
    intrinsic_params = {
        "EVENT0": {
            "K" : EVENT0_intrinsic_matrix,
            "dist" : EVENT0_distortion_matrix,
            },
        "EVENT1": {
            "K" : EVENT1_intrinsic_matrix,
            "dist" : EVENT1_distortion_matrix,
            },
        }
    
    extrinsic_params = {
        "EVENT0": {
            "R" : R1,
            "P" : P1,
            },
        "EVENT1": {
            "R" : R2,
            "P" : P2,
            },
        }
        
    for location, event_name in location_mapping.items():
        
        save_to_rectify_file = save_to / 'events' / location / 'rectify_map.h5'
        save_to_rectify_file.parent.mkdir(parents=True, exist_ok=True)
        
        K_r = extrinsic_params[event_name]["P"][:3,:3]
        K = intrinsic_params[event_name]["K"]
        dist_coeffs = intrinsic_params[event_name]["dist"]
        R_r_0 = extrinsic_params[event_name]["R"]
        coords = np.stack(np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))).reshape((2, -1)).astype("float32")
        term_criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 0.001)
        points = cv2.undistortPointsIter(coords, K, dist_coeffs, R_r_0, K_r, criteria=term_criteria)
        rectify_map = points.reshape((HEIGHT, WIDTH, 2))
        
        with h5py.File(save_to_rectify_file, "w") as f:
            f.create_dataset('rectify_map', data=rectify_map, dtype=np.float32)
        
        logging.info(f"Saved rectification map for {location}.")

def process_sequence_event(args: Tuple[Path, Path, Path, Path, bool, Path, dict, int]):
    """
    Function to process a single sequence. Designed for multiprocessing.

    Parameters:
        args (Tuple): Tuple containing (seq, kitchen_folder, train, save_to, LOCATIONS, chunk_size)
    """
    seq, kitchen_folder, annotations_folder, calibration_folder, train, save_to, location_mapping, chunk_size = args
    try:
        logging.info(f"Processing sequence events: {seq} in {kitchen_folder} ...")
        seq_name = f"{kitchen_folder.name}_{seq}"
        dataset_type = "train" if train else "test"
        seq_folder = save_to / dataset_type / seq_name
        annotations_seq_folder = annotations_folder / seq
        seq_folder.mkdir(parents=True, exist_ok=True)
        convert_event_sequences(
                raw_folder=kitchen_folder / seq, 
                save_to=seq_folder, 
                chunk_size=chunk_size, 
                location_mapping=location_mapping,
                )
        create_rectify_map(
                calibration_folder=calibration_folder, 
                save_to=seq_folder,
                location_mapping=location_mapping,
                )
    except Exception as e:
        logging.error(f"Failed to process sequence {seq}: {e}")
        exit(1)

def process_sequence_depth(args: Tuple[Path, Path, Path, Path, bool, Path, dict, int]):
    """
    Function to process a single sequence. Designed for multiprocessing.

    Parameters:
        args (Tuple): Tuple containing (seq, kitchen_folder, train, save_to, LOCATIONS, chunk_size)
    """
    seq, kitchen_folder, annotations_folder, calibration_folder, train, save_to, location_mapping, chunk_size = args
    try:
        logging.info(f"Processing sequence depth: {seq} in {kitchen_folder} ...")
        seq_name = f"{kitchen_folder.name}_{seq}"
        dataset_type = "train" if train else "test"
        seq_folder = save_to / dataset_type / seq_name
        annotations_seq_folder = annotations_folder / seq
        seq_folder.mkdir(parents=True, exist_ok=True)
        convert_depth_map(
                annotations_folder=annotations_seq_folder, 
                calibration_folder=calibration_folder,
                save_to=seq_folder,
                )
    except Exception as e:
        logging.error(f"Failed to process depth for sequence {seq}: {e}")
        exit(1)

def read_intrinsic_calibration(calibration_path, cam):
    intrinsic_path = os.path.join(calibration_path,'intrinsic_calibration_results')
    _intrinsic_file = os.path.join(intrinsic_path, f'{cam}_intrinsic.xml')
    assert os.path.isfile(_intrinsic_file), f'{_intrinsic_file} does not exist!'
    _intrinsic_loader = cv2.FileStorage(_intrinsic_file, cv2.FileStorage_READ)
    _intrinsic_matrix = _intrinsic_loader.getNode('Intrinsic_Matrix').mat()
    _distortion_matrix = _intrinsic_loader.getNode('Distortion_Matrix').mat()

    return _intrinsic_matrix, _distortion_matrix

def read_extrinsic_calibration(calibration_path, camA, camB):
    extrinsic_path = os.path.join(calibration_path, 'stereo_calibration_results')
    _extrinsic_file = os.path.join(extrinsic_path, f'{camA}_{camB}_stereo_calibration.xml')
    assert os.path.isfile(_extrinsic_file), f'{_extrinsic_file} does not exist'
    _extrinsic_loader = cv2.FileStorage(_extrinsic_file, cv2.FileStorage_READ)
    _rotation_matrix = _extrinsic_loader.getNode('Rotation_Matrix').mat()
    _translation_matrix = _extrinsic_loader.getNode('Translation_Matrix').mat()

    return _rotation_matrix, _translation_matrix

def convert_NSEK2DSEC(
        kitchen_folder: Path, 
        train: bool, 
        save_to: Path, 
        chunk_size: int,
        annotations_folder: Path,
        calibration_folder: Path,
        ):
    """
    Convert all event sequences in the kitchen folder to DSEC format using multiprocessing.

    Parameters:
        kitchen_folder (Path): Path to the kitchen folder containing sequences.
        train (bool): Flag indicating whether to save in training or testing set.
        save_to (Path): Path to save the converted data.
        chunk_size (int): Number of events to process per chunk.
        annotations_folder (Path): Path to the annotations folder.
        calibration_folder (Path): Path to the calibration folder.
    """
    if not kitchen_folder.exists() or not kitchen_folder.is_dir():
        logging.error(f"Kitchen folder {kitchen_folder} does not exist or is not a directory.")
        raise NotADirectoryError(f"Kitchen folder {kitchen_folder} does not exist or is not a directory.")
    if not annotations_folder.exists() or not annotations_folder.is_dir():
        logging.error(f"Annotations folder {annotations_folder} does not exist or is not a directory.")
        raise NotADirectoryError(f"Annotations folder {annotations_folder} does not exist or is not a directory.")
    if not calibration_folder.exists() or not calibration_folder.is_dir():
        logging.error(f"Calibration folder {calibration_folder} does not exist or is not a directory.")
        raise NotADirectoryError(f"Calibration folder {calibration_folder} does not exist or is not a directory.")
    sequences = [seq.stem for seq in kitchen_folder.iterdir() if seq.is_dir()]
    if not sequences:
        logging.warning(f"No sequences found in {kitchen_folder}.")
        return
    annotation_sequences = [seq.stem for seq in annotations_folder.iterdir() if seq.is_dir()]
    if not annotation_sequences:
        logging.warning(f"No annotation sequences found in {annotations_folder}.")
        return
    if not all(seq in sequences for seq in annotation_sequences):
        logging.error(f"All annotation sequences for {kitchen_folder} do not exist in event sequences.")
        raise FileNotFoundError

    logging.info(f"Found {len(sequences)} sequences in {kitchen_folder}.")
    logging.info(f"Found {len(annotation_sequences)} annotation sequences in {annotations_folder}.")
    logging.info(f"Considering only the sequences present in both folders.")
    sequences = annotation_sequences

    # Prepare arguments for multiprocessing
    args_list = [
        (seq, kitchen_folder, annotations_folder, calibration_folder, train, save_to, LOCATIONS, chunk_size)
        for seq in sequences
    ]

    # Determine the number of processes to use
    num_processes = min(multiprocessing.cpu_count(), len(2 * sequences))
    if sys.gettrace():
        num_processes = 1
        logging.warning("Debugger is active. Running with a single process.")
        
    logging.info(f"Starting multiprocessing with {num_processes} processes.")
    # Initialize multiprocessing pool
        
    with multiprocessing.Pool(processes=num_processes) as pool2:
        results_depth = pool2.imap(process_sequence_depth, args_list)
        
        # Using tqdm for progress bars (you can also iterate concurrently if desired)
        list(tqdm(results_depth, total=len(args_list), desc="Processing sequences: depth"))

    logging.info("Depth has been processed.")

# ----------------------------- Main Function -----------------------------

def main():
    """
    Main function to parse arguments and initiate the conversion process.
    """
    parser = argparse.ArgumentParser(description="Convert NSEK to DSEC format.")
    parser.add_argument("--kitchen-name", type=str, required=True, help="Kitchen name to convert.")
    parser.add_argument("--raw-folder", type=str, default="data/Raw_events", help="Path to the Raw folder.")
    parser.add_argument("--calibration-folder", type=str, default="data/Calibration/calibration_results/calibration_with_435")
    parser.add_argument("--annotations-folder", type=str, default="data/annotations")
    parser.add_argument("--train", action='store_true', help="Flag to indicate training data.")
    parser.add_argument("--save-to", type=str, default="data/NSEK", help="Folder to save the converted data.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Number of events to process per chunk.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    parser.add_argument("--log-file", type=str, default=None, help="Path to the log file. If not set, logs will not be saved to a file.")

    args = parser.parse_args()

    # Configure logging
    log_handlers = []
    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    log_handlers.append(console_handler)

    # File handler (if log_file is provided)
    if args.log_file:
        log_file = Path(args.log_file)
    else:
        log_file = Path(args.kitchen_name + ".log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(log_format))
    log_handlers.append(file_handler)

    # Set the logging level
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"Invalid log level: {args.log_level}. Defaulting to INFO.")
        numeric_level = logging.INFO

    logging.basicConfig(level=numeric_level, handlers=log_handlers)

    # Resolve paths using pathlib
    raw_folder = Path(args.raw_folder)
    kitchen_folder = raw_folder / args.kitchen_name
    save_to = Path(args.save_to)
    annotations_folder = Path(args.annotations_folder)
    annotations_folder = annotations_folder / args.kitchen_name
    calibration_folder = Path(args.calibration_folder) 

    # Validate kitchen_folder
    if not kitchen_folder.exists() or not kitchen_folder.is_dir():
        logging.error(f"Kitchen folder {kitchen_folder} does not exist or is not a directory.")
        exit(1)
    
    logging.info(f"Converting kitchen '{args.kitchen_name}' ...")
    logging.info(f"Raw folder: {raw_folder}")
    logging.info(f"Annotations folder: {annotations_folder}")
    logging.info(f"Calibration folder: {calibration_folder}")
    logging.info(f"Save to: {save_to}")
    logging.info(f"Chunk size: {args.chunk_size}")
    logging.info(f"Training data: {args.train}")

    # Start conversion
    try:
        convert_NSEK2DSEC(
            kitchen_folder=kitchen_folder,
            train=args.train,
            save_to=save_to,
            chunk_size=args.chunk_size,
            annotations_folder=annotations_folder,
            calibration_folder=calibration_folder,
        )
    except Exception as e:
        logging.critical(f"Event conversion failed: {e}")
        exit(1) 

# ----------------------------- Unit Tests -----------------------------

import unittest

class TestMonotonicity(unittest.TestCase):
    def test_monotonic_true(self):
        arr = np.array([1, 2, 3, 4, 5], dtype='uint32')
        non_mono, idx = check_monotonicity(arr)
        self.assertFalse(non_mono)
        self.assertEqual(len(idx), 0)

    def test_monotonic_false_single(self):
        arr = np.array([1, 3, 2, 4, 5], dtype='uint32')
        non_mono, idx = check_monotonicity(arr)
        self.assertTrue(non_mono)
        np.testing.assert_array_equal(idx, np.array([2]))

    def test_monotonic_false_multiple(self):
        arr = np.array([1, 3, 2, 4, 3, 5, 7], dtype='uint32')
        non_mono, idx = check_monotonicity(arr)
        self.assertTrue(non_mono)
        np.testing.assert_array_equal(idx, np.array([2, 4]))

    def test_monotonic_empty(self):
        arr = np.array([], dtype='uint32')
        non_mono, idx = check_monotonicity(arr)
        self.assertFalse(non_mono)
        self.assertEqual(len(idx), 0)

    def test_monotonic_single_element(self):
        arr = np.array([1], dtype='uint32')
        non_mono, idx = check_monotonicity(arr)
        self.assertFalse(non_mono)
        self.assertEqual(len(idx), 0)

# ----------------------------- Entry Point -----------------------------

if __name__ == "__main__":
    # To run unit tests, execute: python convert_nsek_to_dsec.py --test
    import sys

    if "--test" in sys.argv:
        sys.argv.remove("--test")
        unittest.main()
    else:
        main()
