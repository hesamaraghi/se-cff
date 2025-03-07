import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

def compute_metrics(gt_image, pred_image):
    """Compute RMSE, MAE, and baseline STD for a given pair of images."""
    valid_mask = gt_image > 0  # Consider only nonzero depth values
    valid_gt = gt_image[valid_mask]
    valid_pred = pred_image[valid_mask]
    
    if valid_gt.size == 0:  # If no valid pixels, return NaN
        return float('nan'), float('nan'), float('nan')
    
    # Compute errors
    rmse = np.sqrt(np.mean((valid_pred - valid_gt) ** 2))
    mae = np.mean(np.abs(valid_pred - valid_gt))
    baseline_std = np.std(valid_gt)  # Standard deviation of ground truth depths
    baseline_mad = np.mean(np.abs(valid_gt - np.mean(valid_gt)))  # Mean absolute deviation of ground truth depths

    return rmse, mae, baseline_std, baseline_mad

def process_experiment(experiment_name, activities_root="data/NSEK/test"):
    """Processes all activities and computes depth estimation errors."""
    results = {}
    all_rmse, all_mae, all_std, all_mad = [], [], [], []
    
    # Get all activity folders
    activity_folders = glob(os.path.join(activities_root, "*"))
    
    for activity_folder in activity_folders:
        print("\n" + "=" * 50, flush=True)
        print(f"Processing activity: {activity_folder}", flush=True)
        print("-" * 50, flush=True)
        activity_name = os.path.basename(activity_folder)
        gt_path = os.path.join(activity_folder, "disparity/event")
        pred_path = os.path.join(experiment_name, "visualize/pred/test", activity_name)
        
        # Ensure the prediction folder exists
        if not os.path.exists(pred_path):
            continue
        
        activity_rmse, activity_mae, activity_std, activity_mad = [], [], [], []
        pred_files = glob(os.path.join(pred_path, "*.png"))
        
        for pred_file in tqdm(pred_files):
            filename = os.path.basename(pred_file)
            gt_file = os.path.join(gt_path, filename)
            
            if not os.path.exists(gt_file):
                print(f"Warning: Ground truth file {gt_file} not found for prediction {pred_file}")
                continue  # Skip missing predictions
            
            # Load images
            gt_image = cv2.imread(gt_file, -1).astype(np.float32)
            pred_image = cv2.imread(pred_file, -1).astype(np.float32)
            
            # Compute metrics
            rmse, mae, baseline_std, baseline_mad = compute_metrics(gt_image, pred_image)
            
            if not (np.isnan(rmse) or np.isnan(mae) or np.isnan(baseline_std) or np.isnan(baseline_mad)):  # Skip if any metric is NaN
                activity_rmse.append(rmse)
                activity_mae.append(mae)
                activity_std.append(baseline_std)
                activity_mad.append(baseline_mad)
            else:
                print(f"Warning: One or more metrics are NaN for file {pred_file}")
                print(f"  - RMSE: {rmse}, MAE: {mae}, Baseline_STD: {baseline_std}, Baseline_MAD: {baseline_mad}")

        # Store results for this activity
        if activity_rmse:
            results[activity_name] = {
                "RMSE": np.mean(activity_rmse),
                "MAE": np.mean(activity_mae),
                "Baseline_STD": np.mean(activity_std),
                "Baseline_MAD": np.mean(activity_mad),
            }
            all_rmse.extend(activity_rmse)
            all_mae.extend(activity_mae)
            all_std.extend(activity_std)
            all_mad.extend(activity_mad)

    # Compute overall metrics
    if all_rmse:
        results["Overall"] = {
            "RMSE": np.mean(all_rmse),
            "MAE": np.mean(all_mae),
            "Baseline_STD": np.mean(all_std),
            "Baseline_MAD": np.mean(all_mad),
        }

    # Save results
    save_results(experiment_name, results)

def save_results(experiment_name, results):
    """Saves the computed metrics in a well-structured results.txt file."""
    result_file = experiment_name + "_results.txt"

    with open(result_file, "w") as f:
        f.write("=" * 50 + "\n")
        f.write(f" Depth Estimation Results for Experiment: {experiment_name} \n")
        f.write("=" * 50 + "\n\n")
        
        for activity, metrics in results.items():
            f.write(f"Activity: {activity}\n")
            f.write(f"  - RMSE: {metrics['RMSE']:.4f}\n")
            f.write(f"  - MAE: {metrics['MAE']:.4f}\n")
            f.write(f"  - Baseline_STD: {metrics['Baseline_STD']:.4f}\n")
            f.write(f"  - Baseline_MAD: {metrics['Baseline_MAD']:.4f}\n")
            f.write("\n")
            f.write("-" * 50 + "\n")

    print(f"Results saved to {result_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute depth estimation error")
    parser.add_argument("--experiment_name", type=str, help="Name of the experiment folder")
    args = parser.parse_args()
    
    process_experiment(args.experiment_name)
