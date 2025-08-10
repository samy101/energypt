import numpy as np
import pandas as pd
import json
from time import time
import argparse
from pathlib import Path

import os
import sys
sys.path.append('./models') # Make sure this path is correct

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from models.model import MixForecast

from tqdm import tqdm

# --- Metrics (Unchanged) ---
def cal_cvrmse(pred, true, eps=1e-8):
    pred = np.array(pred)
    true = np.array(true)
    y_bar = true.mean()
    if abs(y_bar) < eps: return np.inf
    return np.sqrt(np.mean((pred - true) ** 2)) / y_bar

def cal_mae(pred, true):
    pred = np.array(pred)
    true = np.array(true)
    return np.mean(np.abs(pred - true))

def cal_nrmse(pred, true, eps=1e-8):
    true = np.array(true)
    pred = np.array(pred)

    M = len(true) // 24
    y_bar = np.mean(true)
    NRMSE = 100 * (1/ (y_bar+eps)) * np.sqrt((1 / (24 * M)) * np.sum((true - pred) ** 2))
    return NRMSE


# --- Data Handling (Aligned with Pre-training) ---

def standardize_series(series, eps=1e-8):
    """Standardizes a time series to have a mean of 0 and a standard deviation of 1."""
    mean = np.mean(series)
    std = np.std(series)
    standardized_series = (series - mean) / (std + eps)
    return standardized_series, mean, std

def unscale_predictions(predictions, mean, std, eps=1e-8):
    """Reverses the standardization process."""
    return predictions * (std + eps) + mean

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series forecasting.
    This is identical to the one in your pre-training script.
    """
    def __init__(self, data, backcast_length, forecast_length, stride=1):
        # Expects data to be pre-filled (NaNs handled before this class)
        self.data, self.mean, self.std = standardize_series(data)
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.stride = stride

    def __len__(self):
        num_samples = (len(self.data) - self.backcast_length - self.forecast_length) // self.stride + 1
        return max(0, num_samples)

    def __getitem__(self, index):
        start_index = index * self.stride
        x = self.data[start_index : start_index + self.backcast_length]
        y = self.data[start_index + self.backcast_length : start_index + self.backcast_length + self.forecast_length]
        # The model expects shape [batch, 1, seq_len]
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0), torch.tensor(y, dtype=torch.float32).unsqueeze(0)

# --- Evaluation Loop ---

def evaluate_zero_shot(args, model, criterion, device):
    test_root = Path(args["dataset_path"])
    result_dir = Path(args["result_path"])
    result_dir.mkdir(parents=True, exist_ok=True)

    median_results = []

    # Loop through each location directory
    for loc_name in tqdm(sorted(os.listdir(test_root)), desc="Processing Locations"):
        loc_path = test_root / loc_name
        if not loc_path.is_dir():
            continue

        loc_results_path = result_dir / loc_name
        loc_results_path.mkdir(exist_ok=True)
        
        location_building_results = []

        # Loop through each file in the location directory
        for filename in os.listdir(loc_path):
            if not filename.endswith('.csv'):
                continue
            
            file_path = loc_path / filename
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"\nCould not read {file_path}: {e}")
                continue

            # Loop through each column (building) in the Parquet file
            for building_id in df.columns:
                if not pd.api.types.is_numeric_dtype(df[building_id]):
                    continue

                series = df[building_id].values.astype(np.float32)

                # Skip if the series is too short
                if len(series) < args["seq_len"] + args["pred_len"]:
                    continue
                
                # --- FIX: Handle NaN values with Zero-Imputation ---
                series = np.nan_to_num(series, nan=0.0)

                dataset = TimeSeriesDataset(series, args["seq_len"], args["pred_len"], args["stride"])
                if len(dataset) == 0:
                    continue

                # --- Start of individual building evaluation ---
                model.eval()
                val_losses = []
                y_true_test = []
                y_pred_test = []

                loader = DataLoader(dataset, batch_size=args.get("batch_size", 128), shuffle=False)

                for x_test, y_test in tqdm(loader, desc=f"Testing {building_id}", leave=False):
                    x_test, y_test = x_test.to(device), y_test.to(device)
                    with torch.no_grad():
                        _, forecast = model(x_test)
                        loss = criterion(forecast, y_test)
                        val_losses.append(loss.item())
                        
                        y_test = y_test.squeeze(1)
                        forecast = forecast.squeeze(1)
                        y_true_test.extend(y_test.cpu().numpy())
                        y_pred_test.extend(forecast.cpu().numpy())
                
                
                y_true_combine =np.concatenate(y_true_test, axis=0)
                y_pred_combine = np.concatenate(y_pred_test, axis=0)
                avg_test_loss = np.mean(val_losses)
                
                y_pred_unscaled = unscale_predictions(y_pred_combine, dataset.mean, dataset.std)
                y_true_unscaled = unscale_predictions(y_true_combine, dataset.mean, dataset.std)
                
                cvrmse = cal_cvrmse(y_pred_unscaled, y_true_unscaled)
                nrmse = cal_nrmse(y_pred_unscaled, y_true_unscaled)
                mae = cal_mae(y_pred_unscaled, y_true_unscaled)

                location_building_results.append([building_id, cvrmse, nrmse, mae, avg_test_loss])
        
        if location_building_results:
            cols = ['building_ID', 'CVRMSE', 'NRMSE', 'MAE', 'Avg_Test_Loss']
            loc_df = pd.DataFrame(location_building_results, columns=cols)
            loc_df.to_csv(loc_results_path / "results.csv", index=False)
            
            med_nrmse = loc_df['NRMSE'].median()
            median_results.append([loc_name, med_nrmse])

    med_cols = ['Dataset', 'NRMSE']
    median_df = pd.DataFrame(median_results, columns=med_cols)
    median_df.to_csv(result_dir / "median_results.csv", index=False)
    print(f"\nEvaluation complete. Results saved to {result_dir}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-Shot Time Series Forecasting with MixForecast')
    parser.add_argument('--config-file', type=str, default='./configs/config_in.json', help='Input config file path', required=True)
    file_path_arg = parser.parse_args()
    
    with open(file_path_arg.config_file, 'r') as f:
        args = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    num_patches = args["seq_len"] // args["patch_size"]
    
    model = MixForecast(
        device=device,
        forecast_length=args["pred_len"],
        backcast_length=args["seq_len"],
        patch_size=args["patch_size"],
        num_patches=num_patches,
        num_features=args["num_features"],
        hidden_dim=args["hidden_dim"],
        nb_blocks_per_stack=args["num_blocks_per_stack"],
        stack_layers=args["stack_layers"],
        factor=args["factor"],
    )#.to(device)

    model_path = Path(args["model_save_path"]) / "best_model.pth"
    print(f"Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))

    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model's parameter count: {param/1e6:.2f}M")

    start_time = time()
    evaluate_zero_shot(args=args, model=model, criterion= torch.nn.HuberLoss(), device=device)
    end_time = time() - start_time
    print(f"\nInference completed in {end_time:.2f} seconds")