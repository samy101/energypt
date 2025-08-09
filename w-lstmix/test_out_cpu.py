#!/usr/bin/env python
# coding: utf-8

# In[ ]:



############ changes for different models
# 1. change the model name in import statement for model
# 2. change the config file name and change the parameters and folders path in the config file

import numpy as np
import pandas as pd
import argparse
import json
import os
import sys
sys.path.append('./model')

import torch
from torch.utils.data import Dataset, DataLoader
from models import GridFlow_lstm_mlp as  GridFlow
from torch.utils.data import ConcatDataset

from tqdm import tqdm
from my_utils.metrics import cal_cvrmse, cal_mae, cal_mse, cal_nrmse
from my_utils.decompose_normalize import standardize_series, unscale_predictions, decompose_series




# In[ ]:

class DecomposedTimeSeriesDataset(Dataset):
    def __init__(self, series, backcast_length, forecast_length, method_decom, stride=1, period=24):
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.stride = stride
        self.method_decom = method_decom

        series = series.astype(np.float32)

        # <<< FIX 1: Check for and handle series that are entirely NaN >>>
        # This is the main cause of the crash.
        if np.all(np.isnan(series)):
            # If the series is all NaNs, it's invalid. Mark it for skipping.
            self.trend = None # Use this as a flag for an invalid dataset
            return

        # Your original NaN-filling code now only runs on series with at least one valid number.
        nan_mask = np.isnan(series)
        if np.any(nan_mask):
            mean_val = np.nanmean(series)
            series[nan_mask] = mean_val

        # This line will now only be reached with valid, NaN-free data.
        trend, seasonality = decompose_series(series, method_decom, period=period)

        # Standardize each component
        self.trend, self.trend_mean, self.trend_std = standardize_series(trend)
        self.season, self.season_mean, self.season_std = standardize_series(seasonality)

    def __len__(self):
        # <<< FIX 2: If the dataset is invalid (all-NaN), its length is 0 >>>
        if self.trend is None:
            return 0
        
        num_samples = (len(self.trend) - self.backcast_length - self.forecast_length) // self.stride + 1
        return max(0, num_samples)

    def __getitem__(self, idx):
        # This part doesn't need to change, as it won't be called for invalid datasets.
        start = idx * self.stride
        # Inputs
        trend_input = self.trend[start : start + self.backcast_length]
        season_input = self.season[start : start + self.backcast_length]
        # Targets
        trend_target = self.trend[start + self.backcast_length : start + self.backcast_length + self.forecast_length]
        season_target = self.season[start + self.backcast_length : start + self.backcast_length + self.forecast_length]

        return {
            'trend_input': torch.tensor(trend_input, dtype=torch.float32),
            'season_input': torch.tensor(season_input, dtype=torch.float32),
            'trend_target': torch.tensor(trend_target, dtype=torch.float32),
            'season_target': torch.tensor(season_target, dtype=torch.float32),
        }

# In[ ]:


def test(args, model, criterion, device):
    folder_path = args['test_dataset_path']
    result_path = args['result_path']
    backcast_length = args['backcast_length']
    forecast_length = args['forecast_length']
    stride = args['stride']
    period = 24  # Assuming a daily period
    method_decom = args['method_decom']

    median_res = []
    # Loop through each location folder (e.g., 'AMPD', 'BDG-2')
    for region in os.listdir(folder_path):
        region_path = os.path.join(folder_path, region)
        if not os.path.isdir(region_path):
            continue

        results_path = os.path.join(result_path, region)
        os.makedirs(results_path, exist_ok=True)
        
        building_results = []

        # Find all Parquet files within the location folder
        #parquet_files = [f for f in os.listdir(region_path) if f.endswith('.parquet')]
        parquet_files = [f for f in os.listdir(region_path) if f.endswith('.csv')]

        for building_file in tqdm(parquet_files, desc=f"Processing Region {region}", leave=False):
            file_path = os.path.join(region_path, building_file)
            
            try:
                #df = pd.read_parquet(file_path)
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"Could not read {file_path}: {e}")
                continue

            # Iterate through each column (each building ID) in the Parquet file
            for building_id in df.columns:
                # Skip non-numeric columns like timestamps
                if not pd.api.types.is_numeric_dtype(df[building_id]):
                    continue
                
                energy_data = df[building_id].values
                dataset = DecomposedTimeSeriesDataset(energy_data, backcast_length, forecast_length, method_decom, stride, period)

                if len(dataset) == 0:
                    continue

                # --- Start of evaluation logic for a single time series ---
                model.eval()
                test_losses = []
                y_true_trend, y_true_seasonal = [], []
                y_pred_trend, y_pred_seasonal = [], []

                loader = DataLoader(dataset, batch_size=args['batch_size'], num_workers=4)
                for batch in loader:
                    trend_input = batch['trend_input'].to(device)
                    season_input = batch['season_input'].to(device)
                    trend_target = batch['trend_target'].to(device)
                    season_target = batch['season_target'].to(device)
                    
                    with torch.no_grad():
                        trend_pred, season_pred = model(trend_input, season_input)
                        loss_trend = criterion(trend_pred, trend_target)
                        loss_season = criterion(season_pred, season_target)
                        sum_loss = loss_trend + loss_season
                        alpha = loss_season / (sum_loss + 1e-8)
                        beta = loss_trend / (sum_loss + 1e-8)
                        loss = alpha * loss_trend + beta * loss_season
                        
                        test_losses.append(loss.item())
                        y_true_trend.extend(trend_target.cpu().numpy())
                        y_true_seasonal.extend(season_target.cpu().numpy())
                        y_pred_trend.extend(trend_pred.cpu().numpy())
                        y_pred_seasonal.extend(season_pred.cpu().numpy())

                if not y_true_trend: continue

                # Unscale and combine predictions
                y_true_combine_trend = np.concatenate(y_true_trend, axis=0)
                y_true_combine_seasonal = np.concatenate(y_true_seasonal, axis=0)
                y_pred_combine_trend = np.concatenate(y_pred_trend, axis=0)
                y_pred_combine_seasonal = np.concatenate(y_pred_seasonal, axis=0)
                
                y_true_trend_unscaled = unscale_predictions(y_true_combine_trend, dataset.trend_mean, dataset.trend_std)
                y_pred_trend_unscaled = unscale_predictions(y_pred_combine_trend, dataset.trend_mean, dataset.trend_std)
                y_true_seasonal_unscaled = unscale_predictions(y_true_combine_seasonal, dataset.season_mean, dataset.season_std)
                y_pred_seasonal_unscaled = unscale_predictions(y_pred_combine_seasonal, dataset.season_mean, dataset.season_std)

                y_pred_unscaled = y_pred_seasonal_unscaled + y_pred_trend_unscaled
                y_true_unscaled = y_true_seasonal_unscaled + y_true_trend_unscaled

                # Calculate metrics
                cvrmse = cal_cvrmse(y_pred_unscaled, y_true_unscaled)
                nrmse = cal_nrmse(y_pred_unscaled, y_true_unscaled)
                mae = cal_mae(y_pred_unscaled, y_true_unscaled)
                mse = cal_mse(y_pred_unscaled, y_true_unscaled)
                
                building_results.append([building_id, cvrmse, nrmse, mae, mse, np.mean(test_losses)])
                # --- End of evaluation logic for a single time series ---

        # Save results for the entire region
        if building_results:
            columns = ['building_ID', 'CVRMSE', 'NRMSE', 'MAE', 'MSE', 'Avg_Test_Loss']
            results_df = pd.DataFrame(building_results, columns=columns)
            results_df.to_csv(os.path.join(results_path, 'result.csv'), index=False)
            
            med_nrmse = results_df['NRMSE'].median()
            med_mae = results_df['MAE'].median()
            med_mse = results_df['MSE'].median()
            median_res.append([region, med_nrmse, med_mae, med_mse])

    # Save median results across all regions
    if median_res:
        med_columns = ['Dataset','NRMSE', 'MAE', 'MSE']
        median_df = pd.DataFrame(median_res, columns=med_columns)
        median_df.to_csv(os.path.join(result_path, "median_results_of_buildings.csv"), index=False)


# In[ ]:


if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='Time Series Forecasting')
    # parser.add_argument('--config-file', type=str, default='./configs/energy_data.json', help='Input config file path', required=True)
    # parser.add_argument('--seq_len', type=int, default=168, help='Input Sequence Length')
    # parser.add_argument('--stride', type=int, default=24, help='Input Stride')
    # parser.add_argument('--patch_size', type=int, default=24, help='Input Patch Length')
    # parser.add_argument('--hidden_dim', type=int, default=256, help='Input Hidden Dim')
    # parser.add_argument('--model_save_path', type=str, default='./checkpoints/MixBEATS', help='Enter model save path')
    # parser.add_argument('--result_path', type=str, default='./results/MixBEATS', help="Enter the results path")

    
    # # Parse known args
    # cli_args = parser.parse_args()

    # # Load config file
    # with open(cli_args.config_file, 'r') as f:
    #     args = json.load(f)


    # args['model_save_path'] = cli_args.model_save_path
    # args['seq_len'] = cli_args.seq_len
    # args['stride'] = cli_args.stride 
    # args['hidden_dim'] = cli_args.hidden_dim
    # args['patch_size'] = cli_args.patch_size
    # args['result_path'] = cli_args.result_path


    
    # # # Parameters
    # backcast_length = args['seq_len']
    # forecast_length = args['pred_len']
    # stride = args['stride']
    # batch_size = args['batch_size']
    # patch_size = args['patch_size']
    # hidden_dim = args['hidden_dim']
    # num_patches = backcast_length // patch_size

    config_file = "./configs/inference_out.json"
    with open(config_file, 'r') as f:
        args = json.load(f)

    # check device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define GridFlow model
    model = GridFlow.Model(
        device=device,
        num_blocks_per_stack=args['num_blocks_per_stack'],
        forecast_length=args['forecast_length'],
        backcast_length=args['backcast_length'],
        patch_size=args['patch_size'],
        num_patches=args['backcast_length'] // args['patch_size'],
        thetas_dim=args['thetas_dim'],
        hidden_dim=args['hidden_dim'],
        embed_dim=args['embed_dim'],
        num_heads=args['num_heads'],
        ff_hidden_dim=args['ff_hidden_dim'],
    )#.to(device)

    model_load_path = '{}/best_model.pth'.format(args['model_save_path'])
    model.load_state_dict(torch.load(model_load_path, weights_only=True, map_location='cpu'))



    # Define loss
    if args['loss'] == 'mse':
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.HuberLoss(reduction="mean", delta=1)


    # training the model and save best parameters
    test(args, model, criterion, device)


