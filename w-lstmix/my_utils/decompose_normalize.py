from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import numpy as np
import pywt

def standardize_series(series, eps=1e-8):
    mean = np.mean(series)
    std = np.std(series)
    standardized_series = (series - mean) / (std + eps)
    return standardized_series, mean, std

def unscale_predictions(predictions, mean, std, eps=1e-8):
    return predictions * (std+eps) + mean




def decompose_series(series, method_decom, period=24, wavelet='db4', level=5):
    """
    Decomposes a time series into trend and seasonal+residual components.
    Assumes hourly data by default (period=24).
    """
    if method_decom == 'seasonal_decompose':
       result = seasonal_decompose(series, model='additive', period=period, extrapolate_trend='freq')
       trend = result.trend
       seasonal_plus_resid = series - trend

       # Handle NaNs from the trend's boundary effects
       # trend = pd.Series(trend).fillna(method='bfill').fillna(method='ffill').values
       trend = pd.Series(trend).bfill().ffill().values
       seasonal_plus_resid = pd.Series(seasonal_plus_resid).fillna(0).values

       return trend, seasonal_plus_resid
    
   
    ##Decomposes a time series into trend and seasonal+residual components using wavelet transform, adjust level to get more in depth decompostion.

    elif method_decom == 'wavelet':
        if level is None:
            level = pywt.dwt_max_level(len(series), pywt.Wavelet(wavelet).dec_len)

        coeffs = pywt.wavedec(series, wavelet, level=level)

        # Keep only the approximation, set detail coeffs to zero for clean trend
        trend_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        trend = pywt.waverec(trend_coeffs, wavelet)[:len(series)]

        seasonal_plus_resid = series - trend
        seasonal_plus_resid = pd.Series(seasonal_plus_resid).fillna(0).values

        return trend, seasonal_plus_resid