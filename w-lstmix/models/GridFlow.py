####### Original Grid FLow Code#########

import torch
from torch import nn
from torch.nn import functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return self.pe[:, :seq_len]
    




class MLPMixer(nn.Module):
    def __init__(self, patch_size, num_patches, patch_hidden_dim, time_hidden_dim):
        super(MLPMixer, self).__init__()

        self.patch_size = patch_size
        self.num_patches = num_patches

        self.patch_mix = nn.Sequential(
            nn.LayerNorm(self.num_patches),
            nn.Linear(self.num_patches, patch_hidden_dim),
            nn.GELU(),
            nn.Linear(patch_hidden_dim, self.num_patches)
        )
        
        self.time_mix = nn.Sequential(
            nn.LayerNorm(self.patch_size),
            nn.Linear(self.patch_size, time_hidden_dim),
            nn.GELU(),
            nn.Linear(time_hidden_dim, self.patch_size)
        )
      
    def forward(self, x):
        batch_size, context_length = x.shape
        x = x.view(batch_size, self.num_patches, self.patch_size)

        
        x = x + self.patch_mix(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.time_mix(x)
        
        x = x.reshape(batch_size, self.num_patches * self.patch_size)
        return x
    





class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, device, positional_embed_flag, dropout=0.2, max_len=5000):
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.positional_embed_flag = positional_embed_flag
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        

    def forward(self, x):
        batch_size, context_length = x.shape
        seq_length = context_length // self.embed_dim
        x = x.view(batch_size, seq_length, self.embed_dim)

        if self.positional_embed_flag:
            pe = self.positional_encoding(x).to(self.device)
            x = x + pe

        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))

        x = x.reshape(batch_size, seq_length * self.embed_dim)
        return x





class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, thetas_dim, device, backcast_length=10, forecast_length=5, num_heads=2, ff_hidden_dim=256, positional_embed_flag=False):
        super(EncoderBlock, self).__init__()
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_hidden_dim = ff_hidden_dim
        self.thetas_dim = thetas_dim

        self.device = device

        self.encoder = Encoder(embed_dim, num_heads, ff_hidden_dim, device, positional_embed_flag)
        
        self.theta_b_fc = nn.Linear(backcast_length, thetas_dim, bias=False)
        self.theta_f_fc = nn.Linear(backcast_length, thetas_dim, bias=False)

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
        x = self.encoder(x)
        
        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)

        backcast = self.backcast_fc(theta_b)
        forecast = self.forecast_fc(theta_f)

        return backcast, forecast
    



class MLPMixerBlock(nn.Module):
    def __init__(self, hidden_dim, thetas_dim, device, backcast_length=10, forecast_length=5, patch_size=8, num_patches=21):
        super(MLPMixerBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.device = device
        
        self.mlpMixer = MLPMixer(self.patch_size, self.num_patches, self.hidden_dim, self.hidden_dim)
        
        self.theta_b_fc = nn.Linear(backcast_length, thetas_dim, bias=False)
        self.theta_f_fc = nn.Linear(backcast_length, thetas_dim, bias=False)

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):

        x = self.mlpMixer(x)


        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)

        backcast = self.backcast_fc(theta_b)
        forecast = self.forecast_fc(theta_f)

        return backcast, forecast
    



class Model(nn.Module):
    def __init__(
            self,
            device=torch.device('cpu'),
            num_blocks_per_stack=3,
            forecast_length=24,
            backcast_length=168,
            patch_size=8,
            num_patches=21,
            thetas_dim=8,
            embed_dim=8,
            hidden_dim=256,
            num_heads=2,
            ff_hidden_dim=256,
            share_weights_in_stack=False,
            positional_embed_flag=True
    ):
        super(Model, self).__init__()

        self.device = device
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_blocks_per_stack = num_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.thetas_dim = thetas_dim
        self.num_heads = num_heads
        self.ff_hidden_dim = ff_hidden_dim
        self.embed_dim = embed_dim

        self.log_sigma_trend = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_season = nn.Parameter(torch.tensor(0.0))
        

        # Trend stack (Transformer-based)
        self.trend_stack = nn.ModuleList([
            EncoderBlock(embed_dim=self.embed_dim, 
                         thetas_dim=self.thetas_dim, 
                         device=self.device, 
                         backcast_length=self.backcast_length,
                         forecast_length=self.forecast_length,
                         num_heads=self.num_heads, 
                         ff_hidden_dim=self.ff_hidden_dim,
                         positional_embed_flag=positional_embed_flag)
            for _ in range(self.num_blocks_per_stack)
        ])

        # Seasonality stack (MLP-Mixer-based)
        self.seasonality_stack = nn.ModuleList([
            MLPMixerBlock(hidden_dim=self.hidden_dim, 
                          thetas_dim=self.thetas_dim,
                          device=self.device,
                          backcast_length=self.backcast_length,
                          forecast_length=self.forecast_length,
                          patch_size=self.patch_size,
                          num_patches=self.num_patches)
            for _ in range(self.num_blocks_per_stack)
        ])

        self.to(self.device)

    def forward(self, trend_input, seasonality_input):
        """
        trend_input: [batch_size, backcast_length]
        seasonality_input: [batch_size, backcast_length]
        """
        trend_forecast = torch.zeros(trend_input.size(0), self.forecast_length).to(self.device)
        seasonality_forecast = torch.zeros(seasonality_input.size(0), self.forecast_length).to(self.device)
        
        # Trend path (Transformer-based)
        for block in self.trend_stack:
            trend_backcast, trend_forecast_block = block(trend_input)
            trend_input = trend_input - trend_backcast
            trend_forecast += trend_forecast_block

        # Seasonality path (MLP-Mixer-based)
        for block in self.seasonality_stack:
            season_backcast, season_forecast_block = block(seasonality_input)
            seasonality_input = seasonality_input - season_backcast
            seasonality_forecast += season_forecast_block

        return trend_forecast, seasonality_forecast


    


