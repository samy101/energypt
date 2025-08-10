import torch
from torch import nn
from torch.nn import functional as F

class TSMixer(nn.Module):
    def __init__(self, patch_size, num_patches, num_features, intra_patch_hidden_dim, inter_patch_hidden_dim, feature_hidden_dim, factor):
        super(TSMixer, self).__init__()

        self.patch_size = patch_size // factor
        self.num_patches = num_patches * factor
        self.num_features = num_features
        self.factor = factor

        self.inter_patch_mix = nn.Sequential(
            nn.LayerNorm(self.num_patches),
            nn.Linear(self.num_patches, inter_patch_hidden_dim),
            nn.GELU(),
            nn.Linear(inter_patch_hidden_dim, self.num_patches)
        )
        
        self.intra_patch_mix = nn.Sequential(
            nn.LayerNorm(self.patch_size),
            nn.Linear(self.patch_size, intra_patch_hidden_dim),
            nn.GELU(),
            nn.Linear(intra_patch_hidden_dim, self.patch_size)
        )

        self.feature_mix = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Linear(self.num_features, feature_hidden_dim),
            nn.GELU(),
            nn.Linear(feature_hidden_dim, self.num_features)
        )

        
        
    def forward(self, x):
        batch_size, num_features, seq_len = x.shape
        x = x.view(batch_size, num_features, self.num_patches , self.patch_size)

        x = x + self.inter_patch_mix(x.transpose(2, 3)).transpose(2, 3)
        x = x + self.intra_patch_mix(x)
        x = x + self.feature_mix(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        x = x.reshape(batch_size, num_features, self.num_patches * self.patch_size)
        return x



class GenericBlock(nn.Module):
    def __init__(self, hidden_dim, thetas_dim, device, backcast_length=10, forecast_length=5, patch_size=8, num_patches=21, num_features=1, factor=1):
        super(GenericBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_features = num_features
        self.device = device
        self.factor = factor
        
        self.TSMixer = TSMixer(self.patch_size, self.num_patches, self.num_features, self.hidden_dim, self.hidden_dim, self.hidden_dim, self.factor)
        
        self.theta_b_fc = nn.Linear(backcast_length, thetas_dim, bias=False)
        self.theta_f_fc = nn.Linear(backcast_length, thetas_dim, bias=False)

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
        
        x = self.TSMixer(x)


        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)

        backcast = self.backcast_fc(theta_b)
        forecast = self.forecast_fc(theta_f)

        return backcast, forecast
    

class MixForecast(nn.Module):
    def __init__(
            self,
            device=torch.device('cpu'),
            forecast_length=24,
            backcast_length=168,
            patch_size=24,
            num_patches=7,
            num_features=1,
            thetas_dim=8,
            hidden_dim=256,
            stack_layers=3,
            nb_blocks_per_stack=3,
            share_weights_in_stack=False,
            factor = [4,2,1],
    ):
        super(MixForecast, self).__init__()
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.thetas_dim = thetas_dim
        self.device = device
        self.stack_layers = stack_layers
        self.factor = factor

        # Using nn.ModuleList to hold blocks so that parameters are registered properly
        self.parameters = []

        self.stacks = [self.create_stack(self.factor[0]) for i in range(self.stack_layers // len(self.factor))]
        self.stacks.extend([self.create_stack(self.factor[1]) for i in range(self.stack_layers // len(self.factor))])
        self.stacks.extend([self.create_stack(self.factor[2]) for i in range(self.stack_layers // len(self.factor))])
        self.parameters = nn.ParameterList(self.parameters)

        self.to(self.device)


    def create_stack(self, fact):
        blocks = []
        for _ in range(self.nb_blocks_per_stack):
            block = GenericBlock(
                self.hidden_dim, self.thetas_dim,
                self.device, self.backcast_length, self.forecast_length, 
                self.patch_size, self.num_patches, self.num_features,
                fact,
            )
            self.parameters.extend(block.parameters())
            blocks.append(block)
        return blocks


    def forward(self, backcast):
        forecast = torch.zeros(backcast.size(0), backcast.size(1), self.forecast_length).to(self.device)
        for stack in self.stacks:
            for block in stack:
                backcast_block, forecast_block = block(backcast)
                backcast = backcast - backcast_block  
                forecast += forecast_block  
        return backcast, forecast
