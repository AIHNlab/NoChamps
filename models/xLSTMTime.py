import torch
import torch.nn as nn
from xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm.blocks.mlstm.block import mLSTMBlockConfig
from xlstm.blocks.slstm.block import sLSTMBlockConfig

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp2(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp2, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    """
    xLSTMTime Model adapted from https://github.com/swjtuer0762/xLSTMTime/blob/master/model.py
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        
        # Decomposition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp2(kernel_size)
        
        # Linear layers for seasonal and trend components
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        
        # Initialize weights
        self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len, self.seq_len]))
        self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len, self.seq_len]))
        
        # Projection layers
        self.projection1 = nn.Linear(self.pred_len, configs.d_model)
        self.projection2 = nn.Linear(configs.d_model, self.pred_len)
        
        # Create xLSTM configuration from configs parameters
        mlstm_config = mLSTMBlockConfig()
        slstm_config = sLSTMBlockConfig()
        
        # Create xLSTM block stack config using parameters from configs
        xlstm_config = xLSTMBlockStackConfig(
            mlstm_block=mlstm_config,
            slstm_block=slstm_config,
            num_blocks=configs.e_layers,
            embedding_dim=configs.d_model,
            add_post_blocks_norm=getattr(configs, 'add_post_blocks_norm', True),
            _block_map=getattr(configs, '_block_map', 1),
            context_length=self.seq_len
        )
        
        self.xlstm_stack = xLSTMBlockStack(xlstm_config)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc shape: [Batch, Sequence Length, Channel]
        
        # Apply decomposition
        seasonal_init, trend_init = self.decompsition(x_enc)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        
        x = seasonal_output + trend_output
        
        # Apply first projection
        x = self.projection1(x)
        
        # Apply xLSTM stack
        x = self.xlstm_stack(x)
        
        # Apply final projection and reshape
        x = self.projection2(x)
        x = x.permute(0, 2, 1)
        
        return x

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # For imputation task, you might want a different approach
        # This is a simple implementation - adjust as needed
        seasonal_init, trend_init = self.decompsition(x_enc)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        
        x = seasonal_output + trend_output
        x = self.projection1(x)
        x = self.xlstm_stack(x)
        x = self.projection2(x)
        x = x.permute(0, 2, 1)
        
        return x

    def anomaly_detection(self, x_enc):
        # For anomaly detection
        seasonal_init, trend_init = self.decompsition(x_enc)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        
        x = seasonal_output + trend_output
        x = self.projection1(x)
        x = self.xlstm_stack(x)
        x = self.projection2(x)
        x = x.permute(0, 2, 1)
        
        return x

    def classification(self, x_enc, x_mark_enc):
        # For classification task
        seasonal_init, trend_init = self.decompsition(x_enc)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        
        x = seasonal_output + trend_output
        x = self.projection1(x)
        x = self.xlstm_stack(x)
        
        # Pooling and classification head
        x = x.mean(dim=1)  # Global average pooling
        x = self.projection2(x)
        
        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        elif self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        elif self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        elif self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        else:
            raise ValueError(f"Unknown task name: {self.task_name}")
        