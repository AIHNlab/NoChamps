import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.iPatch_EncDec import PeriodicityReshape, PositionalEncoding, iPatchEncoderLayer
from layers.Autoformer_EncDec import series_decomp

class Model(nn.Module):
    '''
    iPatch is a hybrid transformer architecture that integrates temporal patching 
    from PatchTST with variate-centric attention from iTransformer.
    '''

    def __init__(self, configs):
        super(Model, self).__init__()
        self.main_cycle = configs.main_cycle
        self.seq_len = configs.seq_len + configs.seq_len % self.main_cycle
        self.n_cycles = configs.seq_len // self.main_cycle
        self.n_features = configs.c_out
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.model_trend = configs.model_trend
        self.use_norm = configs.use_norm
        self.x_mark_size = configs.x_mark_size
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.main_cycle, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.positional_encoding = PositionalEncoding(configs.d_model)
        self.periodicity_reshape = PeriodicityReshape(self.main_cycle)
        self.trend_projection = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.pred_len)
        )
        self.decomposition = series_decomp(self.main_cycle)
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.main_cycle, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.positional_encoding = PositionalEncoding(configs.d_model)
        self.periodicity_reshape = PeriodicityReshape(self.main_cycle)
        # Encoder
        self.encoder = Encoder(
            [
                iPatchEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    self.n_cycles,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model*self.n_cycles, configs.pred_len, bias=True)
        else:
            raise NotImplementedError("Only forecasting task is implemented in iPatch model.")
            

    def _apply_positional_encoding(self, enc_out):
        enc_out_parts = torch.chunk(enc_out, self.n_features+self.x_mark_size, dim=1) # Split along the second dimension
        encoded_parts = [self.positional_encoding(part) for part in enc_out_parts] # Apply positional encoding to each part
        enc_out = torch.cat(encoded_parts, dim=1) # Combine the parts
        return enc_out
    
    def _apply_padding(self, x_enc, x_mark_enc):
        if x_enc.shape[1] % self.main_cycle:
            pad_len = self.main_cycle - x_enc.shape[1] % self.main_cycle
            x_enc = F.pad(x_enc, (0, 0, 0, pad_len), 'replicate')
            x_mark_enc = F.pad(x_mark_enc, (0, 0, 0, pad_len), 'constant', 0)
        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        if self.x_mark_size == 0:
            x_mark_enc = None
        else:
            x_mark_enc = self.periodicity_reshape(x_mark_enc, self.x_mark_size, 'apply')
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            
        # Reshape by periodicity
        x_enc = self.periodicity_reshape(x_enc, self.n_features, 'apply')

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self._apply_positional_encoding(enc_out)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out.reshape(enc_out.shape[0], self.n_features+self.x_mark_size, self.n_cycles, self.d_model).reshape(enc_out.shape[0], self.n_features+self.x_mark_size, self.d_model*self.n_cycles)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :self.n_features]
        
        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * (stdev[:, -1, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, -1, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out

    def forecast_with_trend(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.x_mark_size == 0:
            x_mark_enc = None
        else:
            x_mark_enc = self.periodicity_reshape(x_mark_enc, self.x_mark_size, 'apply')
        # Add decomposition from DLinear
        seasonal_init, trend_init = self.decomposition(x_enc)
        
        if self.use_norm:
            # Normalize seasonal component
            means_seasonal = seasonal_init.mean(1, keepdim=True).detach()
            seasonal_init = seasonal_init - means_seasonal
            stdev_seasonal = torch.sqrt(torch.var(seasonal_init, dim=1, keepdim=True, unbiased=False) + 1e-5)
            seasonal_init /= stdev_seasonal
            
            # Normalize trend component separately
            means_trend = trend_init.mean(1, keepdim=True).detach()
            trend_init = trend_init - means_trend  
            stdev_trend = torch.sqrt(torch.var(trend_init, dim=1, keepdim=True, unbiased=False) + 1e-5)
            trend_init /= stdev_trend

        # Process seasonal component
        x_enc = self.periodicity_reshape(seasonal_init, self.n_features, 'apply')
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self._apply_positional_encoding(enc_out)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out.reshape(enc_out.shape[0], self.n_features+self.x_mark_size, self.n_cycles, self.d_model).reshape(enc_out.shape[0], self.n_features+self.x_mark_size, self.d_model*self.n_cycles)
        seasonal_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :self.n_features]
        
        # Process trend with simple linear projection like DLinear
        trend_out = self.trend_projection(trend_init.permute(0, 2, 1)).permute(0, 2, 1)
        
        if self.use_norm:
            # De-normalize seasonal
            seasonal_out = seasonal_out * (stdev_seasonal[:, -1, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            seasonal_out = seasonal_out + (means_seasonal[:, -1, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            
            # De-normalize trend
            trend_out = trend_out * (stdev_trend[:, -1, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            trend_out = trend_out + (means_trend[:, -1, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        # Combine components
        dec_out = seasonal_out + trend_out
        
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x_enc, x_mark_enc = self._apply_padding(x_enc, x_mark_enc) # Apply padding if necessary
            
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.model_trend:
                dec_out = self.forecast_with_trend(x_enc, x_mark_enc, x_dec, x_mark_dec)
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        
        else:
            raise NotImplementedError("Only forecasting task is implemented in iPatch model.")
