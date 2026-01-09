import torch
import torch.nn as nn
import torch.nn.functional as F

        
class iPatchEncoderLayer(nn.Module):
    def __init__(self, attention_var, attention_cycle, d_model, num_cycles, d_ff=None, dropout=0.1, activation="relu"):
        super(iPatchEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention_var = attention_var
        self.attention_cycle = attention_cycle
        self.N = num_cycles
        
        self.conv1 = nn.Conv1d(in_channels=d_model*self.N, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model*self.N, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
            
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        
        # (batch_size, num_cycles * num_variates, d_model)
        B, NC, D = x.size()
        C = int(NC / self.N)
        
        # Reshape for attending over num_variates 
        x_var = x.view(B, C, self.N, D).permute(0, 2, 1, 3).reshape(B * self.N, C, D)
        
        new_x, attn_var = self.attention_var(
            x_var, x_var, x_var,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        
        # Reshape for attending over num_cycles 
        new_x = new_x.reshape(B, self.N, C, D).permute(0, 2, 1, 3).reshape(B * C, self.N, D)
        
        new_x, attn_cycle = self.attention_cycle(
            new_x, new_x, new_x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )

        # Reshape back to the original shape (batch_size, num_cycles * num_variates, d_model)
        new_x = new_x.reshape(B, self.N*C, D)
        
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)        
        
        y = y.reshape(B, C, self.N, D).reshape(B, C, D*self.N) # (batch_size, num_variates, num_cycles * d_model)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        y = y.reshape(B, C, self.N, D).permute(0, 2, 1, 3).reshape(B, self.N, C*D).reshape(B, self.N, C, D).permute(0, 2, 1, 3).reshape(B, self.N * C, D) #(batch_size, num_variates * num_cycles, d_model)
            
        return self.norm2(x + y), (attn_var, attn_cycle)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        # Create a long enough P matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x
    

class PeriodicityReshape(nn.Module):
    def __init__(self, main_cycle):
        super(PeriodicityReshape, self).__init__()
        if main_cycle < 1:
            raise ValueError(f'Invalid main_cycle: {main_cycle}. Must be >= 1.')
        self.main_cycle = main_cycle
        
    def __assert_seq_len(self, x):
        _, n_steps, _ = x.shape
        seq_too_long = (n_steps % self.main_cycle) # is > 0 if n_steps is not a multiple of main_cycle -> True
        if seq_too_long:
            raise ValueError(f'''Number of steps {n_steps} is not a multiple of the main cycle ({n_steps}%{self.main_cycle}={n_steps%self.main_cycle}).
                             Suggested: Fill the sequence with zeros at the end to make it a multiple of the main cycle.''') 
            
    def apply(self, x, batch_size, n_features):
        self.__assert_seq_len(x)
        x = x.reshape(batch_size, -1, self.main_cycle, n_features).permute(0, 3, 1, 2)
        x = x.reshape(batch_size, -1, self.main_cycle).permute(0, 2, 1)
        return x

    def revert(self,x, batch_size, n_features):
        x = x.permute(0, 2, 1).reshape(batch_size, n_features, -1, self.main_cycle)
        x = x.permute(0, 2, 3, 1).reshape(batch_size, -1, n_features)
        return x

    def forward(self, x, n_features, direction):
        batch_size = x.shape[0]
        if direction == 'apply':
            return self.apply(x, batch_size, n_features)
        elif direction == 'revert':
            return self.revert(x, batch_size, n_features)
        else:
            raise ValueError(f'Invalid direction: {direction}. Use "apply" or "revert".')
        