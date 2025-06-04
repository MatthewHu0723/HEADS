import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DensityAwareModel(nn.Module):
    def __init__(self, backbone, feature_dim, dropout, hid_dim=64, lamb=0.5, pooling=True):
        super().__init__()
        self.backbone = backbone  # assumed to take (B, T, F) and output (B, L, F)
        self.dens_plugin = DensityPluginGlobal(feature_dim, hid_dim, dropout, lamb, pooling)
    
    def forecasting(self, batch_dict):
        y_pred, y, y_mask = self.backbone.forecasting(batch_dict)
        d_in, d_out = batch_dict['regi_dens'].squeeze(0), batch_dict['regi_dens_y'].squeeze(0)
        y_final = self.dens_plugin(d_in, d_out, y_pred)

        return y_final, y, y_mask
    
class DensityPluginGlobal(nn.Module):
    def __init__(self, F, H, dropout=0.2, lamb=0.5, pooling=True):
        super().__init__()
        if(pooling):
            self.mlp = nn.Sequential(
                nn.Linear(F+2, 2*H), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(2*H, F), nn.Sigmoid()
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(F, 2*H), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(2*H, F), nn.Sigmoid()
            )
        
        self.F = F
        self.lamb = lamb # 超参
        self.pooling = pooling

    def forward(self, d_in, d_out, y_pred):
        d_diff = d_out - d_in
        if(self.pooling):
            # Global Shift Pooling
            d_mean = d_diff.mean(dim=1, keepdim=True)                    # (B,1)
            d_max  = d_diff.max(dim=1, keepdim=True)[0]                  # (B,1)
            # Concat + Weight Learning
            inp = torch.cat([d_diff, d_mean, d_max], dim=-1)             # (B,2F)
        else:
            inp = d_diff

        W = self.mlp(inp).unsqueeze(1)                                # (B,1,F)

        return y_pred * W * self.lamb + y_pred * (1 - self.lamb)    # (B,L,F)