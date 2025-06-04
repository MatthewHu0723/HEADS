import torch
import torch.nn as nn

import torch.nn.functional as F

def compute_x_density(x_mask, x_time, Feat):
    # 特征密度 (B, T, 1)
    feat_density = torch.sum(x_mask, dim=-1, keepdim=True) / Feat
    
    # 时间密度计算 (B, T, 1)
    time_diff = x_time[:, 1:] - x_time[:, :-1]
    time_diff = F.pad(time_diff, (0,0,1,0), value=1e-6)  # 首项补零
    time_density = 1 / (time_diff.unsqueeze(-1) + 1e-6)
    
    # 综合密度 (可训练融合)
    density_gate = torch.sigmoid(torch.cat([feat_density, time_density], dim=-1))
    x_density = feat_density * density_gate[:,:,0:1] + time_density * density_gate[:,:,1:2]
    
    return x_density


class InputFusion(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.density_proj = nn.Linear(1, input_dim)
        
    def forward(self, x, x_density):
        # 密度增强 (B, T, F)
        density_features = self.density_proj(x_density)
        
        # 门控融合
        fusion_gate = torch.sigmoid(density_features)
        return x * fusion_gate + density_features


class OutputAdapter(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.density_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
        self.output_layer = nn.Linear(hidden_dim*2, hidden_dim)
        
    def forward(self, backbone_output, x_density):
        # 密度预测 (B, L, 1)
        y_density = self.density_predictor(backbone_output)
        
        # 上下文增强
        context = torch.cat([backbone_output, y_density.expand(-1,-1,backbone_output.shape[-1])], dim=-1)
        return self.output_layer(context), y_density


class DensityAwareLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, pred, target, y_density, y_mask):
        # 基础MSE
        base_loss = F.mse_loss(pred*y_mask, target*y_mask)
        
        # 密度加权损失
        density_weight = torch.sigmoid(y_density.detach()) # 解耦梯度
        weighted_loss = torch.mean((pred - target)**2 * density_weight * y_mask)
        
        # 密度正则项
        density_reg = F.mse_loss(y_density[:,1:], y_density[:,:-1].detach())
        
        return self.alpha*weighted_loss + (1-self.alpha)*base_loss + 0.1*density_reg


class DensityAwareModel(nn.Module):
    def __init__(self, backbone, input_dim, hidden_dim=64):
        super().__init__()
        self.input_fusion = InputFusion(input_dim)
        self.backbone = backbone  # 用户自定义主干网络
        self.output_adapter = OutputAdapter(hidden_dim)
        self.loss_fn = DensityAwareLoss()
        
    def forecasting(self, batch_dict):

        x, x_mask, x_time = \
            batch_dict["observed_data"].squeeze(0), \
            batch_dict["observed_mask"].squeeze(0), \
            batch_dict["observed_tp"].squeeze(0), \
        
        B, F = x.shape[0], x.shape[-1]
        if(len(x.shape)==4):
            x, x_mask, x_time = x.reshape(B, -1, F), x_mask.reshape(B, -1, F), x_time.reshape(B, -1, F)
        # 计算输入密度
        x_density = batch_dict["regi_dens"]
        # x_density = compute_x_density(x_mask, x_time, x.shape[-1])
        
        # 输入融合
        fused_x = self.input_fusion(x, x_density.permute(1, 0, 2))
        batch_dict["observed_data"] = fused_x
        
        # 主干网络处理
        backbone_out, y, y_mask = self.backbone.forecasting_wo_density(batch_dict)
        
        # 输出适配
        final_out, y_density = self.output_adapter(backbone_out, x_density)
        
        return final_out, y, y_mask, y_density

    def compute_loss(self, pred, y_density, target, y_mask):
        return self.loss_fn(pred, target, y_density, y_mask)
        