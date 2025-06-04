import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import NamedTuple
from model import gratif_layers
from torch.nn.utils.rnn import pad_sequence

class Batch(NamedTuple):
    r"""A single sample of the data."""

    x_time: Tensor  # B×N:   the input timestamps.
    x_vals: Tensor  # B×N×D: the input values.
    x_mask: Tensor  # B×N×D: the input mask.

    y_time: Tensor  # B×K:   the target timestamps.
    y_vals: Tensor  # B×K×D: the target values.
    y_mask: Tensor  # B×K×D: teh target mask.

class Inputs(NamedTuple):
    r"""A single sample of the data."""

    t: Tensor
    x: Tensor
    t_target: Tensor

class Sample(NamedTuple):
    r"""A single sample of the data."""

    key: int
    inputs: Inputs
    targets: Tensor



def tsdm_collate(batch: list) -> Batch:
    r"""Collate tensors into batch.

    Transform the data slightly: t, x, t_target → T, X where X[t_target:] = NAN
    """
    x_vals: list[Tensor] = []
    y_vals: list[Tensor] = []
    x_time: list[Tensor] = []
    y_time: list[Tensor] = []
    x_mask: list[Tensor] = []
    y_mask: list[Tensor] = []

    context_x: list[Tensor] = []
    context_vals: list[Tensor] = []
    context_mask: list[Tensor] = []
    target_vals: list[Tensor] = []
    target_mask: list[Tensor] = []


    for sample in batch:
        t, x, t_target = sample.inputs
        y = sample.targets

        # get whole time interval
        sorted_idx = torch.argsort(t)

        # create a mask for looking up the target values
        mask_y = y.isfinite()
        mask_x = x.isfinite()

        # nan to zeros
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)


        x_vals.append(x[sorted_idx])
        x_time.append(t[sorted_idx])
        x_mask.append(mask_x[sorted_idx])

        y_time.append(t_target)
        y_vals.append(y)
        y_mask.append(mask_y)
        
        context_x.append(torch.cat([t, t_target], dim = 0))
        x_vals_temp = torch.zeros_like(x)
        y_vals_temp = torch.zeros_like(y)
        context_vals.append(torch.cat([x, y_vals_temp], dim=0))
        context_mask.append(torch.cat([mask_x, y_vals_temp], dim=0))
        # context_y = torch.cat([context_vals, context_mask], dim=2)

        target_vals.append(torch.cat([x_vals_temp, y], dim=0))
        target_mask.append(torch.cat([x_vals_temp, mask_y], dim=0))
        # target_y = torch.cat([target_vals, target_mask], dim=2)

    return Batch(
        x_time=pad_sequence(context_x, batch_first=True),
        x_vals=pad_sequence(context_vals, batch_first=True, padding_value=0),
        x_mask=pad_sequence(context_mask, batch_first=True),
        y_time=pad_sequence(context_x, batch_first=True),
        y_vals=pad_sequence(target_vals, batch_first=True, padding_value=0),
        y_mask=pad_sequence(target_mask, batch_first=True),
    )


def transform_data_for_grafiti(x_time, x_vals, x_mask, y_time, y_vals, y_mask, device):
    x_time = torch.cat([x_time, y_time], axis=1)
    y_time = x_time.clone()
    x_likes = torch.zeros_like(x_mask)
    y_likes = torch.zeros_like(y_mask)

    x_mask = torch.cat([x_mask, y_likes], axis=1)
    y_mask = torch.cat([x_likes, y_mask], axis=1)
    x_vals = torch.cat([x_vals, y_likes], axis=1)
    y_vals = torch.cat([x_likes, y_vals], axis=1)
    return x_time, x_vals, x_mask, y_time, y_vals, y_mask

class GrATiF(nn.Module):

    def __init__(
        self,
        args,
        input_dim):
        super().__init__()
        self.dim = input_dim
        self.attn_head = args.attn_head
        self.latent_dim = args.latent_dim
        self.n_layers = args.nlayers
        self.device = args.device
        self.enc = gratif_layers.Encoder(self.dim, self.latent_dim, self.n_layers, self.attn_head, device=args.device)
        self.plugin = args.plugin

        if(self.plugin):
            self.epsilon = 1e-6
    
    def get_extrapolation(self, context_x, context_w, target_x, target_y):
        context_mask = context_w[:, :, self.dim:]
        X = context_w[:, :, :self.dim]
        X = X*context_mask
        context_mask = context_mask + target_y[:,:,self.dim:]
        output, target_U_, target_mask_ = self.enc(context_x, X, context_mask, target_y[:,:,:self.dim], target_y[:,:,self.dim:])
        return output, target_U_, target_mask_

    def convert_data(self,  x_time, x_vals, x_mask, y_time, y_vals, y_mask):
        return x_time, torch.cat([x_vals, x_mask],-1), y_time, torch.cat([y_vals, y_mask],-1)  

    def forward(self, y_time, x_vals, x_time, x_mask, y_mask, y_vals):
        context_x, context_y, target_x, target_y = self.convert_data(x_time, x_vals, x_mask, y_time, y_vals, y_mask)
        # pdb.set_trace()
        if len(context_y.shape) == 2:
            context_x = context_x.unsqueeze(0)
            context_y = context_y.unsqueeze(0)
            target_x = target_x.unsqueeze(0)
            target_y = target_y.unsqueeze(0)
        output, target_U_, target_mask_ = self.get_extrapolation(context_x, context_y, target_x, target_y)

        return output, target_U_, target_mask_.to(torch.bool)
    
    def forecasting(self, batch_dict):
        y_time, x_vals, x_time, x_mask, y_mask, y_vals = \
        	batch_dict["tp_to_predict"].squeeze(0), \
			batch_dict["observed_data"].squeeze(0), \
			batch_dict["observed_tp"].squeeze(0), \
			batch_dict["observed_mask"].squeeze(0), \
			batch_dict["mask_predicted_data"].squeeze(0),\
			batch_dict["data_to_predict"].squeeze(0),\
            
        x_vals[abs(x_vals) > 1e4] = 1e-5
        x_time, x_vals, x_mask, y_time, y_vals, y_mask = (
            transform_data_for_grafiti(x_time, x_vals, x_mask, y_time, y_vals, y_mask, device=self.device)
        )
        output, target_U_, target_mask_ = self.forward(
            x_time=x_time,
            x_vals=x_vals,
            y_time=y_time,
            x_mask=x_mask,
            y_vals=y_vals,
            y_mask=y_mask,
        )
        y_pred = torch.zeros_like(y_vals)
        y_pred[y_mask.to(bool)] = output.squeeze(-1)[target_mask_]
        y_vals_ = torch.zeros_like(y_vals)
        y_vals_[y_mask.to(bool)] = target_U_[target_mask_]
        assert (y_vals_ == y_vals).all()

        if((y_pred>1e2).any()):
            print("y_pred > 1e2!")

        return y_pred, y_vals, y_mask