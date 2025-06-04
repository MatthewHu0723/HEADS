import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import pickle

import lib.utils as utils
from torch.distributions import uniform

from torch.utils.data import DataLoader
from lib.physionet import *
from lib.ushcn import *
from lib.mimic import MIMIC
from lib.person_activity import *
from sklearn import model_selection

import matplotlib.pyplot as plt
from scipy.stats import norm

#####################################################################################################
def l_p_norm(x, p, d=0):
    """
    计算 tensor x 的 l_p 范数
    """
    return torch.sum(torch.abs(x) ** p, dim=d).pow(1 / p)

def sparsity_metric_4(x, p=2, d=0):
    """
    实现稀疏性指标 S_1,p 公式：S1,p(x) = l1(x) / (n * lp(x))
    """
    n = x.numel()
    l1_norm = torch.sum(torch.abs(x), dim=d)
    lp_norm = l_p_norm(x, p)
    return l1_norm / (n * lp_norm + 1e-8)

def perfect_sparsity_metric(x, q=1, p=2):
    """
    实现完美稀疏性指标公式 S_q,p*(x) = n^(1/p - 1/q) * (l_q(x) / l_p(x))
    注意：仅适用于 q < p
    """
    n = x.numel()
    lq_norm = l_p_norm(x, q)
    lp_norm = l_p_norm(x, p)
    return (n ** (1 / p - 1 / q)) * (lq_norm / (lp_norm + 1e-8))
	
def parse_datasets(args, patch_ts=False):

	device = args.device
	dataset_name = args.dataset

	##################################################################
	if(dataset_name not in ["physionet", "mimic"] or args.history == 24):
		if(args.alpha == 2.0):
			pkl_pth = f"../data/{dataset_name}/density_processed/data_objects_n{args.n}_patch{patch_ts}.pkl"
		else:
			pkl_pth = f"../data/{dataset_name}/density_processed/data_objects_n{args.n}_patch{patch_ts}_alpha{args.alpha}.pkl"
	else: # History-varied Sensitivity Study
		pkl_pth = f"../data/{dataset_name}/density_processed/data_objects_n{args.n}_patch{patch_ts}_his{args.history}.pkl"

	with open(pkl_pth, "rb") as f:
		src_data = pickle.load(f)
		
		data_list, data_min, data_max, time_max = src_data["data_list"], src_data["data_min"], src_data["data_max"], src_data["time_max"]
		train_data, val_data, test_data = data_list[0], data_list[1], data_list[2]
		input_dim = val_data[0]["observed_data"].size(-1)
		seen_data = train_data + val_data

		batch_size = min(min(seen_data[0]["observed_data"].size(0), args.batch_size), args.n)

	if dataset_name in ["physionet", "mimic"]:
		train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
		val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)
		test_dataloader = DataLoader(test_data, batch_size =1, shuffle=False)
		
		data_objects = {
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"val_dataloader": utils.inf_generator(val_dataloader),
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_val_batches": len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					# "attr": total_dataset.params, #optional
					"data_max": data_max, #optional
					"data_min": data_min,
					"time_max": time_max
					} #optional
		
		return data_objects

	##################################################################
	### USHCN dataset ###
	elif dataset_name == "ushcn":
		args.n_months = 48 # 48 monthes
		args.pred_window = 1 # predict future one month

		print("Dataset n_samples after time split:", len(train_data)+len(val_data)+len(test_data),\
			len(train_data), len(val_data), len(test_data))
		train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
		val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)
		test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

		data_objects = {
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"val_dataloader": utils.inf_generator(val_dataloader),
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_val_batches": len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					# "attr": total_dataset.params, #optional
					"data_max": data_max, #optional
					"data_min": data_min,
					"time_max": time_max
					} #optional
		
		return data_objects
		

	##################################################################
	### Activity dataset ###
	elif dataset_name == "activity":
		args.pred_window = 1000 # predict future 1000 ms
		batch_size = args.batch_size

		train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
		val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)
		test_dataloader = DataLoader(test_data, batch_size =1, shuffle=False)

		data_objects = {
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"val_dataloader": utils.inf_generator(val_dataloader),
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_val_batches": len(val_dataloader),
					"n_test_batches": len(test_dataloader),
					# "attr": total_dataset.params, #optional
					"data_max": data_max, #optional
					"data_min": data_min,
					"time_max": time_max
					} #optional
		
		return data_objects
	
