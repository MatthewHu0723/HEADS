import gc
import numpy as np
import sklearn as sk
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

import lib.utils as utils
from lib.utils import get_device

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent

def l_p(x, p):
	return torch.sum(torch.abs(x) ** p) ** (1 / p)

def calculate_s_nodata(t, p=2, q=2):
	# 定义 l_p 和 l_1 函数
	t = torch.tensor([0.5]).to(t.device) if len(t) == 0 else torch.cat((t, torch.tensor([0.5]).to(t.device)))
	t = torch.tensor([t[i] * 2 if(i==0) else (t[i] - t[i-1]) * 2 for i in range(len(t))])
	sum_prob_q = torch.sum(torch.tensor([ts ** q for ts in t])).to(t.device)
	# 防止出现对数函数的非法输入
	if sum_prob_q <= 0:
		# print("∑P(t_i)^q 的结果小于等于 0，无法计算对数。")
		return torch.tensor(1e-5).to(t.device)
	
	s = (1 / (1 - q)) * torch.log(sum_prob_q) / (torch.log(torch.tensor(t.shape[0])) + 1e-5)

	return s

def calculate_s_nodata(t, p=2, q=2):
    if len(t) == 0:
        t = np.array([0.5])
    else:
        t = np.concatenate([t, np.array([0.5])])
    t_trans = np.array([t[i] * 2 if i == 0 else (t[i] - t[i - 1]) * 2 for i in range(len(t))])
    sum_prob_q = np.sum(t_trans ** q)
    
    if sum_prob_q <= 0:
        return 1e-5

    s = (1 / (1 - q)) * np.log(sum_prob_q) / (np.log(len(t)) + 1e-5)
    return s

def mse(mu, data, indices = None):
	n_data_points = mu.size()[-1]

	if n_data_points > 0:
		mse = nn.MSELoss()(mu, data)
	else:
		mse = torch.zeros([1]).to(get_device(data)).squeeze()
	return mse
	
def compute_error_featurewise(truth, pred_y, mask, func, reduce, norm_dict=None, p = 2):
	# pred_y shape [n_traj_samples, n_batch, n_tp, n_dim]
	# truth shape  [n_bacth, n_tp, n_dim] or [B, L, n_dim]
	n_tp, n_dim = pred_y.size()
	truth_repeated = truth
	mask = mask

	if(func == "MSE"):
		error = ((truth_repeated - pred_y)**2) * mask # (n_traj_samples, n_batch, n_tp, n_dim)
	elif(func == "MAE"):
		error = torch.abs(truth_repeated - pred_y) * mask # (n_traj_samples, n_batch, n_tp, n_dim)
	elif(func == "MAPE"):
		if(norm_dict == None):
			mask = (truth_repeated != 0) * mask
			truth_div = truth_repeated + (truth_repeated == 0) * 1e-8
			error = torch.abs(truth_repeated - pred_y) / truth_div * mask
		else:
			data_max = norm_dict["data_max"]
			data_min = norm_dict["data_min"]
			truth_rescale = truth_repeated * (data_max - data_min) + data_min
			pred_y_rescale = pred_y * (data_max - data_min) + data_min
			mask = (truth_rescale != 0) * mask
			truth_rescale_div = truth_rescale + (truth_rescale == 0) * 1e-8
			error = torch.abs(truth_rescale - pred_y_rescale) / truth_rescale_div * mask
	else:
		raise Exception("Error function not specified")

	error_var_sum = error.reshape(-1).sum(dim=0) # (n_dim, )
	mask_count = mask.reshape(-1).sum(dim=0) # (n_dim, )
	if(reduce == "mean"):
		### 1. Compute avg error of each variable first 
		### 2. Compute avg error along the variables 
		error_var_avg = error_var_sum / (mask_count + 1e-8) # (n_dim, ) 
		# print("error_var_avg", error_var_avg.max().item(), error_var_avg.min().item(), (1.0*error_var_avg).mean().item())
		n_avai_var = torch.count_nonzero(mask_count)
		error_avg = error_var_avg.sum() / (n_avai_var + 1e-8)# (1, )
		
		return error_avg # a scalar (1, ) 
	
	elif(reduce == "sum"):
		# (n_dim, ) , (n_dim, ) 
		# error_var_avg = error_var_sum / (mask_count + 1e-8)
		return error_var_sum, mask_count  

	else:
		raise Exception("Reduce argument not specified!")

def compute_error_samplewise(truth, pred_y, mask, func, reduce, norm_dict=None, p = 2):
	# pred_y shape [n_traj_samples, n_batch, n_tp, n_dim]
	# truth shape  [n_bacth, n_tp, n_dim] or [B, L, n_dim]
	truth_repeated = truth

	if(func == "MSE"):
		error = ((truth_repeated - pred_y)**2) * mask # (n_traj_samples, n_batch, n_tp, n_dim)
	elif(func == "MAE"):
		error = torch.abs(truth_repeated - pred_y) * mask # (n_traj_samples, n_batch, n_tp, n_dim)
	elif(func == "MAPE"):
		if(norm_dict == None):
			mask = (truth_repeated != 0) * mask
			truth_div = truth_repeated + (truth_repeated == 0) * 1e-8
			error = torch.abs(truth_repeated - pred_y) / truth_div * mask
		else:
			data_max = norm_dict["data_max"]
			data_min = norm_dict["data_min"]
			truth_rescale = truth_repeated * (data_max - data_min) + data_min
			pred_y_rescale = pred_y * (data_max - data_min) + data_min
			mask = (truth_rescale != 0) * mask
			truth_rescale_div = truth_rescale + (truth_rescale == 0) * 1e-8
			error = torch.abs(truth_rescale - pred_y_rescale) / truth_rescale_div * mask

	else:
		raise Exception("Error function not specified")

	error_var_sum = error.reshape(-1).sum(dim=0) # (n_dim, )
	mask_count = mask.reshape(-1).sum(dim=0) # (n_dim, )
	if(reduce == "mean"):
		### 1. Compute avg error of each variable first 
		### 2. Compute avg error along the variables 
		error_var_avg = error_var_sum / (mask_count + 1e-8) # (n_dim, ) 
		# print("error_var_avg", error_var_avg.max().item(), error_var_avg.min().item(), (1.0*error_var_avg).mean().item())
		n_avai_var = torch.count_nonzero(mask_count)
		error_avg = error_var_avg.sum() / (n_avai_var + 1e-8)# (1, )
		
		return error_avg # a scalar (1, ) 
	
	elif(reduce == "sum"):
		# (n_dim, ) , (n_dim, ) 
		# error_var_avg = error_var_sum / (mask_count + 1e-8)
		return error_var_sum, mask_count  

	else:
		raise Exception("Reduce argument not specified!")
	
def compute_error(truth, pred_y, mask, func, reduce, norm_dict=None, p = 2):
	# pred_y shape [n_traj_samples, n_batch, n_tp, n_dim]
	# truth shape  [n_bacth, n_tp, n_dim] or [B, L, n_dim]

	if len(pred_y.shape) == 3: 
		pred_y = pred_y.unsqueeze(dim=0)
	n_traj_samples, n_batch, n_tp, n_dim = pred_y.size()
	truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
	mask = mask.repeat(pred_y.size(0), 1, 1, 1)

	if(func == "MSE"):
		error = ((truth_repeated - pred_y)**2) * mask # (n_traj_samples, n_batch, n_tp, n_dim)
	elif(func == "MAE"):
		error = torch.abs(truth_repeated - pred_y) * mask # (n_traj_samples, n_batch, n_tp, n_dim)
	elif(func == "MAPE"):
		if(norm_dict == None):
			mask = (truth_repeated != 0) * mask
			truth_div = truth_repeated + (truth_repeated == 0) * 1e-8
			error = torch.abs(truth_repeated - pred_y) / truth_div * mask
		else:
			data_max = norm_dict["data_max"]
			data_min = norm_dict["data_min"]
			truth_rescale = truth_repeated * (data_max - data_min) + data_min
			pred_y_rescale = pred_y * (data_max - data_min) + data_min
			mask = (truth_rescale != 0) * mask
			truth_rescale_div = truth_rescale + (truth_rescale == 0) * 1e-8
			error = torch.abs(truth_rescale - pred_y_rescale) / truth_rescale_div * mask
	else:
		raise Exception("Error function not specified")

	error_var_sum = error.reshape(-1, n_dim).sum(dim=0) # (n_dim, )
	mask_count = mask.reshape(-1, n_dim).sum(dim=0) # (n_dim, )
	if(reduce == "mean"):
		### 1. Compute avg error of each variable first 
		### 2. Compute avg error along the variables 
		error_var_avg = error_var_sum / (mask_count + 1e-8) # (n_dim, ) 
		# print("error_var_avg", error_var_avg.max().item(), error_var_avg.min().item(), (1.0*error_var_avg).mean().item())
		n_avai_var = torch.count_nonzero(mask_count)
		error_avg = error_var_avg.sum() / (n_avai_var + 1e-8)# (1, )
		
		return error_avg # a scalar (1, ) 
	
	elif(reduce == "sum"):
		# (n_dim, ) , (n_dim, ) 
		# error_var_avg = error_var_sum / (mask_count + 1e-8)
		return error_var_sum, mask_count  

	else:
		raise Exception("Reduce argument not specified!")

def compute_all_losses(model, batch_dict, itr, plugin=False, vis_spar=False, vis_mse=False):
	# Condition on subsampled points
	# Make predictions for all the points
	# shape of pred --- [n_traj_samples=1, n_batch, n_tp, n_dim]
	pred_y, y, y_mask = model.forecasting(batch_dict) 
	
	# Compute avg error of each variable first, then compute avg error of all variables
	
	mse = compute_error(y, pred_y, y_mask, func="MSE", reduce="mean") # a scalar

	rmse = torch.sqrt(mse)
	# print(mse, rmse)
	mae = compute_error(y, pred_y, y_mask, func="MAE", reduce="mean") # a scalar
	

	################################
	# mse loss
	loss = mse
	results = {}
	results["loss"] = loss
	results["mse"] = mse.item()
	results["rmse"] = rmse.item()
	results["mae"] = mae.item()

	'''
	if(vis_spar):
		return results, np.array(x_sparsity_mat) / float(pred_y.shape[0]), np.array(y_true_sparsity_mat) / float(pred_y.shape[0])
	elif(vis_mse):
		return results, partial_mse
	'''
	return results

def evaluation(model, dataloader, n_batches, plugin=False, dataset='physionet', method='tPatchGNN', itr=None, vis=False, logx=False, case_vis=False):

	n_eval_samples = 0
	n_eval_samples_mape = 0
	total_results = {}
	total_results["loss"] = 0
	total_results["mse"] = 0
	total_results["mae"] = 0
	total_results["rmse"] = 0
	total_results["mape"] = 0
	
	# Case Study 4
	mae_list, bias_list = [], []

	for ind in range(n_batches):

		batch_dict = utils.get_next_batch(dataloader)
		pred_y, y, y_mask = model.forecasting(batch_dict)

		pred_len = batch_dict["data_to_predict"].shape[-2]
		pred_y, y, y_mask = pred_y.squeeze(0), y.squeeze(0), y_mask.squeeze(0)
		pred_y, y, y_mask = pred_y[:, -pred_len:, :], y[:, -pred_len:, :], y_mask[:, -pred_len:, :]
		
		# Vis Start
		batch_size, input_dim = pred_y.shape[0], y.shape[-1]
		if(vis):
			x, x_tp, mask_x, y_pred, y_gt, mask_y, y_tp= batch_dict["observed_data"].reshape(batch_size, -1, input_dim), \
					batch_dict["observed_tp"].reshape(batch_size, -1, 1).repeat(1, 1, input_dim) if len(batch_dict["observed_tp"].shape) < 4 else batch_dict["observed_tp"].reshape(batch_size, -1, input_dim), \
					batch_dict["observed_mask"].reshape(batch_size, -1, input_dim), \
					pred_y, \
					batch_dict["data_to_predict"].squeeze(0), \
					batch_dict["mask_predicted_data"].squeeze(0), \
					batch_dict["tp_to_predict"].squeeze(0)
			
			for bat in range(pred_y.shape[0]):
				x_tp0 = x_tp[bat].cpu().detach().numpy()  # (L1, F)
				y_tp0 = y_tp[bat].cpu().detach().numpy()  # (L2)
				mask_x0 = mask_x[bat, :].cpu().detach().numpy()  # (L1, F)
				mask_y0 = mask_y[bat, :].cpu().detach().numpy()  # (L2, F)

				tmp_x_bias, tmp_y_bias = 0, 0
				# Scaling timestamps to fit in different tasks
				if(dataset=='activity'):
					split_ts=0.75
				elif(dataset=='ushcn'):
					split_ts=0.96
				else:
					split_ts=0.5	

				in_ratio, out_ratio = 0.5 / split_ts, 0.5 / (1-split_ts)

				for feat in range(pred_y.shape[-1]):
					xmask = mask_x0[:, feat]
					ymask = mask_y0[:, feat]

					new_x_tp = x_tp0[:, feat][xmask==1] * in_ratio
					new_y_tp = y_tp0[ymask==1]
					new_y_tp[new_y_tp >= split_ts] = new_y_tp[new_y_tp >= split_ts] - split_ts
					new_y_tp *= out_ratio

					tmp_y_bias += calculate_s_nodata(new_y_tp) * np.sum(ymask) / (ymask.size + 1e-5) + 1e-5
					# tmp_y_bias += np.sum(xmask) / (xmask.size + 1e-5) + 1e-5
					tmp_x_bias += calculate_s_nodata(new_x_tp) * np.sum(xmask) / (xmask.size + 1e-5) + 1e-5
					# tmp_x_bias += np.sum(ymask) / (ymask.size + 1e-5) + 1e-5

				tmp_mae, tmp_mask_cnt = compute_error_samplewise(batch_dict["data_to_predict"].squeeze(0)[bat,:,:], pred_y.squeeze(0)[bat,:,:], mask=batch_dict["mask_predicted_data"].squeeze(0)[bat,:,:], func="MAE", reduce="sum")
				if(logx): # 横坐标取对数
					if((tmp_y_bias / (tmp_x_bias + 1e-8)) < 1000 and (tmp_y_bias / (tmp_x_bias + 1e-8) > 0.001)):
						bias_list.append(np.log10(tmp_y_bias / (tmp_x_bias + 1e-8)))
						mae_list.append(float(tmp_mae / (tmp_mask_cnt + 1e-8)))
				else:
					if((tmp_y_bias / (tmp_x_bias + 1e-8)) < 10 and (tmp_y_bias / (tmp_x_bias + 1e-8) > 0)):
						bias_list.append(tmp_y_bias / (tmp_x_bias + 1e-8))
						mae_list.append(float(tmp_mae / (tmp_mask_cnt + 1e-8)))
				
		if(case_vis and ind <= 10):

			bias_diff = (batch_dict["regi_dens_y"] - batch_dict["regi_dens"]).squeeze(0)
			
			max_bias_feat = torch.argmax(bias_diff).item()
			b_idx, c_idx = np.unravel_index(max_bias_feat, bias_diff.shape)
			seq_pred = pred_y[b_idx, :, c_idx]  # 长度 L 的预测序列
			seq_true = y[b_idx, :, c_idx]      # 长度 L 的真值序列
			mask     = y_mask[b_idx, :, c_idx]
			# 如果只想画出有效位置，可以用 mask 进行过滤
			t = batch_dict["tp_to_predict"].squeeze(0)
			
			t_valid = t[b_idx][mask==1].cpu().detach().numpy()
			pred_valid = seq_pred[mask==1].cpu().detach().numpy()
			true_valid = seq_true[mask==1].cpu().detach().numpy()

			mask_x  = batch_dict["observed_mask"].reshape(batch_size, -1, input_dim)[b_idx,:,c_idx].cpu().detach().numpy()
			if(len(batch_dict["observed_tp"].shape)>=4):
				t_x = batch_dict["observed_tp"].reshape(batch_size, -1, input_dim)[b_idx,:,c_idx].cpu().detach().numpy()
			else: 
				t_x = batch_dict["observed_tp"].reshape(batch_size, -1)[b_idx,:].cpu().detach().numpy()
			x = batch_dict["observed_data"].reshape(batch_size, -1, input_dim)[b_idx,:,c_idx].cpu().detach().numpy()
			t_x, x= t_x[mask_x==1], x[mask_x==1]
			t_x = np.append(t_x, t_valid[0])
			x = np.append(x, true_valid[0])

			plt.figure(figsize=(10, 4))
			plt.plot(t_x, x, label='Observed', linestyle='-.', marker='o', linewidth=2)
			plt.plot(t_valid, true_valid, label='True', marker='o', linewidth=2)
			plt.plot(t_valid, pred_valid, label='Predicted', marker='o', linestyle='--')
			plt.xlabel('Time Step')
			plt.ylabel('Value')
			plt.title(f'Batch {b_idx}, Channel {c_idx} (Max d_diff={bias_diff[b_idx, c_idx]:.3f})')
			plt.legend()
			plt.tight_layout()
			plt.savefig(f"/home/yimianmatthew/Irregular-time-series-forecasting/baseline/t-PatchGNN/visualize/{dataset}/method{method}_plugin{plugin}_epoch{itr}_ind{ind}.png")
			plt.close()
			print("HEY")

			out = {"t_x":t_x,"x":x,"t_valid":t_valid,"pred_valid":pred_valid,"true_valid":true_valid}
			pickle.dump(out,open(f'/home/yimianmatthew/Irregular-time-series-forecasting/baseline/t-PatchGNN/visualize/data_storage/{method}/{dataset}/case_plugin{plugin}_epoch{itr}_ind{ind}.pkl','wb'))
		# Case Study 4: Sample-varied sparsity
		## data_max[data_max==0] = 1e8
		
		# print('consistency test:', batch_dict["data_to_predict"][batch_dict["mask_predicted_data"].bool()].sum(), batch_dict["mask_predicted_data"].sum()) # consistency test
		
		# (n_dim, ) , (n_dim, ) 
		
		se_var_sum, mask_count = compute_error(y, pred_y, y_mask, func="MSE", reduce="sum") # a vector

		ae_var_sum, _ = compute_error(y, pred_y, y_mask, func="MAE", reduce="sum") # a vector
		
		# sp_var_sum, _ = compute_error(y, pred_y, y_mask, func="SPAR", reduce="sum")
		# norm_dict = {"data_max": batch_dict["data_max"], "data_min": batch_dict["data_min"]}
		ape_var_sum, mask_count_mape = compute_error(y, pred_y, y_mask, func="MAPE", reduce="sum") # a vector

		# add a tensor (n_dim, )
		total_results["loss"] += se_var_sum
		total_results["mse"] += se_var_sum
		total_results["mae"] += ae_var_sum
		total_results["mape"] += ape_var_sum

		# n_eval_samples_featurewise = torch.stack(n_eval_samples_featurewise)
		n_eval_samples += mask_count
		n_eval_samples_mape += mask_count_mape
		
		if(plugin):
			d_target = batch_dict["regi_dens_y"].squeeze(0)
			if((d_target <= 1e-5).all()):
				print("HERE")
			d_mask = (d_target > 1e-5).squeeze(0)
			# loss_density = F.mse_loss(d_out[d_mask], d_target[d_mask])
			# total_results["loss"] += loss_density
		
	n_avai_var = torch.count_nonzero(n_eval_samples)
	n_avai_var_mape = torch.count_nonzero(n_eval_samples_mape)
	# n_avai_var_featurewise = torch.stack([torch.count_nonzero(n_featurewise) for n_featurewise in n_eval_samples_featurewise])
	
	### 1. Compute avg error of each variable first
	### 2. Compute avg error along the variables 
	total_results["loss"] = (total_results["loss"] / (n_eval_samples + 1e-8)).sum() / n_avai_var
	total_results["mse"] = (total_results["mse"] / (n_eval_samples + 1e-8)).sum() / n_avai_var
	total_results["mae"] = (total_results["mae"] / (n_eval_samples + 1e-8)).sum() / n_avai_var
	total_results["rmse"] = torch.sqrt(total_results["mse"])
	total_results["mape"] = (total_results["mape"] / (n_eval_samples_mape + 1e-8)).sum() / n_avai_var_mape
	

	if(vis):
		## bias 1 & bias 3
		combined = np.vstack((np.array([bias.item() for bias in bias_list]), np.array(mae_list)))
		np.save(f'/home/yimianmatthew/Irregular-time-series-forecasting/baseline/t-PatchGNN/visualize/data_storage/{method}/{dataset}/density-mae-bias-new-our-best.npy',combined)

	for key, var in total_results.items(): 
		if isinstance(var, torch.Tensor) and len(var.shape) == 0:
			var = var.item()
		total_results[key] = var

	return total_results

