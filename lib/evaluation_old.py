# 3.11 Version
import gc
import numpy as np
import sklearn as sk
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu
import matplotlib.pyplot as plt
import time

import lib.utils as utils
from lib.utils import get_device

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent

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

def calculate_s(x, t, p=2, q=2):
	return calculate_s_onlydata(x, p, q) * calculate_s_nodata(t, p, q)


def calculate_s_nodata(t, p=2, q=2):
	# 计算 P(t)^q 并求和
	prob_t = t #已归一化，无须处理
	sum_prob_q = np.sum([ts ** q for ts in prob_t])

	# 防止出现对数函数的非法输入
	if sum_prob_q <= 0:
		# print("∑P(t_i)^q 的结果小于等于 0，无法计算对数。")
		return 0

	s =  (1 / (1 - q)) * np.log(sum_prob_q)
	return s

def calculate_s_onlydata(x, p=2, q=2):
	# 定义 l_p 和 l_1 函数
	def l_p(x, p):
		return np.sum(np.abs(x) ** p) ** (1 / p)

	# 计算 l_p(x) 和 l_1(x)
	l_p_x = l_p(x, p)
	l_1_x = l_p(x, 1)

	# 防止出现分母为 0 的情况
	if l_1_x == 0:
		# print("l_1(x) 计算结果为 0，无法进行后续计算。")
		return 0

	s = len(x) * (l_p_x / l_1_x)
	return s
'''
# 修改数个版本后的calculate_s
def l_p(x, p):
	return torch.sum(torch.abs(x) ** p + 1e-5) ** (1 / p)

def calculate_s(x, t, p=2, q=2, switch=False):

	# 计算 l_p(x) 和 l_1(x)
	x = torch.tensor(x)
	t = torch.tensor(t)
	l_p_x = l_p(x, p)
	l_1_x = l_p(x, 1)

	# 防止出现分母为 0 的情况
	if l_1_x == 0:
		# print("l_1(x) 计算结果为 0，无法进行后续计算。")
		return torch.tensor(1e-5).to(x.device)
	# 计算 P(t)^q 并求和
	t = torch.tensor([0.5]) if len(t) == 0 else torch.cat((t, torch.tensor([0.5])))
	t = torch.tensor([t[i] * 2 if(i==0) else (t[i] - t[i-1]) * 2 for i in range(len(t))])

	prob_t = t
	sum_prob_q = torch.sum(torch.tensor([ts ** q for ts in prob_t]).to(x.device))
	# 防止出现对数函数的非法输入
	if sum_prob_q <= 0:
		# print("∑P(t_i)^q 的结果小于等于 0，无法计算对数。")
		return torch.tensor(1e-5).to(x.device)
	
	s = l_1_x / (len(x) * l_p_x + 1e-5) * (1 / (1 - q)) * torch.log(sum_prob_q) 
	return s.cpu().detach().numpy()

def calculate_s_nodata(t, p=2, q=2, switch=False):

	# 计算 l_p(x) 和 l_1(x)
	t = torch.tensor(t)

	# 计算 P(t)^q 并求和
	t = torch.tensor([0.5]) if len(t) == 0 else torch.cat((t, torch.tensor([0.5])))
	t = torch.tensor([t[i] * 2 if(i==0) else (t[i] - t[i-1]) * 2 for i in range(len(t))])

	prob_t = t
	sum_prob_q = torch.sum(torch.tensor([ts ** q for ts in prob_t]).to(t.device))
	# 防止出现对数函数的非法输入
	if sum_prob_q <= 0:
		# print("∑P(t_i)^q 的结果小于等于 0，无法计算对数。")
		return torch.tensor(1e-5).to(t.device)
	
	s = (1 / (1 - q)) * torch.log(sum_prob_q) 
	return s.cpu().detach().numpy()
'''

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
	elif(func == "SPAR"):
		error = torch.abs(perfect_sparsity_metric(truth_repeated * mask, 1, p) - perfect_sparsity_metric(pred_y * mask, 1, p))
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
	elif(func == "SPAR"):
		error = torch.abs(perfect_sparsity_metric(truth_repeated * mask, 1, p) - perfect_sparsity_metric(pred_y * mask, 1, p))
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
	elif(func == "SPAR"):
		error = torch.abs(perfect_sparsity_metric(truth_repeated * mask, 1, p) - perfect_sparsity_metric(pred_y * mask, 1, p))
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


def compute_all_losses(model, batch_dict, itr, detach=False, detach_para_dict=None, vis_spar=False, vis_mse=False):
	# Condition on subsampled points
	# Make predictions for all the points
	# shape of pred --- [n_traj_samples=1, n_batch, n_tp, n_dim]
	if(detach):
		pred_y, y, y_mask = model.forecasting(batch_dict["tp_to_predict"], 
			batch_dict["observed_data"], batch_dict["observed_tp"], 
			batch_dict["observed_mask"], batch_dict["mask_predicted_data"],
			batch_dict["data_to_predict"], 
			detach_para_dict["glob_ux"], detach_para_dict["glob_sx"],
			detach_para_dict["regi_ux"], detach_para_dict["regi_sx"],
			detach_para_dict["glob_dens"], detach_para_dict["regi_dens"]
		) 

	else:
		pred_y, y, y_mask = model.forecasting_wo_density(batch_dict["tp_to_predict"], 
				batch_dict["observed_data"], batch_dict["observed_tp"], 
				batch_dict["observed_mask"], batch_dict["mask_predicted_data"],
				batch_dict["data_to_predict"])
	
	
	# 每一个batch的timesteps是不固定的
	# batch['tp_to_predict'].shape = (batch_size, pred_timesteps)
	# batch['observed_data'].shape = (batch_size, 3, obs_timesteps, features)
	# print("pred:", pred_y.shape, batch_dict["mask_predicted_data"].shape)
	
	# Case Study 2: Time-Varied
	'''
	x, x_tp0, mask_x, y_pred, y_gt, mask_y, y_tp0= batch_dict["observed_data"].reshape(batch_size, -1, input_dim), batch_dict["observed_tp"].reshape(batch_size, -1, input_dim), batch_dict["observed_mask"].reshape(batch_size, -1, input_dim), pred_y.squeeze(0), batch_dict["data_to_predict"].squeeze(0), batch_dict["mask_predicted_data"].squeeze(0), batch_dict["tp_to_predict"]
	
	if(itr == 0 and vis_spar):
		# x_values = torch.sum(x * mask_x, axis=0) / (torch.sum(mask_x, axis=0) + 1e-8)
		# mean_values = torch.sum(y_pred * mask_y, axis=0) / (torch.sum(mask_y, axis=0) + 1e-8)
		# true_values = torch.sum(y_gt * mask_y, axis=0) / (torch.sum(mask_y, axis=0) + 1e-8)
		x_sparsity_mat = [0] * pred_y.shape[-1]
		y_pred_sparsity_mat = [0] * pred_y.shape[-1]
		y_true_sparsity_mat = [0] * pred_y.shape[-1]
		for bat in range(pred_y.shape[0]):
			x_values = (x * mask_x)[bat, :].cpu().detach().numpy()
			mean_values = (y_pred * mask_y)[bat, :].cpu().detach().numpy()
			true_values = (y_gt * mask_y)[bat, :].cpu().detach().numpy()
			x_tp0 = x_tp0[bat].cpu().detach().numpy()
			y_tp0 = y_tp0[bat].cpu().detach().numpy()

			for feat in range(pred_y.shape[-1]):

				x_value = x_values[:, feat]
				mean_value = mean_values[:, feat]
				true_value = true_values[:, feat]


				new_x_tp = x_tp0[:,feat][x_tp0[:,feat] != 0] # Erase 0
				new_x_tp = np.array([new_x_tp[i] if(i==0) else new_x_tp[i] - new_x_tp[i-1] for i in range(len(new_x_tp))]) # Differentiate
				new_x_tp = np.array([0.5]) if len(new_x_tp) == 0 else np.concatenate((new_x_tp, np.array([0.5 - new_x_tp[-1]])))
				new_y_tp = np.where((true_value != 0) & (y_tp0 != 0), y_tp0-0.5, 0)
				new_y_tp = new_y_tp[new_y_tp != 0]
				new_y_tp = np.array([new_y_tp[i] if(i==0) else new_y_tp[i] - new_y_tp[i-1] for i in range(len(new_y_tp))]) # Differentiate
				new_y_tp = np.array([0.5]) if len(new_y_tp) == 0 else np.concatenate((new_y_tp, np.array([0.5 - new_y_tp[-1]])))
				x_sparsity_mat[feat] += calculate_s(x_value, new_x_tp)
				# y_pred_sparsity_mat.append(calculate_s(mean_value, y_tp0))
				y_true_sparsity_mat[feat] += calculate_s(true_value, new_y_tp)

				x_tp = x_tp0[:, feat][x_value>1e-30]
				y_tp = y_tp0[true_value > 1e-30]
				x_value = x_value[x_value > 1e-30]
				
				mean_value = mean_values[:, feat][mean_value > 1e-30]
				true_value = true_values[:, feat][true_value > 1e-30]
				x_tp_arg = np.argsort(x_tp)
				y_tp_arg = np.argsort(y_tp)

				plt.scatter(x_tp[x_tp_arg], x_value[x_tp_arg], alpha=0.6, label='observed')
				plt.plot(y_tp[y_tp_arg], mean_value[y_tp_arg], alpha=0.6, label='pred_y')
				plt.plot(np.concatenate([x_tp[x_tp_arg], y_tp[y_tp_arg]]), np.concatenate([x_value[x_tp_arg],true_value[y_tp_arg]]), alpha=0.6, label='gt')
				plt.legend()
				plt.xlabel('Time(Standardized)')
				plt.ylabel('Data')
				plt.title('Batch 0, Feature '+ str(feat))

				plt.grid(True)
				# plt.savefig('/home/yimianmatthew/Irregular-time-series-forecasting/baseline/t-PatchGNN/visualize/pred_y_epoch'+str(itr)+'_feat'+str(feat)+'.png')
				plt.close()
				
		
			
			
		dim = np.arange(feat + 1)
		plt.plot(dim, np.array(x_sparsity_mat) / float(pred_y.shape[0]), alpha=0.6, label='observed')
		# plt.plot(dim, y_pred_sparsity_mat, alpha=0.6, label='pred_y')
		plt.plot(dim, np.array(y_true_sparsity_mat) / float(pred_y.shape[0]), alpha=0.6, label='gt')
		plt.legend()
		plt.xlabel('Features')
		plt.ylabel('Data')
		plt.title('Density Avg for Batch 0(Sample 0 ~ 31), Feature 0 ~ '+ str(feat))
		plt.savefig('/home/yimianmatthew/Irregular-time-series-forecasting/baseline/t-PatchGNN/visualize/density_epoch'+str(itr)+'_batch0.png')
		plt.close()
		'''

	# Compute avg error of each variable first, then compute avg error of all variables
	
	mse = compute_error(y, pred_y, y_mask, func="MSE", reduce="mean") # a scalar
	if(torch.isnan(mse).any()):
		print("HERE!")
	rmse = torch.sqrt(mse)
	# print(mse, rmse)
	mae = compute_error(y, pred_y, y_mask, func="MAE", reduce="mean") # a scalar
	'''
	artial_mse = [0] * x.shape[-1]
	for feat in range(x.shape[-1]):
		partial_mse[feat] += compute_error(batch_dict["data_to_predict"][:, :, feat], pred_y[:, :, :, feat], mask = batch_dict["mask_predicted_data"][:, :, feat], func="MSE", reduce="mean").item()
	'''
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

def evaluation(model, dataloader, n_batches, input_dim=41, batch_size=32, detach=False, detach_para_loader=None, dataset='physionet', method='tPatchGNN', itr=None, vis=False, logx=False):

	n_eval_samples = 0
	n_eval_samples_mape = 0
	total_results = {}
	total_results["loss"] = 0
	total_results["mse"] = 0
	total_results["mae"] = 0
	total_results["rmse"] = 0
	total_results["mape"] = 0
	'''
	n_eval_samples_featurewise = [0] * 41
	total_results["featurewise_mae"] = [0] * 41
	total_results["featurewise_mse"] = [0] * 41
	total_results["featurewise_mask"] = [0] * 41
	'''
	# Case Study 3
	x_sparsity_mat = [0] * input_dim
	y_pred_sparsity_mat = [0] * input_dim
	y_true_sparsity_mat = [0] * input_dim
	
	x_mask_ratio_mat = [0] * input_dim
	y_mask_ratio_mat = [0] * input_dim
	data_max = [0] * input_dim
	mask1_cnt = [0] * input_dim
	
	# Case Study 4
	mae_list, mae2_list, mae3_list, mae4_list, bias_list, bias2_list, bias3_list, bias4_list = [], [], [], [], [], [], [], []
	
	for ind in range(n_batches):
		batch_dict = utils.get_next_batch(dataloader)

		if(detach):
			detach_para_dict = utils.get_next_batch(detach_para_loader)
			pred_y, y, y_mask = model.forecasting(batch_dict["tp_to_predict"], 
				batch_dict["observed_data"], batch_dict["observed_tp"], 
				batch_dict["observed_mask"], batch_dict["mask_predicted_data"],
				batch_dict["data_to_predict"],
				detach_para_dict["glob_ux"], detach_para_dict["glob_sx"],
				detach_para_dict["regi_ux"], detach_para_dict["regi_sx"],
				detach_para_dict["glob_dens"], detach_para_dict["regi_dens"]
			) 
		else:
			pred_y, y, y_mask = model.forecasting_wo_density(batch_dict["tp_to_predict"], 
					batch_dict["observed_data"], batch_dict["observed_tp"], 
					batch_dict["observed_mask"], batch_dict["mask_predicted_data"], 
					batch_dict["data_to_predict"])
	
		pred_len = batch_dict["data_to_predict"].shape[-2]
		if(pred_y.shape[0]==1): pred_y = pred_y.squeeze(0)
		pred_y, y, y_mask = pred_y[:, -pred_len:, :], y[:, -pred_len:, :], y_mask[:, -pred_len:, :]

		batch_size = batch_dict["observed_data"].shape[0]
		x, x_tp, mask_x, y_pred, y_gt, mask_y, y_tp= batch_dict["observed_data"].reshape(batch_size, -1, input_dim), \
				batch_dict["observed_tp"].unsqueeze(-1).repeat(1, 1, input_dim).cpu().detach().numpy() if len(batch_dict["observed_tp"].shape) < 4 else batch_dict["observed_tp"].reshape(batch_size, -1, input_dim).cpu().detach().numpy(), \
				batch_dict["observed_mask"].reshape(batch_size, -1, input_dim), \
				pred_y, \
				batch_dict["data_to_predict"].squeeze(0), \
				batch_dict["mask_predicted_data"].squeeze(0), \
				batch_dict["tp_to_predict"].cpu().detach().numpy()

		# Case Study 1: Dimension-Varied
		mean_values = torch.mean(y_pred.squeeze(0) * mask_y, axis=(0, 1))
		true_values = torch.mean(y, axis=(0, 1))

		# Plot
		'''
		if(_ == 0 and itr != None):
			plt.plot(x_values.cpu(), mean_values.cpu(), label='pred_y')
			plt.plot(x_values, true_values.cpu(), label='gt')
			plt.legend()
			plt.title('The average of first two dimensions')
			plt.xlabel('Index')
			plt.ylabel('Average')

			plt.grid(True)
			plt.savefig('/home/yimianmatthew/Irregular-time-series-forecasting/baseline/t-PatchGNN/visualize/pred_y_epoch_'+str(itr)+'_batch_'+_+'.png')
			plt.close()
		'''
		
		# Case Study 3: Dimension-varied sparsity & Case Study 4: Sample-varied sparsity
		if(vis):
			for bat in range(y_pred.shape[0]): # Full Test Data
			# for bat in range(0,1): # 75 Test Sample
				x_values = (x * mask_x)[bat, :].cpu().detach().numpy()
				mean_values = (y_pred * mask_y)[bat, :].cpu().detach().numpy()
				true_values = (y * mask_y)[bat, :].cpu().detach().numpy()
				x_tp0 = x_tp[bat] # (L1, F)
				y_tp0 = y_tp[bat] # (L2)
				mask_x0 = mask_x[bat, :] # (L1, F)
				mask_y0 = mask_y[bat, :] # (L2, F)

				tmp_x_bias, tmp_y_bias = 0,0
				tmp_x_bias2, tmp_y_bias2 = 0,0
				tmp_x_bias4, tmp_y_bias4 = 0,0

				for feat in range(y_pred.shape[-1]):
					
					x_value = x_values[:, feat]
					mean_value = mean_values[:, feat]
					true_value = true_values[:, feat]
					xmask = mask_x0[:, feat]
					ymask = mask_y0[:, feat]
					
					new_x_index = (xmask != 0).cpu().detach().numpy()
					new_x, new_x_tp = x_value[new_x_index], x_tp0[:, feat][new_x_index] # Erase 0
					new_x_tp = np.array([0.5]) if len(new_x_tp) == 0 else np.concatenate((new_x_tp, np.array([0.5 - new_x_tp[-1]])))
					new_x_tp = np.array([new_x_tp[i] if(i==0) else new_x_tp[i] - new_x_tp[i-1] for i in range(len(new_x_tp))]) # Differentiate
					
					new_y_index = (ymask != 0).cpu().detach().numpy() 
					new_y, new_y_tp = true_value[new_y_index], y_tp0[new_y_index] - 0.5 # Erase 0
					new_y_tp = np.array([0.5]) if len(new_y_tp) == 0 else np.concatenate((new_y_tp, np.array([0.5 - new_y_tp[-1]])))
					new_y_tp = np.array([new_y_tp[i] if(i==0) else new_y_tp[i] - new_y_tp[i-1] for i in range(len(new_y_tp))]) # Differentiate
					
					# Case Study 3 Crossing Line
					
					x_sparsity_mat[feat] += calculate_s(x_value, new_x_tp)
					# y_pred_sparsity_mat.append(calculate_s(mean_value, y_tp0))
					y_true_sparsity_mat[feat] += calculate_s(true_value, new_y_tp)

					x_mask_ratio_mat[feat] += mask_x[:, :, feat].sum() / (mask_x.shape[0] * mask_x.shape[1] * mask_x.shape[2])
					y_mask_ratio_mat[feat] += mask_y[:, :, feat].sum() / (mask_y.shape[0] * mask_y.shape[1] * mask_y.shape[2])
					
					# Case Study 4 Crossing Line
					tmp_y_bias += calculate_s(new_y, new_y_tp) 
					tmp_x_bias += calculate_s(new_x, new_x_tp)

					tmp_y_bias2 += calculate_s_nodata(new_y_tp)
					tmp_x_bias2 += calculate_s_nodata(new_x_tp)

					tmp_y_bias4 += calculate_s_onlydata(new_y_tp)
					tmp_x_bias4 += calculate_s_onlydata(new_x_tp)

				tmp_mae, tmp_mask_cnt = compute_error_samplewise(batch_dict["data_to_predict"][bat,:,:], y_pred.squeeze(0)[bat,:,:], mask=batch_dict["mask_predicted_data"][bat,:,:], func="MAE", reduce="sum")
				
				x_mask_ratio_cnt, y_mask_ratio_cnt = mask_x0.sum(), mask_y0.sum()
				
				if(logx): # 横坐标取对数
					if((tmp_y_bias / (tmp_x_bias + 1e-8)) < 1000 and (tmp_y_bias / (tmp_x_bias + 1e-8) > 0.001)):
						bias_list.append(np.log10(tmp_y_bias / (tmp_x_bias + 1e-8)))
						mae_list.append(float(tmp_mae / (tmp_mask_cnt + 1e-8)))
					if((tmp_y_bias2 / (tmp_x_bias2 + 1e-8)) < 1000 and (tmp_y_bias2 / (tmp_x_bias2 + 1e-8) > 0.001)):
						bias2_list.append(np.log10(tmp_y_bias2 / (tmp_x_bias2 + 1e-8)))
						mae2_list.append(float(tmp_mae / (tmp_mask_cnt + 1e-8)))
					if((y_mask_ratio_cnt / (x_mask_ratio_cnt + 1e-8)) < 1000 and (y_mask_ratio_cnt / (x_mask_ratio_cnt + 1e-8) > 0.001)):
						bias3_list.append(np.log10(float(y_mask_ratio_cnt / (x_mask_ratio_cnt + 1e-8))))
						mae3_list.append(float(tmp_mae / (tmp_mask_cnt + 1e-8)))
					if((tmp_y_bias4 / (tmp_x_bias4 + 1e-8)) < 1000 and (tmp_y_bias4 / (tmp_x_bias4 + 1e-8) > 0.001)):
						bias4_list.append(np.log10(tmp_y_bias4 / (tmp_x_bias4 + 1e-8)))
						mae4_list.append(float(tmp_mae / (tmp_mask_cnt + 1e-8)))

				else:
					if((tmp_y_bias / (tmp_x_bias + 1e-8)) < 10 and (tmp_y_bias / (tmp_x_bias + 1e-8) > 0)):
						bias_list.append(tmp_y_bias / (tmp_x_bias + 1e-8))
						mae_list.append(float(tmp_mae / (tmp_mask_cnt + 1e-8)))
					if((tmp_y_bias2 / (tmp_x_bias2 + 1e-8)) < 10 and (tmp_y_bias2 / (tmp_x_bias2 + 1e-8) > 0)):
						bias2_list.append(tmp_y_bias2 / (tmp_x_bias2 + 1e-8))
						mae2_list.append(float(tmp_mae / (tmp_mask_cnt + 1e-8)))
					if((y_mask_ratio_cnt / (x_mask_ratio_cnt + 1e-8)) < 10 and (y_mask_ratio_cnt / (x_mask_ratio_cnt + 1e-8) > 0)):
						bias3_list.append(float(y_mask_ratio_cnt / (x_mask_ratio_cnt + 1e-8)))
						mae3_list.append(float(tmp_mae / (tmp_mask_cnt + 1e-8)))
					if((tmp_y_bias4 / (tmp_x_bias4 + 1e-8)) < 10 and (tmp_y_bias4 / (tmp_x_bias4 + 1e-8) > 0)):
						bias4_list.append(tmp_y_bias4 / (tmp_x_bias4 + 1e-8))
						mae4_list.append(float(tmp_mae / (tmp_mask_cnt + 1e-8)))

				'''
				for times in range(pred_y.shape[-2]):
					data_max = np.maximum(data_max, true_values[times, :])

				for times in range(x_values.shape[-2]):
					data_max = np.maximum(data_max, x_values[times, :])
				'''
		
		# Case Study 4: Sample-varied sparsity
		## data_max[data_max==0] = 1e8
		
		# print('consistency test:', batch_dict["data_to_predict"][batch_dict["mask_predicted_data"].bool()].sum(), batch_dict["mask_predicted_data"].sum()) # consistency test
		
		#/ (n_dim, ) , (n_dim, ) 
		se_var_sum, mask_count = compute_error(y, pred_y, y_mask, func="MSE", reduce="sum") # a vector

		ae_var_sum, _ = compute_error(y, pred_y, y_mask, func="MAE", reduce="sum") # a vector

		sp_var_sum, _ = compute_error(y, pred_y, y_mask, func="SPAR", reduce="sum")
		# norm_dict = {"data_max": batch_dict["data_max"], "data_min": batch_dict["data_min"]}
		ape_var_sum, mask_count_mape = compute_error(y, pred_y, y_mask, func="MAPE", reduce="sum") # a vector

		# add a tensor (n_dim, )
		total_results["loss"] += se_var_sum
		total_results["mse"] += se_var_sum
		total_results["mae"] += ae_var_sum
		total_results["mape"] += ape_var_sum
		'''
		for feat in range(input_dim):
			tmp_mse, tmp_mask = \
				compute_error_featurewise(batch_dict["data_to_predict"][:,:,feat], pred_y.squeeze(0)[:,:,feat], mask=batch_dict["mask_predicted_data"][:,:,feat], func="MSE", reduce="sum")
			total_results["featurewise_mse"][feat] += tmp_mse
			total_results["featurewise_mask"][feat] += tmp_mask
			tmp_mae, _ = \
				compute_error_featurewise(batch_dict["data_to_predict"][:,:,feat], pred_y.squeeze(0)[:,:,feat], mask=batch_dict["mask_predicted_data"][:,:,feat], func="MAE", reduce="sum")
			total_results["featurewise_mae"][feat] += tmp_mae
			n_eval_samples_featurewise[feat] += tmp_mask
		'''
		
		# n_eval_samples_featurewise = torch.stack(n_eval_samples_featurewise)
		n_eval_samples += mask_count
		n_eval_samples_mape += mask_count_mape
	'''
	total_results["featurewise_mse"], total_results["featurewise_mask"], total_results["featurewise_mae"] = \
		torch.stack(total_results["featurewise_mse"]), torch.stack(total_results["featurewise_mask"]), torch.stack(total_results["featurewise_mae"])
	n_eval_samples_featurewise = torch.stack(n_eval_samples_featurewise)
	'''
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
	'''
	total_results["featurewise_mse"] = (total_results["featurewise_mse"] / (n_eval_samples_featurewise + 1e-8)) / (n_avai_var_featurewise + 1e-8)
	total_results["featurewise_mae"] = (total_results["featurewise_mae"] / (n_eval_samples_featurewise + 1e-8)) / (n_avai_var_featurewise + 1e-8)
	'''

	if(vis):
		## Visualization for Case 3
		# dim = np.arange(feat + 1)
		
		# observed_gt_bias1 = np.maximum( \
		# 					    np.array(y_true_sparsity_mat) / float(pred_y.shape[0]) / (np.array(x_sparsity_mat) / float(pred_y.shape[0]) + 1e-8), \
		# 						np.array(x_sparsity_mat) / float(pred_y.shape[0]) / (np.array(y_true_sparsity_mat) / float(pred_y.shape[0]) + 1e-8), \
		# 					)
		# '''
		# observed_gt_bias1 = np.array(y_true_sparsity_mat) / float(pred_y.shape[0]) / (np.array(x_sparsity_mat) / float(pred_y.shape[0]) + 1e-8)
		# x_mask_ratio_mat, y_mask_ratio_mat = torch.stack(x_mask_ratio_mat), torch.stack(y_mask_ratio_mat)
		# '''
		# observed_gt_bias3 = np.maximum( \
		# 						np.array(y_mask_ratio_mat.cpu()) / (np.array(x_mask_ratio_mat.cpu())  + 1e-8), \
		# 						np.array(x_mask_ratio_mat.cpu()) / (np.array(y_mask_ratio_mat.cpu())  + 1e-8), \
		# 					)
		# '''
		# observed_gt_bias3 = np.array(y_mask_ratio_mat.cpu()) / (np.array(x_mask_ratio_mat.cpu())  + 1e-8)
		# observed_gt_bias3[observed_gt_bias3>1e3] = 0
		# '''
		# plt.plot(dim, np.array(x_sparsity_mat) / float(pred_y.shape[0]), alpha=0.6, label='observed')
		# # plt.plot(dim, y_pred_sparsity_mat, alpha=0.6, label='pred_y')
		# plt.plot(dim, np.array(x_sparsity_mat) / float(pred_y.shape[0]), alpha=0.6, label='gt')
		# '''
		# plt.plot(dim, observed_gt_bias1, alpha=0.6, label='bias1')
		# plt.plot(dim, observed_gt_bias3, alpha=0.6, label='bias3')
		# plt.legend(loc='upper center')

		# ax2 = plt.twinx()
		# # ax2.plot(dim, total_results["featurewise_mse"].cpu(), alpha=0.6, color='r', label='mse')
		# ax2.plot(dim, total_results["featurewise_mae"].cpu() / (data_max + 1e-8), alpha=0.6, color='g', label='scaled_mae')
		
		# ax2.set_ylabel('Featurewise_metric', color='r')
		# ax2.tick_params(axis='y', labelcolor='r')
		# ax2.legend(loc='upper right')

		# plt.xlabel('Features')
		# plt.ylabel('Metric')
		# plt.title('Density Avg for All Eval Sample, Feature 0 ~ '+ str(feat) +', Epoch ' + str(itr))
		# plt.savefig('/home/yimianmatthew/Irregular-time-series-forecasting/baseline/t-PatchGNN/visualize/density_allTest_epoch'+str(itr)+'_y-x.png')
		# plt.close()

		## Visualization for Case 4
		## bias 1 & bias 3
		combined1 = np.vstack((np.array(bias_list), np.array(mae_list)))
		combined2 = np.vstack((np.array(bias2_list), np.array(mae2_list)))
		combined3 = np.vstack((np.array(bias3_list), np.array(mae3_list)))
		combined4 = np.vstack((np.array(bias4_list), np.array(mae4_list)))
		np.save(f'/home/yimianmatthew/Irregular-time-series-forecasting/baseline/t-PatchGNN/visualize/data_storage/{method}/{dataset}/density-mae-bias1-best.npy',combined1)
		np.save(f'/home/yimianmatthew/Irregular-time-series-forecasting/baseline/t-PatchGNN/visualize/data_storage/{method}/{dataset}/density-mae-bias2-best.npy',combined2)
		np.save(f'/home/yimianmatthew/Irregular-time-series-forecasting/baseline/t-PatchGNN/visualize/data_storage/{method}/{dataset}/density-mae-bias3-best.npy',combined3)
		np.save(f'/home/yimianmatthew/Irregular-time-series-forecasting/baseline/t-PatchGNN/visualize/data_storage/{method}/{dataset}/density-mae-bias4-best.npy',combined4)
		
		'''
		parameter1 = np.polyfit(bias_list, mae_list, 1)
		maeb1 =  parameter1[0] * np.array(bias_list) + parameter1[1]

		parameter2 = np.polyfit(bias2_list, mae2_list, 1)
		maeb2 =  parameter2[0] * np.array(bias2_list) + parameter2[1]

		parameter3 = np.polyfit(bias3_list, mae3_list, 1)
		maeb3 =  parameter3[0] * np.array(bias3_list) + parameter3[1]

		parameter4 = np.polyfit(bias4_list, mae4_list, 1)
		maeb4 =  parameter4[0] * np.array(bias4_list) + parameter4[1]
		# plt.scatter(bias_list, [mae * 100 for mae in mae_list], alpha=0.6, label='mae')
		

		if(logx):
			plt.plot(bias_list, [mae * 100 for mae in maeb1], alpha=0.6, color = 'brown',label=f'log(Dy/Dx), k={parameter1[0]:.4f}, b={parameter1[1]:.4f}')
			plt.plot(bias2_list, [mae * 100 for mae in maeb2], alpha=0.6, color = 'green',label=f'log(Ty/Tx), k={parameter2[0]:.4f}, b={parameter2[1]:.4f}')
			plt.plot(bias3_list, [mae * 100 for mae in maeb3], alpha=0.6, color = 'orange',label=f'log(My/Mx), k={parameter3[0]:.4f}, b={parameter3[1]:.4f}')
			plt.plot(bias4_list, [mae * 100 for mae in maeb4], alpha=0.6, color = 'violet',label=f"log(Dy'/Dx'), k={parameter4[0]:.4f}, b={parameter4[1]:.4f}")
		else:
			plt.plot(bias_list, [mae * 100 for mae in maeb1], alpha=0.6, color = 'brown',label=f'Dy/Dx, k={parameter1[0]:.4f}, b={parameter1[1]:.4f}')
			plt.plot(bias2_list, [mae * 100 for mae in maeb2], alpha=0.6, color = 'green',label=f'Ty/Tx, k={parameter2[0]:.4f}, b={parameter2[1]:.4f}')
			plt.plot(bias3_list, [mae * 100 for mae in maeb3], alpha=0.6, color = 'orange',label=f'My/Mx, k={parameter3[0]:.4f}, b={parameter3[1]:.4f}')
			plt.plot(bias4_list, [mae * 100 for mae in maeb4], alpha=0.6, color = 'violet',label=f"Dy'/Dx', k={parameter4[0]:.4f}, b={parameter4[1]:.4f}")
		plt.xlabel('Density Bias (Dy / Dx)')
		plt.ylabel('MAE(×10^-2)')
		plt.title('Bias-MAE scatter for all test data')
		plt.legend()
		if(logx):
			plt.savefig(f'/home/yimianmatthew/Irregular-time-series-forecasting/baseline/t-PatchGNN/visualize/{method}/{dataset}/density-mae-type1~4bias_old_allTest_epoch{str(itr)}_logx.png')
		else:
			plt.savefig(f'/home/yimianmatthew/Irregular-time-series-forecasting/baseline/t-PatchGNN/visualize/{method}/{dataset}/density-mae-type1~4bias_old_allTest_epoch{str(itr)}.png')
		plt.close()

		bias_list_list = [bias_list, bias2_list, bias3_list, bias4_list]
		mae_list_list = [[mae * 100 for mae in maeb1],[mae * 100 for mae in maeb2],[mae * 100 for mae in maeb3],[mae * 100 for mae in maeb4]]
		utils.plot_violin_by_xgroup(bias_list_list, mae_list_list, f'/home/yimianmatthew/Irregular-time-series-forecasting/baseline/t-PatchGNN/visualize/{method}/{dataset}/density-mae-type1~4_violin_old_allTest_epoch{str(itr)}_logx.png')
		## bias 2
		
		parameter = np.polyfit(bias3_list, mae_list, 1)
		mae2 =  parameter[0] * np.array(bias3_list) + parameter[1]
		plt.scatter(bias3_list, [mae * 100 for mae in mae_list], alpha=0.6, label='mae')
		plt.plot(bias3_list, [mae * 100 for mae in mae2], alpha=0.6, color = 'r',label=f'polyfit, k={parameter[0]:.4f}, b={parameter[1]:.4f}')
		plt.xlabel('Mask Ratio Bias (My / Mx)')
		plt.ylabel('MAE (×10^-2)')
		plt.title('Bias-MAE scatter for all test data(Type 3)')
		plt.legend()
		plt.savefig(f'/home/yimianmatthew/Irregular-time-series-forecasting/baseline/t-PatchGNN/visualize/graflti_density-mae-type3-bias_allTest_epoch{str(itr)}.png')
		plt.close()
		'''
		pass

	for key, var in total_results.items(): 
		if isinstance(var, torch.Tensor) and len(var.shape) == 0:
			var = var.item()
		total_results[key] = var

	return total_results

