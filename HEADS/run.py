import os
import sys
sys.path.append("..")

import time
import datetime
import argparse
import numpy as np
import pandas as pd
import random
from random import SystemRandom
from sklearn import model_selection

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import lib.utils as utils
from lib.parse_datasets import parse_datasets

from model.mtan import *
from model.tPatchGNN import *
from model.gratif import *
from lib.evaluation import *

from HEADS.model.heads import *


parser = argparse.ArgumentParser('IMTS Forecasting')
# Plugin-related parameters
parser.add_argument('--plugin', action='store_true', help="Apply plugin method")
parser.add_argument('--alpha', type=float, default=2., help="Hyperparameter for Renyi entropy")
parser.add_argument('--plugin_dim', type=int, default=32, help="plugin dimension")
parser.add_argument('--time_dim', type=int, default=10, help="Time embed dimension")
parser.add_argument('--dropout', type=float, default=0.2, help="Dropout ratio in training")
parser.add_argument('--attempt', type=int, default=0, help="Attempt number, 0 for original model")
parser.add_argument('--lamb', type=float, default=0.5, help="Hyperparameter for combination")

# Forecast training related parameters
parser.add_argument('--state', type=str, default='def')
parser.add_argument('--n',  type=int, default=12000, help="Size of the dataset")
parser.add_argument('--epoch', type=int, default=100, help="training epoches")
parser.add_argument('--patience', type=int, default=10, help="patience for early stop")
parser.add_argument('--history', type=int, default=24, help="number of hours (months for ushcn and ms for activity) as historical window")
parser.add_argument('-ps', '--patch_size', type=float, default=24, help="window size for a patch")
parser.add_argument('--stride', type=float, default=24, help="period stride for patch sliding")
parser.add_argument('--logmode', type=str, default="a", help='File mode of logging.')

parser.add_argument('--lr',  type=float, default=1e-3, help="Starting learning rate.")
parser.add_argument('--w_decay', type=float, default=0.0, help="weight decay.")
parser.add_argument('-b', '--batch_size', type=int, default=32)

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('--seed', type=int, default=1, help="Random seed")
parser.add_argument('--dataset', type=str, default='physionet', help="Dataset to load. Available: physionet, mimic, ushcn")

parser.add_argument('--quantization', type=float, default=0.0, help="Quantization on the physionet dataset.")
parser.add_argument('--model', type=str, default='tPatchGNN', help="Model name")
parser.add_argument('--outlayer', type=str, default='Linear', help="Out layer name")
parser.add_argument('-hd', '--hid_dim', type=int, default=64, help="Number of units per hidden layer")
parser.add_argument('-td', '--te_dim', type=int, default=10, help="Number of units for time encoding")
parser.add_argument('-nd', '--node_dim', type=int, default=10, help="Number of units for node vectors")
parser.add_argument('--gpu', type=str, default='0', help='which gpu to use.')

# Parameters for tPatchGNN
parser.add_argument('--hop', type=int, default=1, help="hops in GNN")
parser.add_argument('--nhead', type=int, default=8, help="heads in Transformer")
parser.add_argument('--tf_layer', type=int, default=1, help="# of layer in Transformer")
parser.add_argument('--nlayer', type=int, default=1, help="# of layer in TSmodel")

# Parameters for GraFITi
parser.add_argument("-nl",  "--nlayers", default=4,   type=int,   help="")
parser.add_argument("-ahd",  "--attn-head", default=1,   type=int,   help="")
parser.add_argument("-ldim", default=128,   type=int,   help="")

# Parameters for mtan
parser.add_argument('--latent-dim', type=int, default=16)
parser.add_argument('--rec-hidden', type=int, default=64)
parser.add_argument('--gen-hidden', type=int, default=50)
parser.add_argument('--embed-time', type=int, default=128)
parser.add_argument('--k-iwae', type=int, default=1)
parser.add_argument('--learn-emb', action='store_true')
parser.add_argument('--enc-num-heads', type=int, default=1)
parser.add_argument('--dec-num-heads', type=int, default=1)
parser.add_argument('--num-ref-points', type=int, default=128)

args = parser.parse_args()

args.npatch = int(np.ceil((args.history - args.patch_size) / args.stride)) + 1 # (window size for a patch) M

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
file_name = os.path.basename(__file__)[:-3]
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.PID = os.getpid()
print("PID, device:", args.PID, args.device)

#####################################################################################################

if __name__ == '__main__':
    mses, rmses, maes, mapes = [], [], [], []
    utils.seed_torch(0)
    
    experimentID = args.load
    if experimentID is None:
        # Make a new experiment ID
        experimentID = int(SystemRandom().random()*100000)
    ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '.ckpt')
    
    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind+2):]
    input_command = " ".join(input_command)
    
    if not os.path.exists("logs/"):
        utils.makedirs("logs/")
    log_dir = "logs/{}".format(args.model)
    if not os.path.exists(log_dir):
        utils.makedirs(log_dir)

    if(args.plugin == False):
        log_path = "logs/{}/{}_{}_final_original.log".format(args.model,args.dataset, args.n)
    else:
        log_path = "logs/{}/{}_{}_final_HEADS.log".format(args.model,args.dataset, args.n)
        
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__), mode=args.logmode)
    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(input_command)
    logger.info(args)

    st_time = time.time()
    data_obj = parse_datasets(args, patch_ts=(args.model=='tPatchGNN')) # Only t-PatchGNN needs patching
    input_dim = data_obj["input_dim"]
    batch_size = args.batch_size

    num_batches = data_obj["n_train_batches"] # n_sample / batch_size
    print("n_train_batches:", num_batches)
    end_time = time.time()
    logger.info("Preprocessing time: {:.2f}".format(end_time - st_time))
    for seed in range(args.seed, args.seed+5):
        utils.seed_torch(seed)

        logger.info(f"Running seed {seed}...")
        # utils.makedirs("results/")

        ##################################################################

        ### Model setting ###\
        args.ndim = input_dim
        if(args.model=='GraFITi'):
            model = GrATiF(args, input_dim).to(args.device)
        elif(args.model=='tPatchGNN'):
            model = tPatchGNN(args).to(args.device)
        elif(args.model=='mTAN'):
            model = mTAN_Model(args, enc_mtan_rnn, dec_mtan_rnn, input_dim, args.device).to(args.device)
        # model = IrMLP(args).to(args.device)
        ##################################################################

        # # Load checkpoint and evaluate the model
        # if args.load is not None:
        # 	utils.get_ckpt_model(ckpt_path, model, args.device)
        # 	exit()

        ##################################################################
        if(args.plugin):
            model = DensityAwareModel(model, input_dim, dropout=args.dropout, hid_dim=args.hid_dim, lamb=args.lamb).to(args.device)
                

        optimizer = optim.Adam(model.parameters(), lr=args.lr)


        best_val_mse = np.inf
        test_res = None

        for itr in range(args.epoch):
            st = time.time()
            ### Training ###
            model.train()
            
            x_list = np.array([0.0] * 41)
            y_list = np.array([0.0] * 41)
            mse_list = np.array([0.0] * 41)
            
            for _ in range(num_batches):
                optimizer.zero_grad()
                # t1 = time.time()
                batch_dict = utils.get_next_batch(data_obj["train_dataloader"])

                train_res = compute_all_losses(model, batch_dict, itr, args.plugin)

                train_res["loss"].backward()
                optimizer.step()

            ### Validation ###
            model.eval()
            with torch.no_grad():
                val_res = evaluation(model, data_obj["val_dataloader"], data_obj["n_val_batches"], args.plugin)
                ### Testing ###
                if(val_res["mse"] < best_val_mse):
                    best_val_mse = val_res["mse"]
                    best_iter = itr
                    test_res = evaluation(model, data_obj["test_dataloader"], data_obj["n_test_batches"], args.plugin)

                logger.info('- Epoch {:03d}, ExpID {}'.format(itr, experimentID))
                logger.info("Train - Loss (one batch): {:.5f}".format(train_res["loss"].item()))
                logger.info("Val - Loss, MSE, RMSE, MAE, MAPE: {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%" \
                    .format(val_res["loss"], val_res["mse"], val_res["rmse"], val_res["mae"], val_res["mape"]*100))
                if(test_res != None):
                    logger.info("Test - Best epoch, Loss, MSE, RMSE, MAE, MAPE: {}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.2f}%" \
                        .format(best_iter, test_res["loss"], test_res["mse"],\
                        test_res["rmse"], test_res["mae"], test_res["mape"]*100))
                logger.info("Time spent: {:.2f}s".format(time.time()-st))
                
            if(itr - best_iter >= args.patience):
                logger.info("Exp has been early stopped!")
                mses.append(test_res["mse"])
                rmses.append(test_res["rmse"])
                maes.append(test_res["mae"])
                mapes.append(test_res["mape"])
                break

    if(args.dataset=='physionet' or args.dataset=='activity'):
        scalar1, scalar2=1000, 100
    elif(args.dataset=='mimic'):
        scalar1, scalar2=100, 100
    elif(args.dataset=='ushcn'):
        scalar1, scalar2=10, 10
    else:
        raise NotImplementedError
    
    logger.info(f"MSE: {np.mean(mses) * scalar1:.2f} ± {np.std(mses) * scalar1:.2f}")
    logger.info(f"RMSE: {np.mean(rmses) * (scalar1/10):.2f} ± {np.std(rmses) * (scalar1/10):.2f}")
    logger.info(f"MAE: {np.mean(maes) * scalar2:.2f} ± {np.std(maes) * scalar2:.2f}")
    logger.info(f"MAPE: {np.mean(mapes) * 100:.1f} ± {np.std(mapes) * 100:.1f}")
