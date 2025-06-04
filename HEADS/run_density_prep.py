import os
import sys
import argparse
import numpy as np
sys.path.append("..")

from lib.parse_density import parse_density

parser = argparse.ArgumentParser('TOD Preprocessing')
parser.add_argument('--alpha', type=float, default=2., help="Hyperparameter for Renyi entropy")
parser.add_argument('--dataset', type=str, default='physionet', help="Dataset to load. Available: physionet, mimic, ushcn, activity")
parser.add_argument('--batch_size',  type=int, default=32, help="Size of batch")
parser.add_argument('--n',  type=int, default=12000, help="Size of the dataset")
parser.add_argument('--history', type=int, default=24, help="number of hours (months for ushcn and ms for activity) as historical window")
parser.add_argument('--pred_window', type=int, default=24, help="number of hours (months for ushcn and ms for activity) as forecasting window")
parser.add_argument('--gpu', type=str, default='0', help='which gpu to use.')
parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
parser.add_argument('--quantization', type=float, default=0.0, help="Quantization on the physionet dataset.")
parser.add_argument('--patch_ts', action='store_true', help="Apply patching method(for t-PatchGNN)")
parser.add_argument('-ps', '--patch_size', type=float, default=24, help="window size for a patch")
parser.add_argument('--stride', type=float, default=24, help="period stride for patch sliding")
args = parser.parse_args()

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.npatch = int(np.ceil((args.history - args.patch_size) / args.stride)) + 1

    if(args.dataset in ["physionet", 'mimic'] and args.history!=24): # Sensitivity Study
        parse_density(args, patch_ts=args.patch_ts, length_stat=False, 
            save_path=f"../data/{args.dataset}/density_processed/data_objects_n{args.n}_patch{args.patch_ts}_alpha{args.alpha}_his{args.history}.pkl", alpha=args.alpha)
    else:
        if(args.alpha!=2.0): # Hyperparameter Study
            parse_density(args, patch_ts=args.patch_ts, length_stat=False, 
                save_path=f"../data/{args.dataset}/density_processed/data_objects_n{args.n}_patch{args.patch_ts}_alpha{args.alpha}.pkl", alpha=args.alpha)
        else: # Original Setting
            parse_density(args, patch_ts=args.patch_ts, length_stat=False, 
                save_path=f"../data/{args.dataset}/density_processed/data_objects_n{args.n}_patch{args.patch_ts}.pkl")

