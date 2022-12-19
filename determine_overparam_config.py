from argparse import ArgumentParser
from ecg_datamodule import ECGDataModule
import math
import numpy as np
import os
import scipy.linalg
import time
import torch
import yaml

import matplotlib.pyplot as plt

def compute_neural_network_covariance_matrix(X, w_distro='uniform'):
    """NN covariance matrix under the assumption that the nonlinearity is RELU."""
    _, d = X.shape
    XX_T = X @ X.T
    num_samples, all_results = 100, []
    import ipdb; ipdb.set_trace()
    for _ in range(num_samples):
        if w_distro == 'uniform': # default torch initialization
            stdv = 1. / math.sqrt(d)
            w = torch.rand((d, 1)) * (2*stdv) - stdv
            # import ipdb; ipdb.set_trace()
        else: # Normal, which is what the proof from the paper assumes.
            w = torch.normal(torch.zeros((d, 1)), torch.ones((d,1)))
        prod = X @ w 
        prod[prod <= 0] = 0
        prod[prod > 0] = 1
        res = torch.mul(prod @ prod.T, XX_T)
        all_results.append(res)
    import ipdb; ipdb.set_trace()
    exp_cov = torch.mean(torch.stack(all_results), axis=0)
    eigs = np.linalg.eigvals(exp_cov)
    Xnorm = np.linalg.norm(X, ord=2)
    return exp_cov, eigs, Xnorm

def compute_LR_and_hiddennum(X, Y, Xnorm, cov_eigs, C, target_prob=0.9, extra_factor=1, mu_bar=0.8):
    num_samples, d = X.shape
    import ipdb; ipdb.set_trace()
    delta = np.sqrt(-1 * np.log(1 - target_prob - (2 / num_samples)) * (Xnorm**2 / num_samples))
    LR = (num_samples * mu_bar) / (3 * np.linalg.norm(Y, ord='fro')**2 * Xnorm**2)
    k = C * (1 + delta)**2 * (num_samples * Xnorm**6) / ((cov_eigs[cov_eigs > 0][-1])**4)
    return LR, k * extra_factor

def compute_theorem2pt3(X, Y, C, mu_bar=0.8, target_prob=0.9, extra_factor=1):
    n, d = X.shape
    import ipdb; ipdb.set_trace()
    Xnorm = np.linalg.norm(X, ord=2)
    delta = np.sqrt(-1 * (Xnorm**2 / n) * np.log(1 - target_prob - (1/n) - n * np.exp(-n)))
    X_kr_X = scipy.linalg.khatri_rao(X.T.cpu().numpy(), X.T.cpu().numpy())
    import ipdb; ipdb.set_trace()
    _, S, _ = np.linalg.svd(X_kr_X, full_matrices=False)
    S = torch.from_numpy(S)
    kappa_x = (np.sqrt(d/n) * Xnorm) / (S[-1]**2)
    k = np.power(C * (1 + delta) * (n**2/d) * kappa**3 * S[-1]**2, 2) / d
    LR = mu_bar * n / (3 * np.linalg.norm(Y)**2 * Xnorm**2)
    return k * extra_factor, LR, kappa_x, S

def compute_basic_LR(X, Y):
    n, _ = X.shape
    return n / (np.linalg.norm(Y)**2 * np.linalg.norm(X, ord=2)**2)

def parse_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--config_name', default='bolts_config.yaml')
    parser.add_argument('-t', '--trafos', nargs='+', help='add transformation to data augmentation pipeline',
        default=["GaussianNoise", "ChannelResize", "RandomResizedCrop", "Normalize"])
    # RandomResizedCrop
    parser.add_argument('--rr_crop_ratio_range',
        help='ratio range for random resized crop transformation', default=[0.5, 1.0], type=float)
    parser.add_argument('--output_size',
        help='output size for random resized crop transformation', default=250, type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--datasets', dest="target_folders",
                        nargs='+', help='used datasets for pretraining')
    parser.add_argument('--log_dir', default="./overparam_configs")
    parser.add_argument('--percentage', help='determines how much of the dataset shall be used during the pretraining', type=float, default=1.0)
    parser.add_argument('--out_dim', type=int, help="output dimension of model")
    return parser

def pretrain_routine(args):
    config = yaml.load(open(args.config_name, "r"), Loader=yaml.FullLoader)
    args_dict = vars(args)
    for key in set(config.keys()).union(set(args_dict.keys())):
        config[key] = config[key] if (key not in args_dict.keys() or key in args_dict.keys(
        ) and key in config.keys() and args_dict[key] is None) else args_dict[key]
    
    if args.target_folders is not None: config["dataset"]["target_folders"] = args.target_folders
    config["dataset"]["percentage"] = args.percentage if args.percentage is not None else config["dataset"]["percentage"]
    
    if args.out_dim is not None: config["model"]["out_dim"] = args.out_dim
    
    date = time.asctime()
    label_to_num_classes = {"label_all": 71, "label_diag": 44, "label_form": 19,
        "label_rhythm": 12, "label_diag_subclass": 23, "label_diag_superclass": 5}
    ptb_num_classes = label_to_num_classes[config["eval_dataset"]["ptb_xl_label"]]
    name = str(date) + "_linearconfig_" + str(time.time_ns())[-3:]
    config["log_dir"] = os.path.join(args.log_dir, name)
    print(config)
    return config, date, ptb_num_classes

def main():
    parser = ArgumentParser()
    parser = parse_args(parser)
    args = parser.parse_args()
    config, date, ptb_num_classes  = pretrain_routine(args)

    # data
    transformations = [] # ["Normalize"]
    ecg_datamodule = ECGDataModule(config, transformations, None, ptb_num_classes=ptb_num_classes)
    train_dataloader = ecg_datamodule.train_dataloader()
    valid_loader_self, _, _ = ecg_datamodule.val_dataloader()

    # import ipdb; ipdb.set_trace()

    all_val_labels = []
    for elem in valid_loader_self:
        all_val_labels.append(elem[0][1])
    all_val_labels = torch.cat(all_val_labels, axis=0)
    Y = torch.argmax(all_val_labels, axis=1, keepdim=True)

    # import ipdb; ipdb.set_trace()

    all_data = []
    # for elem in train_dataloader:
    for elem in valid_loader_self:
        all_data.append(elem[0][0])
    X = torch.cat(all_data, axis=0)
    X = torch.flatten(X, start_dim=1)
    inds_to_keep = torch.sum(X, axis=1) != 0
    X = X[inds_to_keep, ...]

    samples_to_use = min(X.shape[0], Y.shape[0])
    X = X[np.random.choice(X.shape[0], samples_to_use, replace=False), :]
    Y = Y[np.random.choice(Y.shape[0], samples_to_use, replace=False), :]

    C = 1

    import ipdb; ipdb.set_trace()

    exp_cov, eigs, Xnorm = compute_neural_network_covariance_matrix(
        X, w_distro='gaussian') # uniform | gaussian

    import ipdb; ipdb.set_trace()

    LR, k = compute_LR_and_hiddennum(X, Y, Xnorm, eigs, C, target_prob=0.9,
        extra_factor=1, mu_bar=0.8)

    # import ipdb; ipdb.set_trace()

    # Right now we run out of memory when we do the Khatri-Rao product.
    # k, LR, kappa_x, S = compute_theorem2pt3(X, Y, C, mu_bar=0.8, target_prob=0.9, extra_factor=1)

    import ipdb; ipdb.set_trace()

    LR = compute_basic_LR(X, Y)

    # TO MAYBE DO?
    ## Change initialization from uniform to std. normal?
    ### https://discuss.pytorch.org/t/why-the-default-negative-slope-for-kaiming-uniform-initialization-of-convolution-and-linear-layers-is-5/29290
    ### https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/2
    ### https://wandb.ai/wandb_fc/tips/reports/How-to-Initialize-Weights-in-PyTorch--VmlldzoxNjcwOTg1
    ### https://discuss.pytorch.org/t/clarity-on-default-initialization-in-pytorch/84696/2
    ### https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch

    ## Set the output v weights to be +/- 1 randomly selected by coinflip.

if __name__ == "__main__":  
    main()
