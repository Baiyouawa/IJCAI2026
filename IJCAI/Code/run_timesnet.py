import torch
import numpy as np
import timesnet_train
import argparse

parser = argparse.ArgumentParser(description='TimesNet')

parser.add_argument('--device', type=str, default="cuda", help='device setting cuda or cpu')
parser.add_argument('--batch', type=int, default=16, help='input batch size')
parser.add_argument('--dataset', type=str, default="kdd", help='data set name')
parser.add_argument('--missing_rate', type=float, default=0.1, help='missing percent for experiment')
parser.add_argument('--seed', type=int, default=3407, help='random seed')

# input data enc_in c_out setting: kdd:99 guangzhou:214 physio:37
parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')
parser.add_argument('--enc_in', type=int, default=99, help='encoder input size')
parser.add_argument('--c_out', type=int, default=99, help='decoder output size')

parser.add_argument('--epoch_diff', type=int, default=10, help='training epoch for diffusion training')
parser.add_argument('--learning_rate_diff', type=float, default=1e-3, help='learning rate of diffusion training')

parser.add_argument('--task_name', type=str, default='imputation', help='task name')
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

# [新增] TimeNet 需要这些 (即使在插补中不使用)，否则 TimeNet.Model 会报错
parser.add_argument('--pred_len', type=int, default=0, help='prediction sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--num_class', type=int, default=0, help='for classification')

if __name__ == '__main__':
    configs = parser.parse_args()

    np.random.seed(configs.seed)
    torch.manual_seed(configs.seed)
    torch.cuda.manual_seed(configs.seed)

    model = timesnet_train.diffusion_train(configs)
    print("TEST")
    timesnet_train.diffusion_test(configs, model)
    torch.cuda.empty_cache()