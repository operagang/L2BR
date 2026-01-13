import argparse
from datetime import datetime

def argparse_common():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('-em', '--embed_dim', type=int, default=128, help='embedding size')
    parser.add_argument('-nh', '--n_heads',  type=int, default=8, help='number of heads in MHA')
    parser.add_argument('-ne', '--n_encode_layers', type=int, default=3, help='number of MHA encoder layers')
    parser.add_argument('-ff', '--ff_hidden_dim', type=int, default=512, help='ff_hidden dimension')
    return parser

def add_problem_params(parser: argparse.ArgumentParser):
    parser.add_argument('-pt', '--problem_type', type=str, default='uBRP', choices=['rBRP', 'uBRP'], help="Type of the problem: 'rBRP' or 'uBRP'")
    parser.add_argument('-bt', '--benchmark_type', type=str, default='CVS', choices=['CVS', 'ZQLZ', 'LL-R', 'LL-U'], help="Type of the benchmark: 'CVS', 'ZQLZ', or 'LL'")
    parser.add_argument('-t','--max_tiers',type=int,default=7, help="number of tiers T (Not H. Be carefull with CVS dataset where T = H + 2)")
    parser.add_argument('-s','--max_stacks' ,type=int ,default=5, help='number of stacks S')

def argparse_train_IL():
    parser = argparse_common()
    parser.add_argument('-pt', '--problem_type', type=str, default='uBRP', choices=['rBRP', 'uBRP'], help="Type of the problem: 'rBRP' or 'uBRP'")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-E', '--epoch', type=int, default=50, help = "epoch num")
    parser.add_argument('-b', '--batch', type=int, default = 256, help='batch size')
    parser.add_argument('-lp', '--log_path', type=str, default=f"./.results/{datetime.now().strftime('%Y%m%d_%H%M%S')}", help='log path')
    return parser.parse_args()

def argparse_train_SIL():
    parser = argparse_common()
    add_problem_params(parser)
    # parser.add_argument('-mp', '--model_path', type=str, default = "./IL/final_models/CVS-uBRP/H=5xS=5.pt", help="Pretrained model path")
    parser.add_argument('-E', '--epoch', type=int, default=10000, help = "epoch num")
    parser.add_argument('-b', '--batch', type=int, default=256, help='batch size')
    parser.add_argument('-pn', '--problem_num', type=int, default=1024, help = "number of problems")
    parser.add_argument('-sm', '--sampling_num', type=int, default=512, help = "number of sampling number (i.e. How many time solving same problem)")
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('-vs', '--valid_sampling', type=int, default=2560, help = "number of validation sampling number")
    parser.add_argument('-lp', '--log_path', type=str, default=f"./.results/{datetime.now().strftime('%Y%m%d_%H%M%S')}", help='log path')
    return parser.parse_args()

def argparse_test():
    parser = argparse_common()
    add_problem_params(parser)
    parser.add_argument('-mp', '--model_path', type=str, default = "./SIL/final_models/CVS-rBRP/H=5xS=6.pt", help="Model path")
    parser.add_argument('-T', '--temp', type=float, default=1, help='Temperature for Softmax')
    parser.add_argument('-N', '--sampling_num', type=int, default=2560, help='Total sampling number')
    return parser.parse_args()

def argparse_train_RL():
    parser = argparse_common()
    add_problem_params(parser)
    parser.add_argument('-mp', '--model_path', type=str, default = "./IL/final_models/CVS-uBRP/H=5xS=5.pt", help="Pretrained model path")
    parser.add_argument('-E', '--epoch', type=int, default=10000, help = "epoch num")
    parser.add_argument('-b', '--batch', type=int, default = 512, help='batch size')
    parser.add_argument('-mb', '--mini_batch', type=int, default = 4, help='mini batch size')
    parser.add_argument('-sm', '--sampling_num', type=int, default=16, help = "POMO")
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('-vs', '--valid_sampling', type=int, default=2560, help = "number of validation sampling number")
    parser.add_argument('-lp', '--log_path', type=str, default=f"./.results/{datetime.now().strftime('%Y%m%d_%H%M%S')}", help='log path')
    return parser.parse_args()