import torch
from params import argparse_test
from ess import ESS
from utils import reset_seed, load_model


def test(args):
    reset_seed(args.seed)
    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'

    model = load_model(args, device)
    model=model.to(device)
    model.eval()

    ess = ESS(device)

    temp = args.temp
    score = ess.validation(args, model, args.sampling_num, temp=temp)
    print(f"{args.problem_type}|{args.benchmark_type}|T={args.max_tiers}|S={args.max_stacks} Test Score: {score:.3f}")




if __name__ == '__main__':
    args = argparse_test()

    test(args)
