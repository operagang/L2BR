import gc
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from params import argparse_train_RL
from ess import ESS
from utils import reset_seed, load_model, create_log_directory, save_args_txt, add_log_txt
from instance_generator import generate_cvs_data, generate_zqlz_data, generate_ll_data
from env import Env



class RL_Sampler(nn.Module):
    def __init__(self, T=1.0, n_samples=1):
        super().__init__()
        self.T = T
        self.n_samples = n_samples

    def forward(self, logits):
        new_logits = logits / self.T

        log_probs_all = F.log_softmax(new_logits, dim=1)
        probs_all     = log_probs_all.exp()

        actions = torch.multinomial(probs_all, self.n_samples, replacement=True)

        if self.n_samples == 1:
            actions = actions.squeeze(1)
            chosen_log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        else:
            chosen_log_probs = log_probs_all.gather(1, actions)
        return actions, chosen_log_probs





def train(args):
    if args.problem_type != 'uBRP':
        raise ValueError("train_RL.py only supports 'uBRP' problem type.")

    reset_seed(args.seed)
    create_log_directory(args)
    save_args_txt(args)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'

    model = load_model(args, device)
    model=model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    selecter  = RL_Sampler(T=1.0, n_samples=1).to(device)
    ess = ESS(device)

    acc_steps = args.batch // args.mini_batch
    assert args.batch % args.mini_batch == 0

    t1 = time()

    for epoch in range(args.epoch):
        model.train()

        optimizer.zero_grad()

        for step_idx in tqdm(range(acc_steps), desc=f"Epoch {epoch+1}/{args.epoch}"):

            gc.collect()

            if args.benchmark_type == 'CVS':
                problems = generate_cvs_data(args, device, n_samples=args.mini_batch)
                traj_limit = 500
            elif args.benchmark_type == 'ZQLZ':
                problems = generate_zqlz_data(args, device, n_samples=args.mini_batch)
                traj_limit = 500
            elif args.benchmark_type in ['LL-R', 'LL-U']:
                problems = generate_ll_data(args, device, n_samples=args.mini_batch)
                traj_limit = 2000

            x0 = problems.repeat(args.sampling_num, 1, 1)
            env = Env(device=device, x=x0)
            env.clear()

            B = x0.size(0)  # = mini_batch * n_samplings

            log_prob_sums = torch.zeros(B, device=device)
            rewards       = torch.full((B,), fill_value=traj_limit, device=device, dtype=torch.float)
            done          = torch.zeros(B, dtype=torch.bool, device=device)

            for t in range(traj_limit):
                if done.all():
                    break

                output = model(env.x)
                actions_idx, log_prob = selecter(output)

                alive = ~done
                log_prob_sums[alive] += log_prob[alive]

                source_node = (actions_idx // args.max_stacks).unsqueeze(1)
                dest_node   = (actions_idx %  args.max_stacks).unsqueeze(1)
                actions     = torch.cat((source_node, dest_node), dim=1)

                env.step_uBRP(actions)

                newly_done = alive & env.empty
                rewards[newly_done] = float(t + 1)
                done |= newly_done

            rewards_2d = rewards.view(args.sampling_num, args.mini_batch)
            baseline_per_instance = rewards_2d.mean(dim=0)
            baseline_full = baseline_per_instance.repeat(args.sampling_num)

            advantages = rewards - baseline_full
            advantages = -advantages

            loss_mb = -(advantages.detach() * log_prob_sums).mean()

            loss_mb_div = loss_mb / acc_steps
            loss_mb_div.backward()

            if epoch == 0:
                torch.cuda.empty_cache()

        optimizer.step()

        torch.cuda.empty_cache()

        model.eval()
        torch.save(model.state_dict(), args.log_path + '/epoch%s.pt' % epoch)
        validation_score = ess.validation(args, model, args.valid_sampling)
        add_log_txt(args, epoch, validation_score, t1)

        torch.cuda.empty_cache()
    






if __name__ == '__main__':
    args = argparse_train_RL()

    train(args)

    pass