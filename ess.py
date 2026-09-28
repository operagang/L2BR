import torch
import gc
from tqdm import tqdm
from instance_generator import generate_cvs_data, generate_zqlz_data, generate_ll_data
from sampler import TopKSampler,CategoricalSampler, New_Sampler
from env import Env
from benchmark_loader import data_CVS, data_ZQLZ, data_LL

import time

class ESS:
    def __init__(self, device):
        self.device = device

    def generate_SIL_data(self, model, args, device):
        model.eval()
        if model is None:
            raise NotImplementedError
        if args.benchmark_type == 'CVS':
            problems = generate_cvs_data(args, device)
            traj_limit = 500
        elif args.benchmark_type == 'ZQLZ':
            problems = generate_zqlz_data(args, device)
            traj_limit = 500
        elif args.benchmark_type in ['LL-R', 'LL-U']:
            problems = generate_ll_data(args, device)
            traj_limit = 2000

        train_data = []
        label_data = []
        selecter = New_Sampler(T=1)
        sampling_batch = args.sampling_num

        for i in tqdm(range(len(problems)), desc='Sampling Self-Improvement Learning Trajectories'):
            # gc.collect()
            env = Env(device = device, x = problems[i:i+1].repeat(sampling_batch,1,1))
            env.clear()
            trajectory = []
            traj_actions = []
            for step in range(traj_limit):
                trajectory.append(env.x.clone())
                if env.empty.any().item():
                    index = env.empty.long().nonzero(as_tuple=False)[0]
                    best_traj = torch.cat(trajectory)[torch.arange(index.item(), index.item()+sampling_batch*(step), sampling_batch),:,:].clone()
                    best_action = torch.cat(traj_actions)[torch.arange(index.item(), index.item()+sampling_batch*(step), sampling_batch)].clone()
                    train_data.append(best_traj)
                    label_data.append(best_action)
                    # print('Problem %d: Finished at step %d (%d)' % (i, step, len(best_traj)))
                    break
                output = model(env.x)
                if args.problem_type == 'rBRP':
                    next_action = selecter(output).view(-1,1)
                    env.step_rBRP(next_action)
                elif args.problem_type == 'uBRP':
                    next_action = selecter(output)
                    source_node, dest_node = next_action//args.max_stacks, next_action%args.max_stacks
                    actions = torch.cat((source_node,dest_node), 1)
                    env.step_uBRP(actions)
                traj_actions.append(next_action.view(-1).clone())
            del trajectory[:]
        train_data = torch.cat(train_data).to(device)
        label_data = torch.cat(label_data).to(device)
        return train_data, label_data




    def generate_SIL_data4(self, model, args, device):
        model.eval()
        if model is None:
            raise NotImplementedError

        if args.benchmark_type == 'CVS':
            problems = generate_cvs_data(args, device); traj_limit = 500
        elif args.benchmark_type == 'ZQLZ':
            problems = generate_zqlz_data(args, device); traj_limit = 500
        elif args.benchmark_type in ['LL-R', 'LL-U']:
            problems = generate_ll_data(args, device); traj_limit = 2000

        selecter = New_Sampler(T=1)
        B = args.sampling_num
        Pchunk = 16

        train_data_list = []
        label_data_list = []

        num_probs = len(problems)

        for base in tqdm(range(0, num_probs, Pchunk),
                        desc=f"Sampling SIL Trajectories (single Env, chunk={Pchunk})"):

            end = min(base + Pchunk, num_probs)
            P = end - base

            x0 = problems[base:end].unsqueeze(1).repeat(1, B, 1, 1).reshape(P * B, *problems.shape[1:])
            env = Env(device=device, x=x0)
            env.clear()

            trajectory = []
            traj_actions = []

            finished = torch.zeros(P, dtype=torch.bool, device='cpu')

            for step in range(traj_limit):
                trajectory.append(env.x.clone())

                empty = env.empty.view(P, B)

                newly = (~finished) & empty.any(dim=1).detach().cpu()
                if newly.any():
                    new_ps = newly.nonzero(as_tuple=False).view(-1)
                    for p in new_ps.tolist():
                        idx = empty[p].long().nonzero(as_tuple=False)[0].item()
                        finished[p] = True

                        if step == 0:
                            continue

                        traj_cat = torch.cat(trajectory, dim=0)
                        act_cat  = torch.cat(traj_actions, dim=0)

                        base_idx = p * B + idx

                        pick = torch.arange(base_idx, base_idx + (P * B) * step, P * B, device=traj_cat.device)

                        best_traj = traj_cat[pick, :, :].clone()
                        best_act  = act_cat[pick].clone()

                        train_data_list.append(best_traj)
                        label_data_list.append(best_act)

                    if finished.all():
                        break

                output = model(env.x)

                if args.problem_type == 'rBRP':
                    next_action = selecter(output).view(-1, 1)
                    env.step_rBRP(next_action)

                elif args.problem_type == 'uBRP':
                    next_action = selecter(output)
                    source_node, dest_node = next_action // args.max_stacks, next_action % args.max_stacks
                    actions = torch.cat((source_node, dest_node), 1)
                    env.step_uBRP(actions)

                traj_actions.append(next_action.view(-1).clone())

            del trajectory[:]
            del traj_actions[:]

        train_data = torch.cat(train_data_list, dim=0).to(device)
        label_data = torch.cat(label_data_list, dim=0).to(device)
        return train_data, label_data


    def sample_valid_action_from_invalid_mask(self, mask_invalid: torch.Tensor) -> torch.Tensor:
        if mask_invalid.dim() == 3:
            mask_invalid = mask_invalid.squeeze(-1)
        assert mask_invalid.dtype == torch.bool

        valid = ~mask_invalid
        a = valid.float().argmax(dim=1).view(-1, 1).long()
        return a


    def generate_SIL_data5(self, model, args, device):
        model.eval()
        if model is None:
            raise NotImplementedError

        if args.benchmark_type == 'CVS':
            problems = generate_cvs_data(args, device); traj_limit = 500
        elif args.benchmark_type == 'ZQLZ':
            problems = generate_zqlz_data(args, device); traj_limit = 500
        elif args.benchmark_type in ['LL-R', 'LL-U']:
            problems = generate_ll_data(args, device); traj_limit = 2000
        else:
            raise ValueError(f"Unknown benchmark_type: {args.benchmark_type}")

        selecter = New_Sampler(T=1)
        B = args.sampling_num
        Pchunk = getattr(args, "problem_chunk", 48)

        train_data_list, label_data_list = [], []
        num_probs = len(problems)

        for base in tqdm(range(0, num_probs, Pchunk),
                        desc=f"SIL (chunk={Pchunk}, active=model, finished=mask-only)"):

            end = min(base + Pchunk, num_probs)
            P = end - base

            x0 = problems[base:end].unsqueeze(1).repeat(1, B, 1, 1).reshape(P * B, *problems.shape[1:])
            env = Env(device=device, x=x0)
            env.clear()

            trajectory = []
            traj_actions = []

            finished_problem = torch.zeros(P, dtype=torch.bool, device='cpu')

            for step in range(traj_limit):
                trajectory.append(env.x.clone())

                empty_flat = env.empty
                if empty_flat.dim() > 1:
                    empty_flat = empty_flat.view(-1)
                empty = empty_flat.view(P, B)

                newly = (~finished_problem) & empty.any(dim=1).detach().cpu()
                if newly.any():
                    new_ps = newly.nonzero(as_tuple=False).view(-1)
                    for p in new_ps.tolist():
                        idx = empty[p].long().nonzero(as_tuple=False)[0].item()
                        finished_problem[p] = True

                        if step == 0:
                            continue

                        traj_cat = torch.cat(trajectory, dim=0)
                        act_cat  = torch.cat(traj_actions, dim=0)

                        base_idx = p * B + idx
                        pick = torch.arange(
                            base_idx,
                            base_idx + (P * B) * step,
                            P * B,
                            device=traj_cat.device
                        )

                        best_traj = traj_cat[pick, :, :].clone()
                        best_act  = act_cat[pick].clone()

                        train_data_list.append(best_traj)
                        label_data_list.append(best_act)

                    if finished_problem.all():
                        break

                active_p = (~finished_problem).nonzero(as_tuple=False).view(-1)
                finished_p = (finished_problem).nonzero(as_tuple=False).view(-1)

                x_full = env.x
                x_view = x_full.view(P, B, *x_full.shape[1:])

                Pa = int(active_p.numel())
                x_active = x_view[active_p].reshape(Pa * B, *x_full.shape[1:])
                logits_active = model(x_active)

                a_active = selecter(logits_active).view(-1, 1).long()

                Pf = int(finished_p.numel())
                if Pf > 0:
                    x_finished = x_view[finished_p].reshape(Pf * B, *x_full.shape[1:])
                    env_f = Env(device=device, x=x_finished)

                    if args.problem_type == 'rBRP':
                        env_f.find_target_stack()
                        mask_f = env_f.create_mask_rBRP()
                    elif args.problem_type == 'uBRP':
                        mask_f = env_f.create_mask_uBRP()
                    else:
                        raise ValueError(f"Unknown problem_type: {args.problem_type}")

                    a_finished = self.sample_valid_action_from_invalid_mask(mask_f)
                else:
                    a_finished = None

                A_full = torch.empty((P * B, 1), device=x_full.device, dtype=torch.long)

                active_p_dev = active_p.to(x_full.device)
                idx_active = (active_p_dev[:, None] * B + torch.arange(B, device=x_full.device)[None, :]).reshape(-1)
                A_full[idx_active, 0] = a_active[:, 0]

                if Pf > 0:
                    finished_p_dev = finished_p.to(x_full.device)
                    idx_finished = (finished_p_dev[:, None] * B + torch.arange(B, device=x_full.device)[None, :]).reshape(-1)
                    A_full[idx_finished, 0] = a_finished[:, 0]

                if args.problem_type == 'rBRP':
                    env.step_rBRP(A_full)
                    traj_actions.append(A_full.view(-1).clone())

                elif args.problem_type == 'uBRP':
                    src = (A_full // args.max_stacks)
                    dst = (A_full %  args.max_stacks)
                    actions = torch.cat([src, dst], dim=1).long()
                    env.step_uBRP(actions)
                    traj_actions.append(A_full.view(-1).clone())

                else:
                    raise ValueError(f"Unknown problem_type: {args.problem_type}")

            del trajectory[:]
            del traj_actions[:]

        train_data = torch.cat(train_data_list, dim=0).to(device)
        label_data = torch.cat(label_data_list, dim=0).to(device)
        return train_data, label_data


    
    def validation(self, args, model, total_sampling, sampling_batch=None, temp=1):
        model.eval()

        if args.benchmark_type == 'CVS':
            data_ = data_CVS(args).to('cuda')
            traj_limit = 500
        elif args.benchmark_type == 'ZQLZ':
            data_ = data_ZQLZ(args).to('cuda')
            traj_limit = 500
        elif args.benchmark_type in ['LL-R', 'LL-U']:
            data_ = data_LL(args).to('cuda')
            traj_limit = 2000

        if sampling_batch is None:
            sampling_batch = total_sampling

        selecter = New_Sampler(T=temp)
        total_length = []

        with torch.no_grad():
            for i in tqdm(range(len(data_))):
                temp_length = []
                for _ in range(total_sampling//sampling_batch):
                    Length = torch.zeros(sampling_batch).to('cuda')
                    env = Env(device = 'cuda', x = data_[i:i+1].repeat(sampling_batch,1,1))
                    env.clear()
                    for step in range(traj_limit):
                        if env.empty.any().item():
                            break
                        output = model.to('cuda')(env.x.to('cuda'))
                        if args.problem_type == 'rBRP':    
                            next_action = selecter(output).view(-1,1)
                            Length += (1.0 - env.empty.type(torch.float64))
                            env.step_rBRP(next_action)
                        elif args.problem_type == 'uBRP':
                            next_action = selecter(output)
                            source_node, dest_node = next_action//args.max_stacks, next_action%args.max_stacks
                            actions = torch.cat((source_node,dest_node), 1)
                            Length += (1.0 - env.empty.type(torch.float64))
                            env.step_uBRP(actions)
                    temp_length.append(int(torch.min(Length[0]).item()))
                total_length.append(min(temp_length))
        return sum(total_length)/len(total_length)