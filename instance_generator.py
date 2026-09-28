import torch
import numpy as np


def generate_cvs_data(args, device, n_samples=None, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
	
    if n_samples is None:
        n_samples = args.problem_num
    max_stacks = args.max_stacks
    max_tiers = args.max_tiers
    n_containers = max_stacks * (max_tiers - 2)
	
    dataset = torch.zeros((n_samples, max_stacks, max_tiers), dtype=float).to(device)
	
    if max_stacks * max_tiers < n_containers:
        assert max_stacks * max_tiers >= n_containers
	
	
    for i in range(n_samples):
        per = np.arange(0, n_containers, 1)
        np.random.shuffle(per)
        per=torch.FloatTensor((per+1)/(n_containers+1.0))
        data = torch.reshape(per,(max_stacks,max_tiers-2)).to(device)
        data = data[torch.randperm(data.size()[0])]
        add_empty= torch.zeros((max_stacks,2),dtype=float).to(device)
        dataset[i]=torch.cat( (data,add_empty) ,dim=1).to(device)
    
    dataset=dataset.to(torch.float32)
    return dataset


def generate_yards(n_samples: int, n_containers: int, max_stacks: int, max_tiers: int, device=None):
    total = max_stacks * max_tiers
    assert 0 <= n_containers <= total

    base = torch.cat([torch.arange(1, n_containers + 1, device=device, dtype=float),
                      torch.zeros(total - n_containers, device=device, dtype=float)]).to(device)

    idx = torch.stack([torch.randperm(total, device=device) for _ in range(n_samples)])
    x = base[idx].view(n_samples, max_stacks, max_tiers)

    mask = (x != 0)
    k = mask.sum(dim=2)

    nz_rank = (mask.cumsum(dim=2) - 1).clamp_min(0)
    z_rank  = ((~mask).cumsum(dim=2) - 1).clamp_min(0)

    target_pos = torch.where(mask, nz_rank, k.unsqueeze(2) + z_rank)

    N, S, T = x.shape
    row_offset = torch.arange(S, device=x.device).view(1, S, 1) * T
    batch_offset = torch.arange(N, device=x.device).view(N,1,1) * S * T
    flat_target = (target_pos + row_offset + batch_offset).view(-1)

    out = torch.zeros_like(x).to(device)
    out.view(-1).scatter_(0, flat_target, x.view(-1))
    return out

def find_valid_samples_all(yards, n_containers, max_stacks, max_tiers):
    N, S, T = yards.shape
    total = S * T
    max_i = max_tiers + n_containers - total - 1

    if max_i < 1:
        return torch.arange(N, device=yards.device)

    valid = torch.ones(N, dtype=torch.bool, device=yards.device)

    tier_index = torch.arange(1, T+1, device=yards.device).view(1,1,T)

    for i in range(1, max_i + 1):
        mask = (yards == i)
        heights = (mask * tier_index).amax(dim=(1,2))

        rhs = (total - n_containers) + (i - 1)
        cond = (max_tiers - heights) <= rhs

        valid &= cond
        if not valid.any():
            break

    return torch.nonzero(valid, as_tuple=True)[0]

def generate_feasible_yards(n_samples: int, n_containers: int, max_stacks: int, max_tiers: int,
                            device=None):
    collected = []
    collected_total = 0

    while collected_total < n_samples:
        batch_size = 1024 * 10

        yards = generate_yards(batch_size, n_containers, max_stacks, max_tiers,
                               device=device)

        idx = find_valid_samples_all(yards, n_containers, max_stacks, max_tiers)

        if idx.numel() > 0:
            feasible_yards = yards[idx]
            mask = feasible_yards != 0
            feasible_yards[mask] = (n_containers + 1 - feasible_yards[mask]) / (n_containers + 1)
            feasible_yards = feasible_yards.to(torch.float32)
            collected.append(feasible_yards)
            collected_total += feasible_yards.size(0)

    result = torch.cat(collected, dim=0)[:n_samples]
    return result



def get_n_containers_range(max_stacks, max_tiers):
    n_containers_ranges = {
        (3,6):range(15,18),
        (3,7):range(18,21),
        (3,8):range(21,24),
        (3,9):range(24,27),
        (3,10):range(27,30),

        (4,6):range(20,24),
        (4,7):range(24,28),
        (4,8):range(28,32),
        (4,9):range(32,36),
        (4,10):range(36,40),

        (5,6):range(25,30),
        (5,7):range(30,35),
        (5,8):range(35,40),
        (5,9):range(40,45),
        (5,10):range(45,50),

        (6,6):range(30,36),
        (6,7):range(36,42),
        (6,8):range(42,48),
        (6,9):range(48,54),
        (6,10):range(54,60),

        (7,6):range(35,42),
        (7,7):range(42,49),
        (7,8):range(49,56),
        (7,9):range(56,63),
        (7,10):range(63,70),
    }
    return n_containers_ranges[max_tiers, max_stacks]
     


def generate_zqlz_data(args, device, n_samples=None, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if n_samples is None:
        n_samples = args.problem_num
    max_stacks = args.max_stacks
    max_tiers = args.max_tiers
    
    n_range = get_n_containers_range(max_stacks, max_tiers)
    n_values = list(n_range)
    num_values = len(n_values)

    base_per_n = n_samples // num_values
    remainder = n_samples % num_values

    samples_per_n = [base_per_n + (1 if i < remainder else 0) for i in range(num_values)]

    all_yards = []

    for n_containers, n_each in zip(n_values, samples_per_n):
        if n_each == 0:
            continue
        yards = generate_feasible_yards(
            n_samples=n_each,
            n_containers=n_containers,
            max_stacks=max_stacks,
            max_tiers=max_tiers,
            device=device
        )
        all_yards.append(yards)

    result = torch.cat(all_yards, dim=0).to(device)
    return result




def generate_ll_data(args, device, n_samples=None, seed=None):
    if n_samples is None:
        n_samples = args.problem_num
    max_stacks = args.max_stacks
    max_tiers = args.max_tiers
    type_ = args.benchmark_type.split('-')[1]
    n_containers_dict = {
            (1,6):70,
            (2,6):140,
            (4,6):280,
            (6,6):430,
            (8,6):570,
            (10,6):720,
            (1,8):90,
            (2,8):190,
            (4,8):380,
            (6,8):570,
    }
    n_bays = max_stacks // 16
    n_tiers = max_tiers
    n_containers = n_containers_dict[n_bays,n_tiers]
    n_stacks = n_bays * 16

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    dataset = torch.zeros((n_samples, n_stacks, n_tiers), dtype=torch.float32).to(device)

    container_sequences = torch.rand((n_samples, n_containers), device=device).argsort(dim=-1).float() + 1

    stack_fill_counts = torch.zeros((n_samples, n_stacks), dtype=torch.int32, device=device)

    for j in range(n_containers):
        valid_stacks = stack_fill_counts < n_tiers
        valid_stacks_float = valid_stacks.float()
        
        stack_probs = valid_stacks_float / valid_stacks_float.sum(dim=-1, keepdim=True)
        selected_stacks = torch.multinomial(stack_probs, 1).squeeze(dim=-1)

        tier_positions = stack_fill_counts[torch.arange(n_samples, device=device), selected_stacks]
        dataset[torch.arange(n_samples, device=device), selected_stacks, tier_positions] = container_sequences[:, j]
        stack_fill_counts[torch.arange(n_samples, device=device), selected_stacks] += 1

    if type_ == 'U':
        mask = dataset > 0
        sorted_data, _ = torch.sort(torch.where(mask, dataset, torch.inf), dim=-1)
        sorted_data[sorted_data == torch.inf] = 0
        dataset[:] = sorted_data

    _, total_stacks, _ = dataset.shape
    assert total_stacks == n_bays * 16

    mask = dataset != 0

    dataset_new = dataset.clone()
    dataset_new[mask] = (n_containers + 1 - dataset[mask]) / (n_containers + 1)
    
    dataset_new = dataset_new.to(torch.float32)
    return dataset_new