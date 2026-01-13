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


# for ZQLZ
def generate_yards(n_samples: int, n_containers: int, max_stacks: int, max_tiers: int, device=None):
    total = max_stacks * max_tiers
    assert 0 <= n_containers <= total

    # 1..n_containers 와 0 패딩
    base = torch.cat([torch.arange(1, n_containers + 1, device=device, dtype=float),
                      torch.zeros(total - n_containers, device=device, dtype=float)]).to(device)

    # 각 샘플별 셔플
    idx = torch.stack([torch.randperm(total, device=device) for _ in range(n_samples)])
    x = base[idx].view(n_samples, max_stacks, max_tiers)

    # 0이 아닌 값 앞으로, 0은 뒤로
    mask = (x != 0)                                   # (N,S,T)
    k = mask.sum(dim=2)                               # 각 stack별 nonzero 개수, (N,S)

    nz_rank = (mask.cumsum(dim=2) - 1).clamp_min(0)   # nonzero의 0-based rank
    z_rank  = ((~mask).cumsum(dim=2) - 1).clamp_min(0)

    target_pos = torch.where(mask, nz_rank, k.unsqueeze(2) + z_rank)   # (N,S,T)

    N, S, T = x.shape
    row_offset = torch.arange(S, device=x.device).view(1, S, 1) * T
    batch_offset = torch.arange(N, device=x.device).view(N,1,1) * S * T
    flat_target = (target_pos + row_offset + batch_offset).view(-1)

    out = torch.zeros_like(x).to(device)
    out.view(-1).scatter_(0, flat_target, x.view(-1))
    return out

# for ZQLZ
def find_valid_samples_all(yards, n_containers, max_stacks, max_tiers):
    """
    yards: (N, S, T) tensor
    return: 모든 i 검사를 통과한 sample index
    """
    N, S, T = yards.shape
    total = S * T
    max_i = max_tiers + n_containers - total - 1

    # 확인할 i 가 없으면 전체 통과
    if max_i < 1:
        return torch.arange(N, device=yards.device)

    # 초기: 모든 sample 후보
    valid = torch.ones(N, dtype=torch.bool, device=yards.device)

    # 높이 index (아래=1 ... 위=T)
    tier_index = torch.arange(1, T+1, device=yards.device).view(1,1,T)

    for i in range(1, max_i + 1):
        mask = (yards == i)   # (N,S,T)
        heights = (mask * tier_index).amax(dim=(1,2))   # (N,)

        rhs = (total - n_containers) + (i - 1)
        cond = (max_tiers - heights) <= rhs

        valid &= cond
        if not valid.any():   # 더 이상 남은 후보 없으면 조기 종료
            break

    return torch.nonzero(valid, as_tuple=True)[0]


# for ZQLZ
def generate_feasible_yards(n_samples: int, n_containers: int, max_stacks: int, max_tiers: int,
                            device=None):
    collected = []
    collected_total = 0

    while collected_total < n_samples:
        # 일단 넉넉히 뽑아옴 (현재 부족한 개수의 2~3배 정도)
        batch_size = 1024 * 10

        # 후보 샘플 생성
        yards = generate_yards(batch_size, n_containers, max_stacks, max_tiers,
                               device=device)

        # feasible index 추출
        idx = find_valid_samples_all(yards, n_containers, max_stacks, max_tiers)

        if idx.numel() > 0:
            feasible_yards = yards[idx]
            mask = feasible_yards != 0
            feasible_yards[mask] = (n_containers + 1 - feasible_yards[mask]) / (n_containers + 1)
            feasible_yards = feasible_yards.to(torch.float32)
            collected.append(feasible_yards)
            collected_total += feasible_yards.size(0)

    # 필요한 개수만큼 잘라서 반환
    result = torch.cat(collected, dim=0)[:n_samples]
    return result



# for ZQLZ
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

    # 각 n_containers 값별로 균등하게 샘플 분배
    base_per_n = n_samples // num_values
    remainder = n_samples % num_values  # 나머지 샘플 분배용

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
    type_ = args.benchmark_type.split('-')[1]  # 'R' or 'U'
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

    # ✅ PyTorch Tensor를 사용하여 초기화 (GPU로 이동 가능)
    dataset = torch.zeros((n_samples, n_stacks, n_tiers), dtype=torch.float32).to(device)

    # ✅ PyTorch를 사용하여 컨테이너 순서를 무작위로 생성
    container_sequences = torch.rand((n_samples, n_containers), device=device).argsort(dim=-1).float() + 1

    # ✅ 스택을 랜덤하게 배정 (완전히 PyTorch 연산으로 변환)
    stack_fill_counts = torch.zeros((n_samples, n_stacks), dtype=torch.int32, device=device)

    for j in range(n_containers):
        valid_stacks = stack_fill_counts < n_tiers  # 공간이 있는 스택
        valid_stacks_float = valid_stacks.float()  # softmax를 위한 float 변환
        
        # ✅ GPU에서 직접 스택을 랜덤 선택
        stack_probs = valid_stacks_float / valid_stacks_float.sum(dim=-1, keepdim=True)  # 확률로 변환
        selected_stacks = torch.multinomial(stack_probs, 1).squeeze(dim=-1)  # 각 샘플별로 하나의 스택 선택

        tier_positions = stack_fill_counts[torch.arange(n_samples, device=device), selected_stacks]
        dataset[torch.arange(n_samples, device=device), selected_stacks, tier_positions] = container_sequences[:, j]
        stack_fill_counts[torch.arange(n_samples, device=device), selected_stacks] += 1

    # ✅ instance_type이 'upsidedown'이면 각 stack을 정렬 (완전 GPU 연산)
    if type_ == 'U':
        mask = dataset > 0  # 0이 아닌 위치 찾기
        sorted_data, _ = torch.sort(torch.where(mask, dataset, torch.inf), dim=-1)  # 0을 무한대로 치환하여 정렬
        sorted_data[sorted_data == torch.inf] = 0  # 다시 0으로 복원
        dataset[:] = sorted_data  # 원본 데이터 업데이트

    _, total_stacks, _ = dataset.shape
    assert total_stacks == n_bays * 16
    # dataset = dataset.reshape(batch_size, n_bays, n_rows, feature_dim)

    # 값 변형 (0~1)
    mask = dataset != 0

    # (N+1 - x_i) / (N+1)
    dataset_new = dataset.clone()
    dataset_new[mask] = (n_containers + 1 - dataset[mask]) / (n_containers + 1)
    
    dataset_new = dataset_new.to(torch.float32)
    return dataset_new