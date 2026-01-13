import os
import re
import torch
import numpy as np

# for CVS
def transform_format(instance_file):
    with open(instance_file, 'r') as file:
        lines = file.readlines()
    # Extracting number_of_stacks and number_of_blocks
    num_stacks, num_blocks = map(int, lines[0].split())
    # Initializing the result list
    result = []
    # Loop through each row and transform the data
    for i in range(1, num_stacks + 1):
        # 0~1 unifrom distribution
        block_values = list(map(lambda x: ((num_blocks+1) - int(x))/(num_blocks+1), lines[i].split()[1:]))
        row = block_values + [0] * 2
        result.append(row)
    return torch.tensor(result)

# for CVS
def process_files_with_regex(directory_path, file_regex):
    # Use re to find files matching the specified regex pattern
    files = [file for file in os.listdir(directory_path) if re.search(file_regex, file)]
    transform_datas = []
    for file_name in files:
        #print(file_name)
        file_path = os.path.join(directory_path, file_name)
        transformed_data = transform_format(file_path)
        transform_datas.append(transformed_data.unsqueeze(0))
    return torch.cat(transform_datas)


def data_CVS(args):
    H = args.max_tiers - 2
    W = args.max_stacks
    file_regex= f"data{H}-{W}-.*"
    directory_path  = 'datasets\\CVS' # This path could be changed in different environment
    transform_datas = process_files_with_regex(directory_path, file_regex)
    return transform_datas



# for ZQLZ
def parse_zqlz_instance(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 첫줄 제거
    lines = lines[1:]

    data = []
    for line in lines:
        parts = line.strip().split()

        # 빈 줄 또는 첫 숫자만 있는 줄 → 전부 0으로 채움
        if len(parts) <= 1:
            data.append([0])  # 임시로 하나 넣고 나중에 패딩
            continue

        # 첫 번째 열 제거 후 숫자로 변환
        row = [int(x) for x in parts[1:]]
        data.append(row)

    # 0-padding: 가장 긴 row 길이에 맞춤
    max_len = max(len(row) for row in data)
    padded = [row + [0] * (max_len - len(row)) for row in data]

    padded = torch.tensor(padded, dtype=torch.float32)
    
    n_containers = padded.max()
    mask = padded != 0
    padded[mask] = (n_containers + 1 - padded[mask]) / (n_containers + 1)
    padded = padded.to(torch.float32)


    return padded

def data_ZQLZ(args, base_dir="./datasets/ZQLZ"):
    max_tiers = args.max_tiers
    max_stacks = args.max_stacks
    pattern = f"{max_tiers}-{max_stacks}"
    all_instances = []

    for folder in os.listdir(base_dir):
        if folder.startswith(pattern):
            folder_path = os.path.join(base_dir, folder)
            for fname in os.listdir(folder_path):
                if fname.endswith(".txt"):
                    file_path = os.path.join(folder_path, fname)
                    instance = parse_zqlz_instance(file_path)
                    all_instances.append(instance)

    if not all_instances:
        raise ValueError(f"No instances found for pattern {pattern}")

    # 모두 같은 크기로 맞춰져 있다고 가정 (S x T)
    tensor = torch.stack(all_instances, dim=0)
    return tensor



# for LL
def parse_LL_file(file_path, n_bays, n_rows, n_tiers):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Create a zero-filled numpy array of shape (n_bays * n_rows, n_tiers)
    container_matrix = np.zeros((n_bays * n_rows, n_tiers), dtype=int)
    
    # Process each subsequent line
    for line in lines[1:]:  # Skip the first line
        values = list(map(int, line.split()))
        bay, stack, num_tiers = values[:3]
        container_numbers = values[3:]
        
        # Remove duplicates (since IDs are repeated twice)
        unique_container_numbers = list(dict.fromkeys(container_numbers))  # Preserve order while removing duplicates
        
        if len(unique_container_numbers) != num_tiers:
            ValueError(f'len(unique_container_numbers)(={len(unique_container_numbers)})'
                       +f' != numtiers(={num_tiers})')
        if len(unique_container_numbers) > n_tiers:
            ValueError(f'len(unique_container_numbers)(={len(unique_container_numbers)})'
                       +f' > n_tiers(={n_tiers})')
        
        # Pad with zeros if the number of unique containers is less than max tiers
        padded_containers = unique_container_numbers + [0] * (n_tiers - len(unique_container_numbers))
        
        # Compute the index in the matrix
        stack_index = (bay - 1) * n_rows + (stack - 1)
        
        # Fill the matrix with container numbers (bottom-up stacking)
        container_matrix[stack_index] = padded_containers
    
    # Convert to torch tensor with shape (1, n_bays * n_rows, n_tiers)
    container_tensor = torch.tensor(container_matrix).unsqueeze(0).float()
    # container_tensor.reshape(container_tensor.shape[0], n_bays, n_rows, container_tensor.shape[-1])

    
    # 값 변형 (0~1)
    mask = container_tensor != 0
    N = mask.sum()

    # (N+1 - x_i) / (N+1)
    x_new = container_tensor.clone()
    x_new[mask] = (N + 1 - container_tensor[mask]) / (N + 1)

    return x_new

def data_LL(args):
    T = args.max_tiers
    S = args.max_stacks
    R = 16
    B = S // R
    type_ = args.benchmark_type.split('-')[1]  # 'R' or 'U'
    tiers_str = f"{T:02d}"
    rows_str = f"{R:02d}"
    bays_str = f"{B:02d}"
    file_regex = f"{type_}{bays_str}{rows_str}{tiers_str}"

    if type_ == 'R':
        directory_path = './datasets/LL/Individual, random'
    elif type_ == 'U':
        directory_path = './datasets/LL/Individual, upside down'

    files = [file for file in os.listdir(directory_path) if re.search(file_regex, file)]

    transform_datas = []
    for file_name in files:
        file_path = os.path.join(directory_path, file_name)
        transformed_data = parse_LL_file(file_path, B, R, T)
        transform_datas.append(transformed_data)
    
    transform_datas_tensor = torch.cat(transform_datas)
    return transform_datas_tensor