import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from params import argparse_train_IL
from utils import reset_seed, create_log_directory
from decoder import L2BR

def bc_loss(model, inputs, labels, device, args):
    inputs = copy.deepcopy(inputs).to(device)
    ps = torch.softmax(model(inputs), dim=1)
    if args.problem_type == 'rBRP':
        labels = torch.tensor(labels).to(device).view(-1,1) % 5 # max_stacks=5
    elif args.problem_type == 'uBRP':
        labels = torch.tensor(labels).to(device).view(-1,1)
    loss = - torch.log(torch.gather(ps, dim=1, index=labels))
    return loss.mean()

def train(args):
    reset_seed(args.seed)
    create_log_directory(args)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'

    model = L2BR(args, device)
    model=model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    dataset = torch.load(f"./IL/optimal_data/CVS-{args.problem_type}-H=5xS=5/train_data.pt", weights_only=False)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True)
    val_dataset = torch.load(f"./IL/optimal_data/CVS-{args.problem_type}-H=5xS=5/valid_data.pt", weights_only=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

    for epoch in range(args.epoch):
        ave_loss = 0
        model.train()
        for _, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.view(len(inputs), 5, 7) # max_stacks=5, max_tiers=7
            loss = bc_loss(model, inputs, labels, device, args)
            optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()

            ave_loss += loss.item()

        val_loss = 0
        val_n = 0
        model.eval()
        for _, (inputs, labels) in enumerate(val_dataloader):
            val_loss += bc_loss(model, inputs, labels, device, args)*len(inputs)
            val_n += len(inputs)
        val_loss/= val_n
        print(f"[Epoch {epoch}] train_loss: {ave_loss / len(dataloader)}, val_loss: {val_loss.item()}")
        torch.save(model.state_dict(), args.log_path + '/epoch%s.pt' % epoch)





if __name__ == '__main__':
    args = argparse_train_IL()
    train(args)