import os, copy
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from params import argparse_train_SIL
from ess import ESS
from utils import reset_seed, load_model, create_log_directory, save_args_txt, add_log_txt





def bc_loss(model, inputs, labels, device):
        inputs = copy.deepcopy(inputs).to(device)
        ps = torch.softmax(model(inputs), dim=1)
        labels = torch.tensor(labels).to(device).view(-1,1)
        loss = - torch.log(torch.gather(ps, dim=1, index=labels))
        return loss.mean()

def train(args):
    print("Training SIL Model for %s-%s with T=%d, S=%d"
          % (args.benchmark_type, args.problem_type, args.max_tiers, args.max_stacks))
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

    t1 = time()

    ess = ESS(device)

    for epoch in range(args.epoch):
        model.eval()
        with torch.no_grad():
            data_pos, labels = ess.generate_SIL_data5(model, args, device)
        dataset = TensorDataset(data_pos, labels)
        dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

        model.train()
        for _, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.view(len(inputs), args.max_stacks, args.max_tiers)
            loss = bc_loss(model, inputs, labels, device)
            optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()

        model.eval()
        torch.save(model.state_dict(), args.log_path + '/epoch%s.pt' % epoch)
        validation_score = ess.validation(args, model, args.valid_sampling)
        add_log_txt(args, epoch, validation_score, t1)

        pass
    






if __name__ == '__main__':
    args = argparse_train_SIL()

    train(args)

    pass