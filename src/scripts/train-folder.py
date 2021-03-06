import os
import sys
sys.path.insert(0, os.path.abspath('../../.'))
from tqdm import tqdm

import torch
import torch.nn as nn

from src.model.SparseNet import SparseNet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.model.FolderDataset import FolderPatchDataset
from src.utils.cmd_line import parse_args
from src.scripts.plotting import plot_rf

arg = parse_args()

# save to tensorboard
board = SummaryWriter("../../runs/{}_{}".format(arg.train_name, arg.train_id))

# if use cuda
device = torch.device(arg.device_name)

# create net
sparse_net = SparseNet(arg.n_neuron, arg.size, R_lr=arg.r_learning_rate, lmda=arg.reg, device=device)
# sparse_net = nn.DataParallel(sparse_net)

# load data
dataloader = DataLoader(FolderPatchDataset(arg.size, arg.size, N=arg.number, folder=arg.root_path), batch_size=arg.batch_size, num_workers=8)

# train
# optim = torch.optim.SGD([{'params': sparse_net.U.weight, "lr": arg.learning_rate}])
optim = torch.optim.Adam([{'params': sparse_net.U.weight, "lr": arg.learning_rate}])

for e in range(arg.epoch):
    running_loss = 0
    c = 0
    
    for img_batch in tqdm(dataloader, desc='training', total=len(dataloader)):
        img_batch = img_batch.reshape(img_batch.shape[0], -1).to(device)

        # update
        pred = sparse_net(img_batch)
        loss = ((img_batch - pred) ** 2).sum()
        running_loss += loss.item()
        loss.backward()
        
        # update U
        optim.step()
        
        # zero grad
        sparse_net.zero_grad()
        
        # norm
        sparse_net.normalize_weights()
        c += 1
        
    board.add_scalar('Loss', running_loss / c, e * len(dataloader) + c)
    
    if e % 5 == 4:
        # plotting
        fig = plot_rf(sparse_net.U.weight.T.reshape(arg.n_neuron, arg.size, arg.size).cpu().data.numpy(), arg.n_neuron, arg.size)
        board.add_figure('RF', fig, global_step=e * len(dataloader) + c)
        
    if e % 10 == 9:
        # save checkpoint
        torch.save(sparse_net, f"../../trained_models/ckpt-{e+1}.pth")
        
torch.save(sparse_net, f"../../trained_models/ckpt-{e+1}.pth")
