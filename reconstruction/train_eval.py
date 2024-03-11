import time
import os
import torch
import torch.nn.functional as F
from psbody.mesh import Mesh
from itertools import cycle
from tqdm import tqdm
import glob
import numpy as np
#import mesh
from utils import save_pv_mesh,images_to_video,export_pred_ply

def get_offsets(model,device, epoch):
    model.eval()
    message =''
    named_layers = dict(model.named_modules())
    for k,v in named_layers.items():
        if hasattr(v, 'mean_ro'):
            message = message + 'Epoch {} {}.{}:{}\n'.format(epoch,k,v, v.mean_ro)
    return message

def get_parameters_stat(model, flag, epoch):
    for _flag in flag:
        message = 'Epoch: {} flag: {}'.format(epoch, _flag)
        for k,v in model.named_parameters():
            if _flag in k:
                message = message + ' {}: {:.3f} '.format(k, v.mean().item())
        return message

def run(model, train_loader, test_loader, eval_loader, epochs, optimizer, scheduler, writer, scaler,
        rank, distributed=0, print_params_epoch=50, meshdata=None, out_dir=None):
    train_losses, test_losses = [], []

    for epoch in range(1, epochs + 1):
        if distributed:
            train_loader.sampler.set_epoch(epoch) 
        t = time.time()

        train_loss = train(model, optimizer, train_loader, scaler, rank)
        t_duration = time.time() - t
        test_loss = test(model, test_loader, rank)
        eval_loss = -1.0
        if eval_loader is not None:
            eval_loss = test(model, eval_loader, rank)
        scheduler.step()
        
        info = {
            'current_epoch': epoch,
            'epochs': epochs,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'eval_loss': eval_loss,
            't_duration': t_duration
        }

        writer.print_info(info)
        writer.save_checkpoint(model, optimizer, scheduler, epoch, distributed)

        if (epoch % print_params_epoch==0 or epoch==1):
            eval_error(model, test_loader, rank, meshdata, out_dir, epoch, name='test')
            if eval_loader is not None:
                eval_error(model, eval_loader, rank, meshdata, out_dir, epoch, name='eval')

def train(model, optimizer, loader, scaler, device):
    model.train()
    total_loss = 0
    for _,data in enumerate(tqdm(loader)):
        optimizer.zero_grad()
        x = data.x.to(device)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(x)
                loss = F.l1_loss(out, x, reduction='mean')
                total_loss += loss.item()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        else:
            out = model(x)
            loss = F.l1_loss(out, x, reduction='mean')
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
    return total_loss / len(loader)

def test(model, loader, device):
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            x = data.x.to(device)
            pred = model(x)
            total_loss += F.l1_loss(pred, x, reduction='mean')
    return total_loss / len(loader)

def eval_error(model, test_loader, device, meshdata, out_dir, epoch=-1, name='test'):
    model.eval()

    errors = []
    mean = meshdata.mean
    std = meshdata.std
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data.x.to(device)
            pred = model(x)
            num_graphs = data.num_graphs
            reshaped_pred = (pred.view(num_graphs, -1, 3).cpu() * std) + mean
            reshaped_x = (x.view(num_graphs, -1, 3).cpu() * std) + mean

            reshaped_pred *= 1000
            reshaped_x *= 1000

            tmp_error = torch.sqrt(
                torch.sum((reshaped_pred - reshaped_x)**2,
                          dim=2))  # [num_graphs, num_nodes]
            errors.append(tmp_error)
        new_errors = torch.cat(errors, dim=0)  # [n_total_graphs, num_nodes]

        mean_error = new_errors.view((-1, )).mean()
        std_error = new_errors.view((-1, )).std()
        median_error = new_errors.view((-1, )).median()

    message = '{} Error: {:.3f}+{:.3f} | {:.3f}'.format(epoch, mean_error, std_error,
                                                     median_error)

    out_error_fp = out_dir + '/{}_euc_errors.txt'.format(name)
    with open(out_error_fp, 'a') as log_file:
        log_file.write('{:s}\n'.format(message))
    print(message)
    model.train()