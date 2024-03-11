import os
import time
import torch
import yaml
from glob import glob


class Writer:
    def __init__(self, args=None):
        self.args = args

        if self.args is not None:
            tmp_log_list = glob(os.path.join(args.out_dir, 'log*'))
            if len(tmp_log_list) == 0:
                self.log_file = os.path.join(
                    args.out_dir, 'log_{:s}.txt'.format(
                        time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())))
            else:
                self.log_file = tmp_log_list[0]
            self.log_model_file = self.log_file.replace('log_', 'log_context_')
            self.log_yaml = self.log_file.replace('log_', 'config_').replace('.txt', '.yaml')
    
    def print_additional_info(self, message):
        with open(self.log_model_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))

    def print_info(self, info):
        message = 'Epoch: {}/{}, Duration: {:.3f}s, Train Loss: {:.4f}, Eval Loss: {:.4f}' \
                .format(info['current_epoch'], info['epochs'], info['t_duration'], \
                info['train_loss'], info['test_loss'])
        with open(self.log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)
    
    def print_parameters(self, epoch, flag):
        message = 'Epoch: {} flag: {}'.format(epoch, flag)
        for k,v in model.named_parameters():
            if flag in k:
                message + str(v.mean().item()) + ' '
        with open(self.log_model_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)        

    def save_config(self, args):
        args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
        with open(self.log_yaml, 'w') as f:
            f.write(args_text)

    def save_checkpoint(self, model, optimizer, scheduler, epoch, distributed):
        if distributed:
            sd = model.module.state_dict()
        else:
            sd = model.state_dict()
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': sd,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            },
            os.path.join(self.args.checkpoints_dir,
                         'checkpoint_{:03d}.pt'.format(epoch)))
