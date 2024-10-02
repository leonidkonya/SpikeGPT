########################################################################################################
# The RWKV v2-RNN Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm.auto import tqdm
import numpy as np
import logging
from src.spikingjelly.clock_driven import functional
import os
import datetime
import sys
import math
import pdb
from accelerate import Accelerator
from src.model import L2Wrap
import wandb

accelerator = Accelerator()

logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

log_file = open("wik8-0.01.txt", "a")


class TrainerConfig:
    max_epochs = 10
    batch_size = 64
    train_micro_batch_size_per_gpu = 64
    learning_rate = 4e-4
    betas = (0.9, 0.99)
    eps = 1e-8
    grad_norm_clip = 1.0
    lr_decay = True  # linear warmup followed by cosine decay
    warmup_tokens = 0
    final_tokens = 0
    epoch_save_frequency = 0
    epoch_save_path = 'trained-'
    num_workers = 0  # for DataLoader
    wandb_logging = False
    wandb_prefix = 'spikeGPT'
    wandb_project = 'spiking_lm'
    early_stopping = False
    early_stopping_steps = 1
    shuffle = False
    wandb_ppl_threshold = 10000
    wandb_entity = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, valid_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.valid_dataset = valid_dataset
        self.config = config
        self.avg_loss = -1
        self.min_dev_loss = 100
        self.dev_loss = -1
        self.steps = 0


        if self.wandb_logging_enabled():
            cfg = model.config
            for k in config.__dict__:
                setattr(cfg, k, config.__dict__[k])  # combine cfg
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.get_run_name(),
            )

        self.device = 'cpu'
        if torch.cuda.is_available():  # take over whatever gpus are on the system
            self.device = torch.cuda.current_device()

    def wandb_logging_enabled(self):
        """do logging on a single thread
        """
        return self.config.wandb_logging and (
            not torch.distributed.is_initialized() or
            (torch.distributed.is_initialized() and torch.distributed.get_rank()==0)
        )

    def get_batch_size(self):
        """Returns the real batch size that's used for updating the weights
            (includes batches distributed across GPUs and accumulated gradients)
        NOTE: config.batch_size refers to the dataloader, so in the case of DP
            it's the "mini batch size"
        """
        return self.config.batch_size * self.gradient_accumulation_steps * accelerator.state.num_processes

    @property
    def zero_stage(self):
        return accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["stage"]

    @property
    def gradient_accumulation_steps(self):
        return accelerator.state.deepspeed_plugin.deepspeed_config['gradient_accumulation_steps']


    def get_run_name(self):
        raw_model = self.model.module if hasattr(
            self.model, "module") else self.model
        cfg = raw_model.config
        timestamp = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        lr_str = f'{self.config.learning_rate}-{self.config.lr_final}' if self.config.lr_decay else f'{self.config.learning_rate}'
        run_name = f'{cfg.model_type}_v{cfg.vocab_size}_bs{self.get_batch_size()}_lr{lr_str}_w{self.config.warmup_tokens}_ctx{cfg.ctx_len}_L{cfg.n_layer}_E{cfg.n_embd}_zero{self.zero_stage}_{timestamp}'
        return run_name

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = self.config.train_micro_batch_size_per_gpu
        
        model, optimizer = accelerator.prepare(model, optimizer)

        def run_epoch(split):
            is_train = split == 'train'
            data = self.train_dataset if is_train else self.test_dataset
            if split == 'valid':
                data = self.valid_dataset

            model.train(is_train)
            loader = DataLoader(
                data,
                shuffle=self.config.shuffle,
                pin_memory=config.num_workers>0,
                batch_size=config.batch_size,
                num_workers=config.num_workers
            )

            loader = accelerator.prepare(loader)

            pbar = tqdm(
                enumerate(loader), 
                total=len(loader),
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                disable=not accelerator.is_local_main_process
            ) if is_train else enumerate(loader)

            model.train(is_train)
            dev_loss_all = 0

            for it, (x, y) in pbar:
                # x = x.to(self.device)  # place data on the correct device
                # y = y.to(self.device)

                with torch.set_grad_enabled(is_train):
                    loss = model(x, y)  # forward the model
                    functional.reset_net(model)

                if is_train:  # backprop and update the parameters
                    model.zero_grad()
                    # loss.backward()
                    accelerator.backward(loss)

                    if config.grad_norm_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.grad_norm_clip)

                    optimizer.step()

                    if config.lr_decay:  # decay the learning rate based on our progress
                        # number of tokens processed this step (i.e. label is not -100)
                        self.tokens += (y >= 0).sum()
                        lr_final_factor = config.lr_final / config.learning_rate
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = lr_final_factor + \
                                      (1 - lr_final_factor) * float(self.tokens) / \
                                      float(config.warmup_tokens)
                            progress = 0
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = \
                                (0.5 + lr_final_factor / 2) + \
                                (0.5 - lr_final_factor / 2) * math.cos(math.pi * progress)  # better 1.0 ~ 0.1
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate
                        self.tokens += (y >= 0).sum()
                        progress = float(self.tokens - config.warmup_tokens) / float(
                            max(1, config.final_tokens - config.warmup_tokens))

                    now_loss = loss.item()  # report progress
                    self.lr = lr

                    if self.wandb_logging_enabled():
                        wandb.log({"loss": now_loss},
                                    step=self.steps * self.config.batch_size)
                        _ppl = math.exp(now_loss)
                        if _ppl < self.config.wandb_ppl_threshold:
                            wandb.log({"PPL": _ppl},
                                        step=self.steps * self.config.batch_size)
                    self.steps += 1

                    if self.avg_loss < 0:
                        self.avg_loss = now_loss
                    else:
                        factor = 1 / (it + 1)
                        self.avg_loss = self.avg_loss * \
                                        (1.0 - factor) + now_loss * factor
                    pbar.set_description(
                        f"mini-epoch {epoch + 1} prog {progress * 100.0:.2f}% iter {it}: ppl {math.exp(self.avg_loss):.2f} loss {self.avg_loss:.4f} lr {lr:e}")
                else:
                    dev_loss_all += loss.item()

                # for debugging purposes
                if config.early_stopping and config.early_stopping_steps == it:
                    break

            if not is_train:
                self.dev_loss = dev_loss_all / len(loader)

        self.tokens = 0  # counter used for learning rate decay
        for epoch in range(config.max_epochs):
            # save_flag = False

            run_epoch('train')
            log_file.write(
                f'{epoch + 1} {self.avg_loss:.6f} {math.exp(self.avg_loss):.4f} {self.lr:.8f} {datetime.datetime.now()} \n')
            log_file.flush()

            # TODO
            # problem: evaluation and testing is only reported on a single thread
            # pointless to do twice the work
            # (is that true?)

            if self.valid_dataset:
                run_epoch('valid')
                log_file.write(
                    f'{epoch + 1} {self.dev_loss:.6f} {math.exp(self.dev_loss):.4f} {self.lr:.8f} {datetime.datetime.now()} \n')
                log_file.flush()
                if self.wandb_logging_enabled():
                    wandb.log({"val loss": self.dev_loss},
                                step=self.steps * self.config.batch_size)
                    _ppl = math.exp(self.dev_loss)
                    if _ppl < self.config.wandb_ppl_threshold:
                        wandb.log({"val PPL": _ppl},
                                    step=self.steps * self.config.batch_size)

            
            if self.test_dataset:
                run_epoch('test')
                log_file.write(
                    f'{epoch+1} {self.dev_loss:.6f} {math.exp(self.dev_loss):.4f} {self.lr:.8f} {datetime.datetime.now()} \n')
                log_file.flush()
                if self.wandb_logging_enabled():
                    wandb.log({"test loss": self.dev_loss},
                                step=self.steps * self.config.batch_size)
                    _ppl = math.exp(self.dev_loss)
                    if _ppl < self.config.wandb_ppl_threshold:
                        wandb.log({"test PPL": _ppl},
                                    step=self.steps * self.config.batch_size)

            # if self.dev_loss < self.min_dev_loss:
            #     self.min_dev_loss = self.dev_loss
            #     save_flag = True

            if (
                    self.config.epoch_save_frequency > 0 and 
                    epoch % self.config.epoch_save_frequency == 0
                ) or (epoch == config.max_epochs - 1):

                # DataParallel wrappers keep raw model object in .module
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                raw_model = unwrapped_model.module if hasattr(
                    unwrapped_model, "module") else unwrapped_model
                
                run_name = self.get_run_name()
                torch.save(
                    raw_model.state_dict(),
                    os.path.join(
                        self.config.epoch_save_path,
                        f'{self.config.wandb_prefix}-{epoch+1}-{run_name}.pth',
                    ),
                )

#             if epoch >=100 and save_flag:
#                 accelerator.wait_for_everyone()
#                 unwrapped_model = accelerator.unwrap_model(model)
#                 raw_model = unwrapped_model.module if hasattr(
#                     unwrapped_model, "module") else unwrapped_model
#                 torch.save(raw_model.state_dict(),
#                            self.config.epoch_save_path + + str(epoch+1) + 'best_dev' + '.pth')
