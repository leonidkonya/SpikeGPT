

from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
# from accelerate import accelerator
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
# from src.model import L2Wrap
import wandb
from types import SimpleNamespace
from argparse import ArgumentParser
import yaml
import datasets
# from torch.utils.data import IterableDataset
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from datasets.arrow_dataset import Dataset as ArrowDataset

from src.model_debug import GPT, GPTConfig

from typing import Union

from src import utils

import torch.distributed as dist

from accelerate import DataLoaderConfiguration
from accelerate.utils import DeepSpeedPlugin

from datasets import load_from_disk

from functools import partial

from accelerate import Accelerator
# accelerator = Accelerator()

from torch.utils.data.sampler import BatchSampler, RandomSampler

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


from src.packing_utils import (
    collate_fn_base,
    print_packed_batch,
    get_packing_stats,
    print_packing_stats,
    SamplePackingBatchSampler,
) 

class TrainerConfig:
    max_epochs = 10
    batch_size = 64
    gradient_accumulation_steps = 1
    # train_micro_batch_size_per_gpu = 64
    learning_rate = 4e-4
    betas = (0.9, 0.99)
    eps = 1e-8
    grad_norm_clip = 1.0
    lr_decay = True  # linear warmup followed by cosine decay
    warmup_tokens = 0
    final_tokens = 0
    epoch_save_frequency = 0
    epoch_save_path = 'trained-'
    epoch_length_fixed  = 1_000
    num_workers = 0

    wandb_logging = False
    wandb_prefix = 'spikeGPT'
    wandb_project = 'spiking_lm'

    early_stopping = False
    early_stopping_steps = 1
    save_mid_epoch = True
    wandb_ppl_threshold = 10000
    wandb_entity = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(
            self,
            model: nn.Module,
            config: TrainerConfig,
            collate_fn,
            accelerator: Accelerator,
            train_batch_sampler: BatchSampler,
            train_dataset: Union[Dataset, ArrowDataset],
            valid_batch_sampler: BatchSampler = None,
            valid_dataset: Union[Dataset, ArrowDataset] = None,
            test_batch_sampler: BatchSampler = None,
            test_dataset: Union[Dataset, ArrowDataset] = None,
        ):

        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.avg_loss = -1
        self.min_dev_loss = 100
        self.dev_loss = -1
        self.steps = 0
        self.collate_fn = collate_fn
        self.train_batch_sampler = train_batch_sampler
        self.valid_batch_sampler = valid_batch_sampler
        self.test_batch_sampler = test_batch_sampler
        self.accelerator = accelerator

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
        return self.config.batch_size * self.gradient_accumulation_steps * self.accelerator.state.num_processes

    @property
    def zero_stage(self):
        return self.accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["stage"]

    @property
    def gradient_accumulation_steps(self):
        return self.accelerator.state.deepspeed_plugin.deepspeed_config['gradient_accumulation_steps']


    def get_run_name(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        cfg = raw_model.config
        timestamp = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        lr_str = f'{self.config.learning_rate}-{self.config.lr_final}' if self.config.lr_decay else f'{self.config.learning_rate}'
        run_name = f'v{cfg.vocab_size}_bs{self.get_batch_size()}_lr{lr_str}_w{self.config.warmup_tokens}_ctx{cfg.ctx_len}_L{cfg.n_layer}_E{cfg.n_embd}_zero{self.zero_stage}_{timestamp}'
        return run_name
    

    def run_epoch(self, split, optimizer, model, epoch, batch_sampler):
        """ split: train | valid | test
        """

        config = self.config
        is_train = split == 'train'
        if split == 'train':
            data = self.train_dataset
        elif split == 'valid':
            data = self.valid_dataset
        else:
            data = self.test_dataset

        # self.model.train(is_train)
        model.train(is_train)
        loader = DataLoader(
            dataset=data,
            batch_sampler=batch_sampler,
            pin_memory=config.pin_memory,
            num_workers=config.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=config.persistent_workers,
        )

        loader = self.accelerator.prepare(loader)

        # print(f'{loader.batch_sampler.batch_sampler.batches} @ {dist.get_rank()}')

        for _ in loader:
            break

        # print(f'{loader.batch_sampler.batch_sampler.batches} @ {dist.get_rank()}')

        # print(f'about to go through pbar @ proc {dist.get_rank()}')
        pbar = tqdm(
            enumerate(loader), 
            total=len(loader), # not the real number, but close enough (re-initialized at each __iter__)
            # total=self.config.epoch_length_fixed,
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
            disable=not self.accelerator.is_local_main_process,
        ) if is_train else enumerate(loader)

        # self.model.train(is_train)
        model.train(is_train)
        dev_loss_all = 0

        eod_token_id = self.config.eod_token_id
        pad_token_id = self.config.pad_token_id

        for it, (x, y) in pbar:

            with torch.set_grad_enabled(is_train):
                # loss = self.model(x, y)
                loss = self.model(x, y)
                # functional.reset_net(self.model)
                functional.reset_net(model)

            if is_train:
                # self.model.zero_grad()
                self.accelerator.backward(loss)

                if config.grad_norm_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        # self.model.parameters(),
                        model.parameters(),
                        config.grad_norm_clip,
                    )

                # perform an update if enough gradients have accumulated
                if (it + 1) % self.config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                if config.lr_decay:  # decay the learning rate based on our progress
                    # number of tokens processed this step (not eod or pad)
                    B, T = y.shape

                    junk_tokens = ((y == eod_token_id) + (y == pad_token_id)).sum()
                    self.tokens += B*T - junk_tokens


                    lr_final_factor = config.lr_final / config.learning_rate
                    if self.tokens < config.warmup_tokens:
                        # linear warmup
                        lr_mult =\
                            lr_final_factor + \
                            (1 - lr_final_factor) * \
                            float(self.tokens) / float(config.warmup_tokens)
                        progress = 0
                    else:
                        # cosine learning rate decay
                        progress =\
                            float(self.tokens - config.warmup_tokens) / \
                            float(max(1, config.final_tokens - config.warmup_tokens))
                        lr_mult = \
                            (0.5 + lr_final_factor / 2) + \
                            (0.5 - lr_final_factor / 2) * math.cos(math.pi * progress)  # better 1.0 ~ 0.1
                    lr = config.learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = config.learning_rate
                    
                    junk_tokens = ((y == eod_token_id) + (y == pad_token_id)).sum()
                    self.tokens += B*T - junk_tokens

                    progress =\
                        float(self.tokens - config.warmup_tokens) / \
                        float(max(1, config.final_tokens - config.warmup_tokens))

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
                    self.avg_loss = \
                        self.avg_loss * (1.0 - factor) +\
                        now_loss * factor

                pbar.set_description(
                    f"epoch {epoch + 1} "
                    f"iter {it}: ppl {math.exp(self.avg_loss):.2f} "
                    f"loss {self.avg_loss:.4f} lr {lr:e}"
                )

            else:
                dev_loss_all += loss.item()

            # for debugging purposes
            if config.early_stopping and config.early_stopping_steps == it:
                break


        if not is_train:
            # self.dev_loss = dev_loss_all / len(loader)
            self.dev_loss = dev_loss_all / (it + 1)


    def train(self):

        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model

        optimizer = raw_model.configure_optimizers(config)
        # accelerator.state.deepspeed_plugin.deepspeed_config[
        #     'train_micro_batch_size_per_gpu'
        # ] = self.config.train_micro_batch_size_per_gpu

        model, optimizer = self.accelerator.prepare(model, optimizer)

        eval_modes = [
            mode for mode, ds in {'valid': self.valid_dataset, 'test': self.test_dataset}.items() if ds
        ]

        self.tokens = 0  # counter used for learning rate decay
        for epoch in range(config.max_epochs):
            # save_flag = False

            self.run_epoch(
                'train',
                optimizer,
                model,
                epoch,
                self.train_batch_sampler,
            )

            # TODO
            # problem: evaluation and testing is only reported on a single thread
            # pointless to do twice the work
            # (is that true?)

            

            for eval_mode in eval_modes:
                self.run_epoch(
                    eval_mode,
                    optimizer,
                    model,
                    epoch,
                    self.valid_batch_sampler if eval_mode=='valid' else self.test_batch_sampler,
                )

                if self.wandb_logging_enabled():
                    wandb.log(
                        {
                            f"{eval_mode} loss": self.dev_loss,
                        },
                        step=self.steps * self.config.batch_size,
                    )
                    _ppl = math.exp(self.dev_loss)
                    if _ppl < self.config.wandb_ppl_threshold:
                        wandb.log(
                            {
                                f"{eval_mode} PPL": _ppl,
                            },
                            step=self.steps * self.config.batch_size,
                        )


            # if self.dev_loss < self.min_dev_loss:
            #     self.min_dev_loss = self.dev_loss
            #     save_flag = True

            if self.config.save_mid_epoch and (
                    (
                        self.config.epoch_save_frequency > 0 and 
                        epoch % self.config.epoch_save_frequency == 0
                    ) or (
                        epoch == config.max_epochs - 1
                    )
                ):

                # DataParallel wrappers keep raw model object in .module
                self.accelerator.wait_for_everyone()
                raw_model = self.accelerator.unwrap_model(model)
                if hasattr(raw_model, "module"):
                    raw_model = raw_model.module

                
                run_name = self.get_run_name()
                torch.save(
                    raw_model.state_dict(),
                    os.path.join(
                        self.config.epoch_save_path,
                        f'{self.config.wandb_prefix}-{epoch+1}-{run_name}.pth',
                    ),
                )


def get_dataset(pretokenized_path):
    """

    """
    if pretokenized_path is None:
        return None
    
    ds = load_from_disk(pretokenized_path)
    ds.__getitems__ = None

    return ds


def get_packing_sampler(
    batch_size: int,
    ctx_len: int,
    dataset: ArrowDataset,
    packing_group_size: int,
):
    
    if dataset is None:
        return None

    return SamplePackingBatchSampler(
        batch_size=batch_size,
        ctx_len=ctx_len,
        group_size=packing_group_size,
        lengths=np.array(dataset['length']),
        sampler=RandomSampler(dataset), # accelerate will synchronize rng states across shards
    )


def main():

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/conf_NEW_test_1.yml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        args = SimpleNamespace(**yaml.safe_load(f))


    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    eod_token_id = tokenizer.token_to_id(args.eod_token)
    pad_token_id = tokenizer.token_to_id(args.pad_token)

    collate_fn = partial(
        collate_fn_base,
        eod_token_id=eod_token_id,
        pad_token_id=pad_token_id,
        ctx_len=args.ctx_len,
    )

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    train_batch_size = args.batch_size * args.gradient_accumulation_steps * world_size

    deepspeed_config = {
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "zero_optimization": args.zero_optimization # dict
    }

    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=args.zero_optimization['stage'],
        hf_ds_config=deepspeed_config,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    dataloader_config = DataLoaderConfiguration(
        use_seedable_sampler=args.use_seedable_sampler,
    )

    accelerator = Accelerator(
        deepspeed_plugin=deepspeed_plugin,
        dataloader_config=dataloader_config,
    )

    train_dataset = get_dataset(args.pretokenized_path_train)
    valid_dataset = get_dataset(args.pretokenized_path_valid)
    test_dataset = get_dataset(args.pretokenized_path_test)

    vocab_size = tokenizer.get_vocab_size()


    train_packing_sampler = get_packing_sampler(
        batch_size=args.batch_size,
        ctx_len=args.ctx_len,
        dataset=train_dataset,
        packing_group_size=args.packing_group_size,
    )
    valid_packing_sampler = get_packing_sampler(
        batch_size=args.batch_size,
        ctx_len=args.ctx_len,
        dataset=valid_dataset,
        packing_group_size=args.packing_group_size,
    )
    test_packing_sampler = get_packing_sampler(
        batch_size=args.batch_size,
        ctx_len=args.ctx_len,
        dataset=test_dataset,
        packing_group_size=args.packing_group_size,
    )


    model = GPT(
        GPTConfig(
            vocab_size=vocab_size,
            ctx_len=args.ctx_len,
            n_layer=args.n_layer,
            n_embd=args.n_embd,
        )
    ).cuda()


    tconf = TrainerConfig(
        max_epochs=args.n_epoch,
        batch_size=args.batch_size,
        
        learning_rate=args.lr_init,
        lr_decay=args.lr_decay,
        lr_final=args.lr_final,
        betas=args.betas,
        eps=args.eps,
        grad_norm_clip=args.grad_norm_clip, 
        
        warmup_tokens=args.warmup_tokens,
        final_tokens=args.final_tokens,

        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        
        epoch_save_frequency=args.epoch_save_frequency,
        epoch_save_path=args.epoch_save_path,
        epoch_length_fixed=args.epoch_length_fixed,
        
        wandb_logging=args.wandb_logging,
        wandb_project=args.wandb_project,
        wandb_prefix=args.wandb_prefix,
        wandb_ppl_threshold=args.wandb_ppl_threshold,
        wandb_entity=args.wandb_entity,

        early_stopping=args.early_stopping,
        early_stopping_steps=args.early_stopping_steps,
        save_mid_epoch=args.save_mid_epoch,

        eod_token_id=eod_token_id,
        pad_token_id=pad_token_id
    )


    trainer = Trainer(
        model=model,
        config=tconf,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        train_batch_sampler=train_packing_sampler,
        valid_batch_sampler=valid_packing_sampler,
        test_batch_sampler=test_packing_sampler,
        accelerator=accelerator,
        collate_fn=collate_fn,
    )

    trainer.train()

    if args.save_after_train:

        run_name = trainer.get_run_name()
        
        torch.save(
            model.state_dict(),
            os.path.join(
                args.epoch_save_path,
                f'{args.wandb_prefix}-{args.n_epoch}_final-{run_name}.pth',
            )
        )


if __name__ == "__main__":
    main()