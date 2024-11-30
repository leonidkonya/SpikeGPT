

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
from torch.utils.data import IterableDataset
from tokenizers import Tokenizer
from torch.utils.data import Dataset

from src.model_debug import GPT, GPTConfig

from src import utils

import torch.distributed as dist


from accelerate import Accelerator
accelerator = Accelerator()

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True



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
    epoch_length_fixed  = 1_000
    num_workers = 0  # for DataLoader
    wandb_logging = False
    wandb_prefix = 'spikeGPT'
    wandb_project = 'spiking_lm'
    early_stopping = False
    early_stopping_steps = 1
    # shuffle = False
    save_mid_epoch = True
    wandb_ppl_threshold = 10000
    wandb_entity = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class CustomDataset(IterableDataset):
    def __init__(self, path, data_files, split, tokenizer, text_field="text", ctx_len=128):
        """
        Initialize the iterable dataset.
        Args:
            dataset_name (str): The name of the dataset (e.g., path to jsonl.gz).
            split (str): The dataset split to use (e.g., "train" or "validation").
            tokenizer (Tokenizer): The tokenizer from `tokenizers` library.
            text_field (str): The key in the dataset to retrieve text.
            ctx_len (int): Maximum length of tokenized sequences.
        """
        # Load dataset with streaming enabled
        self.dataset = datasets.load_dataset(
            path,
            data_files=data_files,
            split=split,
            streaming=True,
        )
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.ctx_len = ctx_len

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def __iter__(self):
        # This method provides an iterator over the dataset.
        for item in self.dataset:
            text = item[self.text_field]
            # print('text being sampled:')
            # print(text)
            # Tokenize the text on-the-fly
            # tokens = self.tokenizer.encode(text, truncation=True, max_length=self.ctx_len+1)
            tokens = self.tokenizer.encode(text)
            # Convert to tensor
            input_ids = torch.tensor(tokens.ids[:-1], dtype=torch.long)
            targets = torch.tensor(tokens.ids[1:], dtype=torch.long)
            yield input_ids, targets

class Trainer:

    def __init__(
            self,
            model: nn.Module,
            config: TrainerConfig,
            train_dataset: Dataset,
            valid_dataset: Dataset = None,
            test_dataset: Dataset = None,
        ):

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
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        cfg = raw_model.config
        timestamp = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        lr_str = f'{self.config.learning_rate}-{self.config.lr_final}' if self.config.lr_decay else f'{self.config.learning_rate}'
        run_name = f'v{cfg.vocab_size}_bs{self.get_batch_size()}_lr{lr_str}_w{self.config.warmup_tokens}_ctx{cfg.ctx_len}_L{cfg.n_layer}_E{cfg.n_embd}_zero{self.zero_stage}_{timestamp}'
        return run_name
    

    def run_epoch(self, split, optimizer, epoch):

        config = self.config
        is_train = split == 'train'
        # data = self.train_dataset if is_train else self.test_dataset
        if split == 'train':
            data = self.train_dataset
        elif split == 'valid':
            data = self.valid_dataset
        # elif split == 'test':
        else:
            data = self.test_dataset

        self.model.train(is_train)
        loader = DataLoader(
            data,
            # shuffle=self.config.shuffle, #shuffle should be disabled with streaming
            pin_memory=config.num_workers>0,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
        )

        loader = accelerator.prepare(loader)

        pbar = tqdm(
            enumerate(loader), 
            # total=len(loader),
            total=self.config.epoch_length_fixed,
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
            disable=not accelerator.is_local_main_process,
        ) if is_train else enumerate(loader)

        self.model.train(is_train)
        dev_loss_all = 0

        for it, (x, y) in pbar:

            with torch.set_grad_enabled(is_train):
                loss = self.model(x, y)
                functional.reset_net(self.model)

            if is_train:
                self.model.zero_grad()
                accelerator.backward(loss)

                if config.grad_norm_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        config.grad_norm_clip,
                    )

                optimizer.step()

                if config.lr_decay:  # decay the learning rate based on our progress
                    # number of tokens processed this step (i.e. label is not -100) but why would it be -100? padding?
                    self.tokens += (y >= 0).sum()
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
                    self.tokens += (y >= 0).sum()
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
                    f"mini-epoch {epoch + 1} prog {progress * 100.0:.2f}% "
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

        # utils._hprint(f'about to configure optimizers',main_only=False)

        optimizer = raw_model.configure_optimizers(config)

        # utils._hprint(f'configured optimizers',main_only=False)

        # utils._hprint(f'old train_micro_batch_size_per_gpu: {accelerator.state.deepspeed_plugin.deepspeed_config.get("train_micro_batch_size_per_gpu",None)}')

        accelerator.state.deepspeed_plugin.deepspeed_config[
            'train_micro_batch_size_per_gpu'
        ] = self.config.train_micro_batch_size_per_gpu

        # utils._hprint(f'new train_micro_batch_size_per_gpu: {self.config.train_micro_batch_size_per_gpu}')
        
        model, optimizer = accelerator.prepare(model, optimizer)

        self.tokens = 0  # counter used for learning rate decay
        for epoch in range(config.max_epochs):
            # save_flag = False

            self.run_epoch('train', optimizer, epoch)

            # TODO
            # problem: evaluation and testing is only reported on a single thread
            # pointless to do twice the work
            # (is that true?)

            for eval_mode in ['valid', 'test']:
                self.run_epoch(eval_mode)
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
                accelerator.wait_for_everyone()
                raw_model = accelerator.unwrap_model(model)
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






def collate_fn(batch):
    # Pad the sequences to the maximum length in the batch
    # return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    input_batch = [item[0] for item in batch]
    output_batch = [item[1] for item in batch]

    input_padded = torch.nn.utils.rnn.pad_sequence(input_batch, batch_first=True, padding_value=0)
    output_padded = torch.nn.utils.rnn.pad_sequence(output_batch, batch_first=True, padding_value=0)

    return input_padded, output_padded


def main():

    parser = ArgumentParser()
    # parser.add_argument('-c', '--config', type=str, default='config/default_conf.yml')
    parser.add_argument('-c', '--config', type=str, default='config/conf_small_test.yml')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        args = SimpleNamespace(**yaml.safe_load(f))


    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    tokenizer.enable_truncation(max_length=args.ctx_len+1)


    # TODO this only supports json, change that
    # TODO max_length in the dataset is not the max_length in the tokenizer truncation
    train_dataset = CustomDataset(
        'json',
        data_files=args.data_train,
        split='train',
        tokenizer=tokenizer,
        text_field='text',
        ctx_len=args.ctx_len,
    )


    valid_dataset = None
    test_dataset = None


    model = GPT(
        GPTConfig(
            vocab_size=train_dataset.vocab_size,
            ctx_len=train_dataset.ctx_len,
            n_layer=args.n_layer,
            n_embd=args.n_embd,
        )
    ).cuda()


    tconf = TrainerConfig(
        max_epochs=args.n_epoch,
        batch_size=args.batch_size,
        train_micro_batch_size_per_gpu=args.train_micro_batch_size_per_gpu,
        learning_rate=args.lr_init,
        lr_decay=args.lr_decay,
        lr_final=args.lr_final,
        betas=args.betas,
        eps=args.eps,
        grad_norm_clip=args.grad_norm_clip, 
        warmup_tokens=args.warmup_tokens,
        # final_tokens=args.n_epoch*len(train_dataset)*args.ctx_len,
        final_tokens=args.final_tokens,
        num_workers=args.num_workers,
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
        # shuffle=args.shuffle,
    )


    trainer = Trainer(
        model=model,
        config=tconf,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
    )


    trainer.train()

    if args.save_after_train:

        run_name = trainer.get_run_name()
        
        torch.save(
            model.state_dict(),
            f'{args.wandb_prefix}-{args.n_epoch}_final-{run_name}.pth',
        )


if __name__ == "__main__":
    main()