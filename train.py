########################################################################################################
# The RWKV v2-RNN Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import logging
import datetime
import json
from src.model import GPT, GPTConfig
from src.trainer import Trainer, TrainerConfig
from src.utils import Dataset
import torch
import numpy as np
from src.spikingjelly.clock_driven import functional
from src.binidx import MMapIndexedDataset
from accelerate import accelerator
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
import sys

from types import SimpleNamespace
from argparse import ArgumentParser
import yaml

# import src.utils
# src.utils.set_seed(42) # remember to change seed if you load a model

np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,)





if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/default_conf.yml')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        args = SimpleNamespace(**yaml.safe_load(f))

    train_dataset = Dataset(
        data=MMapIndexedDataset(args.datafile_train),
        ctx_len=args.ctx_len,
        epoch_length_fixed=args.epoch_length_fixed,
    )

    # print()
    # print(f'vocab size: {train_dataset.vocab_size}')
    # print()

    valid_dataset = Dataset(
        data=MMapIndexedDataset(args.datafile_valid),
        ctx_len=args.ctx_len,
        epoch_length_fixed=args.epoch_length_fixed_valid,
    ) if args.datafile_valid else None

    test_dataset = Dataset(
        data=MMapIndexedDataset(args.datafile_test),
        ctx_len=args.ctx_len,
        epoch_length_fixed=args.epoch_length_fixed_test,
    ) if args.datafile_test else None

    model = GPT(
        GPTConfig(
            train_dataset.vocab_size,
            train_dataset.ctx_len,
            model_type=args.model_type,
            n_layer=args.n_layer,
            n_embd=args.n_embd,
        )
    ).cuda()

    # # load a trained model. remember to change random seed
    # m2 = torch.load('medium/trained-30L-768E-936.pth',map_location=torch.device('cpu'))
    # model.load_state_dict(m2)

    tconf = TrainerConfig(
        model_type=args.model_type,
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
        final_tokens=args.n_epoch*len(train_dataset)*args.ctx_len,
        num_workers=args.num_workers, epoch_save_frequency=args.epoch_save_frequency,
        epoch_save_path=args.epoch_save_path,
        wandb_logging=args.wandb_logging,
        wandb_project=args.wandb_project,
        wandb_prefix=args.wandb_prefix,
        wandb_ppl_threshold=args.wandb_ppl_threshold,
        wandb_entity=args.wandb_entity,
        early_stopping=args.early_stopping,
        early_stopping_steps=args.early_stopping_steps,
        shuffle=args.shuffle,
    )
    
    trainer = Trainer(
        model,
        train_dataset,
        valid_dataset,
        test_dataset,
        tconf,
    )

    trainer.train()

    run_name = trainer.get_run_name()
    timestamp = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')

    torch.save(
        model.state_dict(),
        f'{args.wandb_prefix}-{args.n_epoch}_final-{run_name}.pth',
    )
