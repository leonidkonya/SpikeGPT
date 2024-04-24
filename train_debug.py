########################################################################################################
# The RWKV v2-RNN Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import logging
import datetime
import json
import yaml
import os
import sys
from argparse import ArgumentParser
from types import SimpleNamespace
from src.model import GPT, GPTConfig
from src.trainer_debug import Trainer, TrainerConfig
from src.utils import Dataset
import torch
import numpy as np
# from src.spikingjelly.clock_driven import functional
from src.binidx import MMapIndexedDataset
# from accelerate import accelerator
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

import pdb


import src.utils
src.utils.set_seed(42) # remember to change seed if you load a model

np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def get_dataset(
        datafile: str,
        dataset_type: str,
        datafile_encoding: str = None,
        ctx_len: int = None,
        epoch_length_fixed: int = None,
    ) -> Dataset:
    
    if dataset_type == "mmap":
        data = MMapIndexedDataset(datafile)
    else: # dataset_type == "other":
        data = open(datafile, "r", encoding=datafile_encoding).read()
    
    dataset = Dataset(
        data=data,
        ctx_len=ctx_len,
        epoch_length_fixed=epoch_length_fixed,
    )

    return dataset

    


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/conf_debug.yaml')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        args = SimpleNamespace(**yaml.safe_load(f))


    train_dataset = get_dataset(
        datafile=args.datafile_train,
        dataset_type=args.dataset_type_train,
        datafile_encoding=args.datafile_encoding,
        ctx_len=args.ctx_len,
        epoch_length_fixed=args.epoch_length_fixed,
    )


    if args.validate:
        valid_dataset = get_dataset(
            datafile=args.datafile_valid,
            dataset_type=args.dataset_type_valid,
            datafile_encoding=args.datafile_encoding,
            ctx_len=args.ctx_len,
            epoch_length_fixed=args.epoch_length_fixed,
        )
    else:
        valid_dataset = None

    if args.test:
        test_dataset = get_dataset(
            datafile=args.datafile_test,
            dataset_type=args.dataset_type_test,
            datafile_encoding=args.datafile_encoding,
            ctx_len=args.ctx_len,
            epoch_length_fixed=args.epoch_length_fixed,
        )
    else:
        test_dataset = None
    

    model = GPT(
        GPTConfig(
            vocab_size=train_dataset.vocab_size,
            ctx_len=train_dataset.ctx_len,
            #kwargs:
            model_type=args.model_type,
            n_layer=args.n_layer,
            n_embd=args.n_embd
        )
    ).cuda()

    # load a trained model. remember to change random seed
    # m2 = torch.load('medium/trained-30L-768E-936.pth',map_location=torch.device('cpu'))
    # model.load_state_dict(m2)
    # print('model', model_type, 'epoch', n_epoch, 'batchsz', batch_size, 'betas',
    #       betas, 'eps', eps, 'ctx', ctx_len, 'layer', n_layer, 'embd', n_embd, )
    
    tconf = TrainerConfig(
        model_type=args.model_type,
        max_epochs=args.n_epoch,
        batch_size=args.batch_size,
        learning_rate=args.lr_init,
        lr_decay=True,
        lr_final=args.lr_final,
        betas=args.betas,
        eps=args.eps,
        grad_norm_clip=args.grad_norm_clip,
        warmup_tokens=args.warmup_tokens,
        final_tokens=args.n_epoch*len(train_dataset)*args.ctx_len,
        num_workers=args.num_workers,
        epoch_save_frequency=args.epoch_save_frequency,
        epoch_save_path=args.epoch_save_path,
        test=args.test,
        validate=args.validate,
        early_stopping=args.early_stopping,
        early_stopping_steps=args.early_stopping_steps,
    )
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        config=tconf,
    )

    trainer.train()

    save_name = "-".join([*map(str,[
        'trained',
        args.n_epoch,
        trainer.get_run_name(),
        datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    ])]) + '.pth'
    
    torch.save(
        model.state_dict(),
        os.path.join(args.epoch_save_path,save_name)
    )
    
