########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np
import math, os, sys, types, time, gc
import torch
from src.utils import TOKENIZER
import matplotlib.ticker as ticker

from torch.nn import functional as F

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=4, suppress=True, linewidth=200)
args = types.SimpleNamespace()

from tokenizers import Tokenizer


########################################################################################################
# Step 1: set model & config (use v4 to run your trained-from-scratch models. v4 and v4neo are compatible)
########################################################################################################

args.RUN_DEVICE = "cuda" # 'cuda' // 'cpu' (already fast)
args.FLOAT_MODE = "fp32" # fp16 (good for GPU, does not work for CPU) // fp32 (good for CPU) // bf16 (less accurate, but works for CPU)

# if args.RUN_DEVICE == "cuda":
#     os.environ["RWKV_RUN_BACKEND"] = 'nvfuser' # !!!BUGGY!!! wrong output
os.environ["RWKV_JIT_ON"] = '1' # '1' or '0'. very useful for GPU/CPU fp32, but might be harmful for GPU fp16. please benchmark !!!

#For BookCorpus Pre-trained model
# TOKEN_MODE = "char"
# WORD_NAME = "vocab_book"
# UNKNOWN_CHAR = ' '
# vocab_size = 77

#For 216M OpenWebText Pre-trained model
# TOKEN_MODE = "pile"
# WORD_NAME = [
#     "20B_tokenizer.json",
#     "20B_tokenizer.json",
# ]  # [vocab, vocab] for Pile model
# UNKNOWN_CHAR = None
# vocab_size = 50277

# MODEL_NAME = 'SpikeGPT-216M'
# n_layer = 18
# n_embd = 768
# ctx_len = 1024


ctx_len = 512
vocab_size = 60000
n_embd = 768
n_layer = 18
MODEL_NAME = '/workspace/spiking_workspace/SpikeGPT/spikeGPT-2_final-RWKV-ffnPre_v60000_ctx512_L18_E768_2024-05-27-13-51-18.pth' # more like model path
model_type = 'RWKV-ffnPre'


args.MODEL_NAME = MODEL_NAME
args.n_layer = n_layer
args.n_embd = n_embd
args.ctx_len = ctx_len
args.vocab_size = vocab_size
args.head_qk = 0
args.pre_ffn = 0
args.grad_cp = 0
args.my_pos_emb = 0
args.model_type = 'RWKV-ffnPre'
os.environ["RWKV_RUN_DEVICE"] = args.RUN_DEVICE

########################################################################################################
# Step 2: set prompt & sampling stuffs
########################################################################################################


# context = 'Prehistoric man sketched an incredible array of prehistoric beasts on the rough limestone walls of a cave in modern day France 36,000 years ago. Now, with the help of cutting-edge technology, those works of art in the Chauvet-Pont-d’Arc Cave have been reproduced to create the biggest replica cave in the world. The manmade cavern named the Caverne du Pont-d’Arc has been built a few miles from the original site in Vallon-Pont-D’arc in Southern France and contains 1,000 painstakingly-reproduced drawings as well as around 450 bones and other features...\n Cavemen and women sketched an incredible array of prehistoric beasts on the rough limestone walls of a cave 36,000 years ago and now a replica has been created (pictured)'

context = "Az ipafai papnak fapipája van, ezért"

NUM_TRIALS = 1
LENGTH_PER_TRIAL = 333

TEMPERATURE = 1.5
top_p = 0.7
top_p_newline = 0.9  # only used in TOKEN_MODE = char

# DEBUG_DEBUG = False  # True False --> show softmax output

########################################################################################################

# print(f'\nUsing {args.RUN_DEVICE.upper()}. Loading {MODEL_NAME}...')
from src.model_run import RWKV_RNN

model = RWKV_RNN(args)

# print(f'\nOptimizing speed...')
#out, _ = model.forward([187], None, None, None)
# print(out)
gc.collect()
torch.cuda.empty_cache()

# input(0)

# print(f'\nLoading tokenizer {WORD_NAME}...')
# tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)
# if TOKEN_MODE == "pile":
#     assert tokenizer.tokenizer.decode([187]) == '\n'

tokenizer = Tokenizer.from_file('/workspace/spiking_workspace/BPE_hu_60000.json')

########################################################################################################

# if tokenizer.charMode:
#     context = tokenizer.refine_context(context)
#     ctx = [tokenizer.stoi.get(s, tokenizer.UNKNOWN_CHAR) for s in context]
# else:
    # ctx = tokenizer.tokenizer.encode(context)

ctx = tokenizer.encode(context).ids
src_len = len(ctx)
src_ctx = ctx.copy()

# print("\nYour prompt has " + str(src_len) + " tokens.")
# print(
#     "Note: currently the first run takes a while if your prompt is long, as we are using RNN to preprocess the prompt. Use GPT to build the hidden state for better speed.\n"
# )

time_slot = {}
time_ref = time.time_ns()

def record_time(name):
    if name not in time_slot:
        time_slot[name] = 1e20
    tt = (time.time_ns() - time_ref) / 1e9
    if tt < time_slot[name]:
        time_slot[name] = tt

init_state = None
init_out = None
state = None
mem1 = None
mem2 = None
out = None

DEBUG_DEBUG = True

def _sample_logits(
        out,
        x,
        ctx_len,
        temperature=1.0,
        top_p_usual=None,
        top_p_newline=None
    ):

    probs = F.softmax(torch.tensor(out), dim=-1)

    top_p = top_p_usual

    sorted_probs, s_index = torch.sort(probs, descending=True)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])

    probs[probs < cutoff] = 0

    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)

    return torch.multinomial(probs, num_samples=1)[0]
    

for TRIAL in range(1 if DEBUG_DEBUG else NUM_TRIALS):
    print(("-" * 50) + '\n' + context, end="")

    time_ref = time.time_ns()
    ctx = src_ctx.copy()

    if TRIAL == 0:
        for i in range(src_len):
            x = ctx[: i + 1]
            if i == src_len - 1:
                init_out, init_state, mem1, mem2 = model.forward(x, init_state, mem1, mem2)
            else:
                init_state, mem1, mem2 = model.forward(x, init_state, mem1, mem2, preprocess_only=True)
        gc.collect()
        torch.cuda.empty_cache()

    record_time('preprocess')
    out_last = src_len
    for i in range(src_len, src_len + (LENGTH_PER_TRIAL)):
        x = ctx[: i + 1]
        x = x[-ctx_len:]

        if i == src_len:
            out = init_out.clone()
            state = init_state.clone()
        else:
            out, state, mem1, mem2 = model.forward(x, state, mem1, mem2)

        ttt = _sample_logits(
            out,
            x,
            ctx_len,
            temperature=TEMPERATURE,
            top_p_usual=top_p,
            top_p_newline=top_p_newline,
        )
        ttt = int(ttt)
        ctx += [ttt]

        char = tokenizer.decode(ctx[out_last:])
        if '\ufffd' not in char: # is valid utf8 string?
            print(char, end="", flush=True)
            out_last = i+1


    record_time('total')
    # print(f'\n\n{time_slot}\n\n')
    print(
        f"\n\n--- preprocess {round(time_slot['preprocess'], 2)}s, generation {round(time_slot['total']-time_slot['preprocess'], 2)}s ", end = ''
    )

print(("-" * 50) + '\n')
