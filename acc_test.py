

from src.model_debug import GPT
# from src.model_debug import GPTConfig

from deepspeed.ops.adam import FusedAdam
from types import SimpleNamespace

config = SimpleNamespace()
config.vocab_size = 32768
config.n_embd = 256
config.ctx_len = 128
config.n_layer = 6


model = GPT(
    config = config,
)

no_decay = set()
for mn, m in model.named_modules():  # here we disable weight_decay
    for pn, p in m.named_parameters():
        fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
        no_decay.add(fpn)

param_dict = {pn: p for pn, p in model.named_parameters()}
optim_groups = [
    {
        "params": [param_dict[pn] for pn in sorted(list(no_decay))],
        "weight_decay": 0.0
    },
]

optimizer = FusedAdam(
    optim_groups,
    lr=0.0006,
    betas=(0.9, 0.99),
    eps=0.0000000004,
    bias_correction=True,
    adam_w_mode=False,
    weight_decay=0,
    amsgrad=False,
)

