

import torch
from tokenizers import Tokenizer
import numpy as np
from torch.utils.data.sampler import BatchSampler
from itertools import chain
from torch.nn.utils.rnn import pad_sequence


def print_packed_batch(
    batch: torch.Tensor,
    tokenizer: Tokenizer,
    eod_token_id: int,
    # pad_token_id: int = None,
    # print_stats: bool = False,
):
    """separates the sequences within elements of a batch. Useful for
        sanity checking
    Optionally, print out things like the amount of padding and EOD tokens used, number of docs
    """
    B, T = batch.shape
    device = batch.device
    for bidx in range(B):
        tokens = batch[bidx,:]
        eod_mask = tokens == eod_token_id
        delim_idx = eod_mask.nonzero(as_tuple=True)[0].to(device)
        split_idx = torch.cat((
            torch.tensor([-1]).to(device),
            delim_idx,
            torch.tensor([T]).to(device),
        ))
        split_seq = [
            tokens[split_idx[i]+1:split_idx[i+1]]
            for i in range(len(split_idx)-1)
        ]

        for seq in split_seq:
            print('\t'+tokenizer.decode(seq.tolist()))
        print('-'*10)


    print('='*10)


def get_packing_stats(
    batch: torch.Tensor,
    eod_token_id: int,
    pad_token_id: int,
):
    """
    returns the amount of pad and eod tokens in the batch,
        as well as the number of documents
    if an eod token is at the last pos of a batch, document count has to be decreased
    """

    B, T = batch.shape

    pad_tokens = (batch == pad_token_id).sum()
    eod_token_mask = batch == eod_token_id
    eod_tokens = eod_token_mask.sum()

    doc_count = eod_tokens + (eod_token_mask.sum(1) == 0).sum() # take into account truncated docs


    return pad_tokens, eod_tokens, doc_count


def print_packing_stats(
    batch: torch.Tensor,
    eod_token_id: int,
    pad_token_id: int,
):
    B, T = batch.shape
    total_tokens = B*T
    pad_tokens, eod_tokens, doc_count = get_packing_stats(batch, eod_token_id, pad_token_id)
    eff = 1-(pad_tokens+eod_tokens)/total_tokens # batch "space" utilization

    print(f'pad: {pad_tokens}')
    print(f'eod: {eod_tokens}')
    print(f'doc: {doc_count}')
    print(f'total tokens: {total_tokens}')
    print(f'packing efficiency: {eff}')
    print('-'*10)

    return pad_tokens, eod_tokens, doc_count




# @numba.njit
def ffd_check(a: np.ndarray, c: int, n: int):
    # First-fit-decreasing bin packing
    # Check if a[] could fit in n bins with capacity c
    # https://en.wikipedia.org/wiki/First-fit-decreasing_bin_packing

    a = np.sort(a)[::-1]
    bins = np.full((n,), c, dtype=a.dtype)
    for size in a:
        not_found = True
        for idx in range(n):
            if bins[idx] >= size:
                bins[idx] -= size
                not_found = False
                break

        if not_found:
            return False

    return True

# @numba.njit
def ffd_with_result(a: np.ndarray, c: int, start_index: int):
    # First-fit-decreasing bin packing (with result return)

    indices = np.argsort(a)[::-1]
    a = a[indices]

    bins = []#: List[Any] = []
    bins_result = []#: List[Any] = []
    for a_id, size in enumerate(a):
        add_new = True
        for idx in range(len(bins)):
            if bins[idx] >= size:
                bins[idx] -= size
                bins_result[idx].append(indices[a_id] + start_index)
                add_new = False
                break

        if add_new:
            bins.append(c - size)
            bins_result.append([indices[a_id] + start_index])

    return bins_result

# @numba.njit
def allocate(
    lengths: np.ndarray, lengths_cumsum: np.ndarray, c: int, n: int,
):
    # Dynamic batch allocator, similar to Multifit
    # https://en.wikipedia.org/wiki/Multifit_algorithm
    # ~99.5% efficiency on OpenChat training set (12 * 2048 ctx len)

    s = 0
    start_index = 0
    result = []

    while True:
        # binary search [l, r)
        left = 1
        right = 1 + np.searchsorted(lengths_cumsum[start_index:], s + c * n, "right")

        while right - left > 1:
            mid = (left + right) // 2
            if ffd_check(lengths[start_index : start_index + mid], c, n):
                left = mid
            else:
                right = mid

        batch = ffd_with_result(
            lengths[start_index : start_index + left], c, start_index
        )

        start_index += left
        if start_index - 1 >= len(lengths):
            break
        s = lengths_cumsum[start_index - 1]

        result.extend(batch)

    return result



class SamplePackingBatchSampler(BatchSampler):
    def __init__(
        self,
        batch_size,
        ctx_len,
        group_size,
        lengths,
        random_sampler=None,
    ):
        self.batch_size = batch_size
        self.ctx_len = ctx_len
        self.group_size = group_size
        self.lengths = lengths
        self.random_sampler = random_sampler
        self.shuffle_idxs = None

        self.batches = self.generate_batches()

    def generate_batches(self):
        """

        returns: List[List[List[int]]]
            - the outer list is the batch dimension
        """

        if self.random_sampler:
            self.shuffle_idxs = [idx for idx in self.random_sampler]
            lengths = self.lengths[self.shuffle_idxs]
        else:
            lengths = self.lengths
        
        lengths_cumsum = np.cumsum(lengths)
        batches = allocate(
            lengths=lengths,
            lengths_cumsum=lengths_cumsum,
            c=self.ctx_len,
            n=self.group_size,
        )

        # map the shuffled indices back to the original ones
        if self.random_sampler:
            _batches = [[] for _ in range(len(batches))]
            for i, batch in enumerate(batches):
                _batches[i] = np.array(self.shuffle_idxs)[batch].tolist()
            batches = _batches

        #group into batches
        _batches = [
            batches[i:i+self.batch_size]
            for i in range(0, len(batches), self.batch_size)
        ]

        return _batches

    def __iter__(self):
        if self.random_sampler: # no need to recalc if deterministic
            self.batches = self.generate_batches()
        self.iterator = iter(self.batches)
        return self
    
    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            raise StopIteration

    def __len__(self):
        """
        NOTE: this can change if a random_sampler was provided
            __init__ and __iter__ recreate the batch packing,
            so len() is only consistent within a single epoch.
            If len() is called outside the epoch, it will report
            a slightly inaccurate size
            (though it isn't expected to change much)
        """
        return len(self.batches)
    


def collate_fn_base(data, eod_token_id, pad_token_id, ctx_len):
    """
    since we expect the arrow dataset's __getitems__ to be set to None,
    this line should be executed inside torch.utils.data.fetch (_MapDatasetFetcher):

        data = [self.dataset[idx] for idx in possibly_batched_index]
    
    data: List[Dict[str,List[List[int]]]]
        - list of batched examples
    """

    inputs = pad_sequence([
        torch.tensor(
            list(chain.from_iterable(doc[:-1] + [eod_token_id] for doc in pack['input_ids']))
        )[:ctx_len]
        for pack in data
    ], batch_first=True, padding_value=pad_token_id)

    targets = pad_sequence([
        torch.tensor(
            list(chain.from_iterable(doc[1:] + [eod_token_id] for doc in pack['input_ids']))
        )[:ctx_len]
        for pack in data
    ], batch_first=True, padding_value=pad_token_id)

    return inputs, targets