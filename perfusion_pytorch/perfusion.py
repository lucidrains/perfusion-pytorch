from math import ceil
from beartype import beartype
from beartype.typing import Union, List, Optional

import torch
from torch import nn, einsum, Tensor, IntTensor, LongTensor, FloatTensor
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange, reduce

from opt_einsum import contract as opt_einsum

from perfusion_pytorch.open_clip import OpenClipAdapter

# helpers

def exists(val):
    return val is not None

IndicesTensor = Union[LongTensor, IntTensor]

# function for calculating C - input covariance

@beartype
@torch.no_grad()
def calculate_input_covariance(
    clip: OpenClipAdapter,
    texts: List[str],
    batch_size = 32,
    **cov_kwargs
):
    embeds, mask = clip.embed_texts(texts)

    num_batches = ceil(len(texts) / batch_size)

    all_embeds = []

    length = len(texts)

    for batch_ind in range(num_batches):
        start_index = batch_ind * batch_size
        batch_texts = texts[start_index:(start_index + batch_size)]

        embeds, mask = clip.embed_texts(batch_texts)
        all_embeds.append(embeds[mask])

    all_embeds = torch.cat(all_embeds, dim = 0)

    return einsum('n d, n e -> d e', all_embeds, all_embeds) / length

@beartype
def find_first_index(
    indices: IndicesTensor,
    concept_or_superclass_id: int
):
    """
    for deriving the concept_indices to be passed into the Rank1EditModule
    """

    edge = (indices == concept_or_superclass_id).cumsum(dim = -1)  # [1, 3, 5, 4, 1, 1], 4 -> [0, 0, 0, 1, 0, 0, 0] -> [0, 0, 0, 1, 1, 1]
    return edge.sum(dim = -1)

@beartype
def return_text_enc_with_concept_and_superclass(
    text_ids: IndicesTensor,
    concept_id: int,
    superclass_id: int,
    clip: Optional[OpenClipAdapter] = None
):
    batch = text_ids.shape[0]
    batch_arange = torch.arange(batch, device = text_ids.device)
    concept_indices = find_first_index(text_ids, concept_id)
    text_ids_with_superclass = text_ids[batch_arange, concept_indices] = superclass_ids

    if not exists(clip):
        return text_ids, concept_indices, text_ids_with_superclass

    concept_text_enc = clip.embed_texts(text_ids)
    superclass_text_enc = clip.embed_texts(text_ids_with_superclass)

    return concept_text_enc, concept_indices, superclass_text_enc

# loss weighted by the mask

@beartype
def loss_fn_weighted_by_mask(
    pred: FloatTensor,
    target: FloatTensor,
    mask: FloatTensor,
    normalized_mask_min_value = 0.
):
    assert mask.shape[-2:] == pred.shape[-2:] == target.shape[-2:]
    assert mask.shape[0] == pred.shape[0] == target.shape[0]

    assert (mask.amin() >= 0.).all(), 'mask should not have values below 0'

    if mask.ndim == 4:
        assert mask.shape[1] == 1
        mask = rearrange(mask, 'b 1 h w -> b h w')

    loss = F.mse_loss(pred, target, reduction = 'none')
    loss = reduce(loss, 'b c h w -> b h w')

    # normalize mask by max

    normalized_mask = mask / mask.amax(dim = -1, keepdim = True).clamp(min = 1e-5)
    normalized_mask = normalized_mask.clamp(min = normalized_mask_min_value)

    loss = loss * normalized_mask

    return loss.mean()

# a module that wraps the keys and values projection of the cross attentions to text encodings

class Rank1EditModule(Module):

    @beartype
    def __init__(
        self,
        key_or_values_proj: nn.Linear,
        *,
        num_concepts: int = 1,
        C: Tensor,                         # covariance of input, precomputed from 100K laion text
        text_seq_len: int = 77,
        is_key_proj: bool = False,
        input_decay = 0.99,
        train_beta = 0.75,
        train_temperature = 0.1,
        eval_beta = 0.70,                  # in paper, specified a range (0.6 - 0.75) for local-key lock, and (0.4 -0.6) for global-key lock
        eval_temperature = 0.15,
        frac_gradient_concept_embed = 0.1  # they use a slower learning rate for the embed - this can be achieved by a trick to reduce the gradients going backwards through an operation
    ):
        super().__init__()
        assert not exists(key_or_values_proj.bias), 'key value projection in attention should not have bias'

        self.num_concepts = num_concepts

        self.weight = key_or_values_proj.weight
        dim_output, dim_input = self.weight.shape

        self.train_beta = train_beta
        self.train_temperature = train_temperature
        self.eval_beta = eval_beta
        self.eval_temperature = eval_temperature

        self.input_decay = input_decay

        self.text_seq_len = text_seq_len

        # for the lowered learning rate on the concept embed (0.006 vs 0.03 or something)

        assert 0 < frac_gradient_concept_embed <= 1.
        self.frac_gradient_concept_embed = frac_gradient_concept_embed

        # for exponentially smoothing the inputs
        # will smooth both concept and superclass token inputs

        self.register_buffer('initted', torch.zeros(num_concepts, 1).bool())
        self.register_buffer('ema_concept_text_encs', torch.zeros(num_concepts, dim_input))

        # concept outputs - only optimized for values, but not keys

        self.is_key_proj = is_key_proj # will lock the output to the super-class, and turn off gradients

        self.concept_output = nn.Parameter(torch.zeros(num_concepts, dim_output), requires_grad = not is_key_proj)

        # C in the paper, inverse precomputed

        self.register_buffer('C_inv', torch.inverse(C))

    def parameters(self):
        if not self.is_key_proj:
            return []

        return [self.concept_output]

    @beartype
    def forward(
        self,
        text_enc: FloatTensor,
        concept_indices: IndicesTensor,
        text_enc_with_superclass: Optional[FloatTensor] = None,
        concept_id: int = 0
    ):
        assert text_enc.shape[-2] == self.text_seq_len, f'CLIP text sequence length is set to be {self.text_seq_len}, but received text encoding with length {text_enc.shape[-2]}'

        """
        following the pseudocode of Algorithm 1 in appendix

        einstein notation:
        b - batch
        n - sequence
        d - feature dimension
        i - input dimension
        o - output dimension
        """

        batch, device = text_enc.shape[0], self.C_inv.device

        weights, decay, Ci = self.weight, self.input_decay, self.C_inv

        # reduce learning rate going back to the text encoder and into the concept embed

        text_enc = text_enc * self.frac_gradient_concept_embed + text_enc.detach() * (1 - self.frac_gradient_concept_embed)

        # beta and temperature depends on whether training or inference

        beta, temperature = (self.train_beta, self.train_temperature) if self.training else (self.eval_beta, self.eval_temperature)

        # extract the concept text encoding input

        batch_indices = torch.arange(batch, device = device)
        batch_indices = rearrange(batch_indices, 'b -> b 1')
        concept_indices = rearrange(concept_indices, 'b -> b 1')

        concept_text_enc = text_enc[batch_indices, concept_indices]
        concept_text_enc = reduce(concept_text_enc, 'b 1 d -> d', 'mean')

        # only if training
        # do exponential smoothing of the inputs, both concept and superclass

        if exists(text_enc_with_superclass):
            superclass_text_enc = text_enc_with_superclass[batch_indices, concept_indices]
            superclass_text_enc = reduce(superclass_text_enc, 'b 1 d -> d', 'mean')

            superclass_output = einsum('i, o i -> o', superclass_text_enc, weights)

        # get the initialization state

        assert concept_id < self.num_concepts

        initted = self.initted[concept_id].item()

        if self.training:
            # store the superclass i* if not all initialized
            # else fetch it from the buffer

            if not initted:
                assert exists(superclass_output), 'text_enc_with_superclass must be passed in for the first batch'

                # init concept output with superclass output - fixed for keys, learned for values
                self.concept_output[concept_id].data.copy_(superclass_output)

            elif exists(superclass_output) and self.is_key_proj:
                # if text enc with superclass is passed in for more than 1 batch
                # just take the opportunity to exponentially average it a bit more for the keys, which have fixed concept output (to superclass)

                ema_concept_output = self.concept_output * decay + superclass_output * (1. - decay)
                self.concept_output[concept_id].data.copy_(ema_concept_output)

            # if any in the batch is not initialized, initialize

            if not initted:
                ema_concept_text_enc = concept_text_enc
            else:
                ema_concept_text_enc = self.ema_concept_text_enc[concept_id]

            # exponential moving average for concept input encoding

            concept_text_enc = ema_concept_text_enc * decay + concept_text_enc * (1. - decay)

            # store

            if not initted:
                self.initted[concept_id].data.copy_(Tensor([True]))
                self.ema_concept_text_encs[concept_id].data.copy_(concept_text_enc)
        else:
            assert initted, 'you have not initialized or trained this module yet'

        # make it easier to match with paper

        i, o, W = self.ema_concept_text_encs[concept_id], self.concept_output[concept_id], weights

        # main contribution eq (3)

        i_energy = opt_einsum('o, o i, i ->', i, Ci, i)

        sim = opt_einsum('b n o, o i, i -> b n', text_enc, Ci, i)
        sim = rearrange(sim, '... -> ... 1')

        sigmoid_term = (((sim / i_energy) - beta) / temperature).sigmoid()

        text_enc_output = einsum('b n i, o i -> b n o', text_enc, W)

        concept_output = einsum('i, o i -> o', i, W)
        concept_output = rearrange(concept_output, 'd -> 1 1 d')

        W_em_orthogonal_term = text_enc_output - (sim * concept_output / i_energy)

        return W_em_orthogonal_term + sigmoid_term * rearrange(o, 'd -> 1 1 d')
