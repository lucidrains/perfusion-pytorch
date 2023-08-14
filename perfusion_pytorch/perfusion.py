from math import ceil
from beartype import beartype
from beartype.typing import Union, List, Optional

import torch
from torch import nn, einsum, Tensor, IntTensor, LongTensor, FloatTensor
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange

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
    open_clip: OpenClipAdapter,
    texts: List[str],
    batch_size = 32,
    **cov_kwargs
):
    embeds, mask = open_clip.embed_texts(texts)

    num_batches = ceil(len(texts) / batch_size)

    all_embeds = []

    for batch_ind in range(num_batches):
        start_index = batch_ind * batch_size
        batch_texts = texts[start_index:(start_index + batch_size)]

        embeds, mask = open_clip.embed_texts(batch_texts)
        all_embeds.append(embeds[mask])

    all_embeds = torch.cat((all_embeds), dim = 0)
    all_embeds = rearrange(all_embeds, 'n d -> d n')
    return torch.cov(all_embeds, **cov_kwargs)

# a module that wraps the keys and values projection of the cross attentions to text encodings

class Rank1EditModule(Module):

    @beartype
    def __init__(
        self,
        key_or_values_proj: nn.Linear,
        *,
        num_finetune_prompts: int,
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

        self.register_buffer('initted', torch.zeros(num_finetune_prompts).bool())
        self.register_buffer('ema_concept_text_encs', torch.zeros(num_finetune_prompts, dim_input))

        # superclass outputs - only optimized for values, but not keys

        self.is_key_proj = is_key_proj # will lock the output to the super-class, and turn off gradients

        self.superclass_outputs = nn.Parameter(torch.zeros(num_finetune_prompts, dim_output), requires_grad = not is_key_proj)

        # C in the paper, inverse precomputed

        self.register_buffer('C_inv', torch.inverse(C))

    def parameters(self):
        if not self.is_key_proj:
            return []

        return [self.superclass_outputs]

    @beartype
    def forward(
        self,
        text_enc: FloatTensor,
        text_enc_with_superclass: FloatTensor,
        concept_indices: IndicesTensor,
        *,
        prompt_ids: Optional[IndicesTensor] = None
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
        concept_text_enc = rearrange(concept_text_enc, 'b 1 d -> b d')

        # only if training, and if prompt ids are given
        # do exponential smoothing of the inputs, both concept and superclass

        if exists(text_enc_with_superclass):
            superclass_text_enc = text_enc_with_superclass[batch_indices, concept_indices]
            superclass_text_enc = rearrange(superclass_text_enc, 'b 1 d -> b d')

            superclass_output = einsum('b i, o i -> b o', superclass_text_enc, weights)

        if self.training and exists(prompt_ids):
            # get the initialization state
            # as well as the exponentially smoothed text encodings

            initted = self.initted[prompt_ids]
            all_initted = initted.all()

            ema_concept_text_enc = self.ema_concept_text_encs[prompt_ids]

            # store the superclass i* if not all initialized
            # else fetch it from the buffer

            if not all_initted:
                assert exists(superclass_output), 'text_enc_with_superclass must be passed in for the first epoch for all prompts to initialize the module correctly'

                non_initted_prompt_ids = prompt_ids[~initted]

                # for the prompt ids not initialized yet, hard copy over the initial superclass outputs
                self.superclass_outputs[non_initted_prompt_ids].data.copy_(superclass_output)

            superclass_output = self.superclass_outputs[prompt_ids]

            # if any in the batch is not initialized, initialize

            if not all_initted:
                ema_concept_text_enc = torch.where(
                    rearrange(initted, 'b -> b 1'),
                    ema_concept_text_enc,
                    concept_text_enc
                )

            # exponential moving average for concept input encoding

            concept_text_enc = ema_concept_text_enc * decay + concept_text_enc * (1. - decay)

            # store

            if not all_initted:
                self.initted[prompt_ids] = True
                self.ema_concept_text_encs[prompt_ids] = ema_concept_text_enc

        # take care of the output
        # for the keys, make sure to turn off gradients as it is 'locked'

        if self.is_key_proj:
            superclass_output = superclass_output.detach()

        # make it easier to match with paper

        i, o, W = concept_text_enc, superclass_output, weights

        # main contribution eq (3)

        i_energy = opt_einsum('b o, o i, b i -> b', i, Ci, i)
        i_energy = rearrange(i_energy, '... -> ... 1 1')

        sim = opt_einsum('b n o, o i, b i -> b n', text_enc, Ci, i)
        sim = rearrange(sim, '... -> ... 1')

        sigmoid_term = (((sim / i_energy) - beta) / temperature).sigmoid()

        text_enc_output = einsum('b n i, o i -> b n o', text_enc, W)

        concept_output = einsum('b i, o i -> b o', i, W)
        concept_output = rearrange(concept_output, 'b d -> b 1 d')

        W_em_orthogonal_term = text_enc_output - (sim * concept_output / i_energy)

        return W_em_orthogonal_term + sigmoid_term * rearrange(o, 'b d -> b 1 d')
