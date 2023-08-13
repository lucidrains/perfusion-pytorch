import torch
from torch import nn, einsum, Tensor
from torch.nn import Module
import torch.nn.functional as F

from beartype import beartype
from einops import rearrange

# helpers

def exists(val):
    return val is not None

# a module that wraps the keys and values projection of the cross attentions to text encodings

class Rank1EditModule(Module):

    @beartype
    def __init__(
        self,
        key_or_values_proj: nn.Linear,
        *,
        num_finetune_prompts: int,
        C: Tensor,                  # covariance of input, precomputed from 100K laion text
        text_seq_len: int = 256,
        is_key_proj: bool = False,
        input_decay = 0.99,
        train_beta = 0.75,
        train_temperature = 0.1,
        eval_beta = 0.70,           # in paper, specified a range (0.6 - 0.75) for local-key lock, and (0.4 -0.6) for global-key lock
        eval_temperature = 0.15
    ):
        super().__init__()
        assert not exists(key_or_values_proj.bias), 'key value projection in attention should not have bias'

        self.weight = key_or_values_proj.weight
        dim_output, dim_input = self.weight.shape

        self.is_key_proj = is_key_proj # will lock the output to the super-class, and turn off gradients

        self.train_beta = train_beta
        self.train_temperature = train_temperature
        self.eval_beta = eval_beta
        self.eval_temperature = eval_temperature

        self.input_decay = input_decay

        self.text_seq_len = text_seq_len

        # C in the paper, inverse precomputed

        self.register_buffer('C_inv', torch.inverse(C))

    @beartype
    def forward(
        self,
        text_enc: Tensor,
        text_enc_with_superclass: Tensor,
        concept_indices: Tensor
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

        # beta and temperature depends on whether training or inference

        beta, temperature = (self.train_beta, self.train_temperature) if self.training else (self.eval_beta, self.eval_temperature)

        # extract the concept text encoding input

        batch_indices = torch.arange(batch, device = device)
        batch_indices = rearrange(batch_indices, 'b -> b 1')
        concept_indices = rearrange(concept_indices, 'b -> b 1')

        concept_text_enc = text_enc[batch_indices, concept_indices]
        concept_text_enc = rearrange(concept_text_enc, 'b 1 d -> b d')

        superclass_text_enc = text_enc_with_superclass[batch_indices, concept_indices]
        superclass_text_enc = rearrange(superclass_text_enc, 'b 1 d -> b d')

        # take care of initializing with superclass prompt
        # for key-locking - this assumes stable diffusion was modified so text encoder takes in a prompt with both the <concept> as well as <superclass> - it seems this also has the limitation that <superclass> must be one token

        text_enc_with_superclass_output = einsum('b n i, o i -> b n o', text_enc_with_superclass, weights)

        if self.is_key_proj:
            text_enc_with_superclass_output = text_enc_with_superclass_output.detach()

        online_estimated_concept_enc = decay * superclass_text_enc + (1. - decay) * concept_text_enc

        # make it easier to match with paper

        i, o, W = online_estimated_concept_enc, text_enc_with_superclass_output, weights

        # main contribution eq (3)

        i_energy = einsum('b d, b d -> b', i @ Ci, i)
        i_energy = rearrange(i_energy, '... -> ... 1 1')

        sim = einsum('b n d, b d -> b n', text_enc, i @ Ci)
        sim = rearrange(sim, '... -> ... 1')

        sigmoid_term = (((sim / i_energy) - beta) / temperature).sigmoid()

        orig_output = einsum('b n i, o i -> b n o', text_enc, W)

        concept_output = einsum('b i, o i -> b o', i, W)
        concept_output = rearrange(concept_output, 'b d -> b 1 d')

        W_em_orthogonal_term = orig_output - (sim * concept_output / i_energy)

        return W_em_orthogonal_term + sigmoid_term * o
