import torch
from torch import nn, einsum, Tensor
from torch.nn import Module
import torch.nn.functional as F

from beartype import beartype
from einops import rearrange

# helpers

def exists(val):
    return val is not None

# main contribution of paper
# a module that wraps the keys and values projection of the cross attentions to text encodings

class Rank1EditModule(Module):

    @beartype
    def __init__(
        self,
        key_or_values_proj: nn.Linear,
        *,
        C: Tensor,
        input_decay = 0.99,
        train_beta = 0.75,
        train_temperature = 0.1,
        eval_beta = 0.70,           # in paper, specified a range (0.6 - 0.75) for local-key lock, and (0.4 -0.6) for global-key lock
        eval_temperature = 0.15
    ):
        super().__init__()
        assert not exists(key_or_values_proj.bias), 'key value projection in attention should not have bias'

        self.weight = key_or_values_proj.weight

        self.train_beta = train_beta
        self.train_temperature = train_temperature
        self.eval_beta = eval_beta
        self.eval_temperature = eval_temperature

        self.input_decay = input_decay

        # buffers

        self.register_buffer('C_inv', torch.inverse(C))

    @beartype
    def forward(
        self,
        text_enc: Tensor,
        concept_indices: Tensor
    ):
        """
        following the pseudocode of Algorithm 1 in appendix
        """

        return text_enc
