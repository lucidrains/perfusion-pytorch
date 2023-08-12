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
        num_finetune_prompts: int,
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
        dim_input = self.weight.shape[-1]

        self.train_beta = train_beta
        self.train_temperature = train_temperature
        self.eval_beta = eval_beta
        self.eval_temperature = eval_temperature

        self.input_decay = input_decay

        # they exponentially smooth the text encoding inputs during training
        # in addition to a lowered learning rate on the text encodings

        self.register_buffer('initted', torch.zeros(num_finetune_prompts).bool())
        self.register_buffer('ema_concept_text_enc', torch.zeros(num_finetune_prompts, dim_input))

        # buffers

        self.register_buffer('C_inv', torch.inverse(C))

    @beartype
    def forward(
        self,
        prompt_ids: Tensor,
        text_enc: Tensor,
        concept_indices: Tensor
    ):
        """
        following the pseudocode of Algorithm 1 in appendix
        """

        batch, device = text_enc.shape[0], self.initted.device

        weights, decay = self.weight, self.input_decay

        # beta and temperature depends on whether training or inference

        beta, temperature = (self.train_beta, self.train_temperature) if self.training else (self.eval_beta, self.eval_temperature)

        # extract the concept text encoding input

        batch_indices = torch.arange(batch, device = device)
        batch_indices = rearrange(batch_indices, 'b -> b 1')
        concept_indices = rearrange(concept_indices, 'b -> b 1')

        concept_text_enc = text_enc[batch_indices, concept_indices]
        concept_text_enc = rearrange(concept_text_enc, 'b 1 d -> b d')

        # during training, keep track of exponentially smoothed input

        if self.training:
            batch_initted = self.initted[prompt_ids]
            ema_concept_text_enc = self.ema_concept_text_enc[prompt_ids]

            ema_concept_text_enc = torch.where(
                rearrange(batch_initted, 'b -> b 1'),
                ema_concept_text_enc,
                concept_text_enc
            )

            # update using exponential moving average

            ema_concept_text_enc = ema_concept_text_enc * decay + concept_text_enc * (1. - decay)

            self.initted[prompt_ids] = True
            self.ema_concept_text_enc[prompt_ids] = ema_concept_text_enc

        return einsum('b n i, o i -> b n o', text_enc, weights)
