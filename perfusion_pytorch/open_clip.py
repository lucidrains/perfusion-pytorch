from beartype import beartype
from beartype.typing import List, Optional

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

import open_clip

def exists(val):
    return val is not None

def l2norm(t):
    return F.normalize(t, dim = -1)

class OpenClipAdapter(nn.Module):
    @beartype
    def __init__(
        self,
        name = 'ViT-B/32',
        pretrained = 'laion400m_e32',
        tokenizer_name = 'ViT-B-32-quickgelu',
        eos_id = 49407
    ):
        super().__init__()

        clip, _, preprocess = open_clip.create_model_and_transforms(name, pretrained = pretrained)
        tokenizer = open_clip.get_tokenizer(tokenizer_name)

        self.clip = clip
        self.tokenizer = tokenizer
        self.eos_id = eos_id

        # hook for getting final text representation

        text_attention_final = self.find_layer('ln_final')
        self._dim_latent = text_attention_final.weight.shape[0]
        self.text_handle = text_attention_final.register_forward_hook(self._text_hook)

        # normalize fn

        self.clip_normalize = preprocess.transforms[-1]
        self.cleared = False

    @property
    def device(self):
        return next(self.parameters()).device

    def find_layer(self,  layer):
        modules = dict([*self.clip.named_modules()])
        return modules.get(layer, None)

    def clear(self):
        if self.cleared:
            return

        self.text_handle()

    def _text_hook(self, _, inputs, outputs):
        self.text_encodings = outputs

    @property
    def dim_latent(self):
        return self._dim_latent

    @property
    def max_text_len(self):
        return self.clip.positional_embedding.shape[0]

    @beartype
    def embed_texts(
        self,
        texts: List[str]
    ):
        ids = self.tokenizer(texts)
        ids = ids.to(self.device)
        ids = ids[..., :self.max_text_len]

        is_eos_id = (ids == self.eos_id)
        text_mask_excluding_eos = is_eos_id.cumsum(dim = -1) == 0
        text_mask = F.pad(text_mask_excluding_eos, (1, -1), value = True)
        text_mask = text_mask & (ids != 0)
        assert not self.cleared

        text_embed = self.clip.encode_text(ids)
        text_encodings = self.text_encodings
        text_encodings = text_encodings.masked_fill(~text_mask[..., None], 0.)
        return text_encodings.float(), text_mask
