from pathlib import Path

import torch
from torch import nn
from torch.nn import Module

from beartype import beartype

from perfusion_pytorch.embedding import EmbeddingWrapper
from perfusion_pytorch.perfusion import Rank1EditModule

def exists(val):
    return val is not None

@beartype
def save(
    text_image_model: Module,
    path: str
):
    path = Path(path)
    path.parents[0].mkdir(exist_ok = True, parents = True)

    embed_params = None
    key_value_params = []
    C_inv = None

    for module in text_image_model.modules():
        if isinstance(module, EmbeddingWrapper):
            assert not exists(embed_params), 'there should only be one wrapped EmbeddingWrapper'
            embed_params = module.concepts.data

        elif isinstance(module, Rank1EditModule):
            key_value_params.append([
                module.ema_concept_text_encs.data,
                module.concept_outputs.data
            ])

            C_inv = module.C_inv.data

    assert exists(C_inv), 'Rank1EditModule not found. you likely did not wire up the text to image model correctly'

    pkg = dict(
        embed_params = embed_params,
        key_value_params = key_value_params,
        C_inv = C_inv
    )

    torch.save(pkg, f'{str(path)}')
    print(f'saved to {str(path)}')

@beartype
def load(
    text_image_model: Module,
    path: str
):
    path = Path(path)
    assert path.exists(), f'file not found at {str(path)}'

    pkg = torch.load(str(path))

    embed_params = pkg['embed_params']
    key_value_params = pkg['key_value_params']
    C_inv = pkg['C_inv']

    for module in text_image_model.modules():
        if isinstance(module, EmbeddingWrapper):
            module.concepts.data.copy_(embed_params)

        elif isinstance(module, Rank1EditModule):
            assert len(key_value_params) > 0, 'mismatch between what was saved vs what is being loaded'
            concept_input, concept_output = key_value_params.pop(0)
            module.ema_concept_text_encs.data.copy_(concept_input)
            module.concept_outputs.data.copy_(concept_output)

            module.C_inv.copy_(C_inv)
            module.initted.copy_(torch.tensor([True]))

    print(f'loaded concept params from {str(path)}')
