import torch
from torch import nn, Tensor
from torch.nn import Module

from collections import namedtuple

from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Optional, Tuple, Union, Callable, List

from einops import rearrange

from open_clip import tokenizer

# constants

EmbeddingReturn = namedtuple('EmbeddingReturn', [
    'embed_with_concept',
    'embed_with_superclass',
    'embed_mask',
    'concept_indices'
])

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def is_all_unique(arr):
    return len(set(arr)) == len(arr)

def filter_tuple_indices(tup, indices):
    return tuple(tup[i] for i in indices)

@beartype
def get_mask(
    x: Tensor,
    ids: Tuple[int, ...]
):
    masks = tuple(x == i for i in ids)
    mask, *rest_masks = masks

    for rest_mask in rest_masks:
        mask = mask | rest_mask

    return mask

# embedding wrapper class

class EmbeddingWrapper(Module):

    @beartype
    def __init__(
        self,
        embed: nn.Embedding,
        num_concepts = 1,
        superclass_embed_id: Optional[Union[int, Tuple[int, ...]]] = None,
        superclass_string: Optional[str] = None,
        tokenize: Callable[[List[str]], Tensor] = tokenizer.tokenize,
        tokenizer_pad_id: int = 0,
        tokenizer_sos_eos_id: Tuple[int, int] = (49406, 49407)
    ):
        super().__init__()
        self.embed = embed
        num_embeds, dim = embed.weight.shape

        self.num_embeds = num_embeds
        self.num_concepts = num_concepts
        self.concepts = nn.Parameter(torch.zeros(num_concepts, dim))

        assert not (exists(superclass_embed_id) and exists(superclass_string)), 'either superclass embed id is given, or the superclass string'

        self.pad_id = tokenizer_pad_id
        self.tokenize = None

        if exists(superclass_string):
            self.tokenize = tokenize

            ids = tokenize([superclass_string])[0]

            mask_for_ids = get_mask(ids, (tokenizer_pad_id, *tokenizer_sos_eos_id))
            ids = ids[~mask_for_ids]

            assert ids.shape[-1] == 1, f'your superclass concept string must map exactly one token id'
            superclass_embed_id = ids[0].item()

            print(f'super class embed for "{superclass_string}"" set as {superclass_embed_id}')
            print(f'you can now pass in a list of strings containing superclass concept, and this wrapper will return the embedding w/ concept and superclass required for finetuning')

        self.superclass_embed_id = superclass_embed_id

        assert not (exists(superclass_embed_id) and num_concepts > 1), 'cannot do multi concept with superclass embed id given'

        if exists(superclass_embed_id):
            # author had better results initializing the concept embed to the super class embed, allow for that option

            if not isinstance(superclass_embed_id, tuple):
                superclass_embed_id = (superclass_embed_id,)

            superclass_embed_indices = torch.tensor(list(superclass_embed_id))
            superclass_embeds = embed(superclass_embed_indices)
            self.concepts.data.copy_(superclass_embeds)
        else:
            # otherwise initialize to usually small init for embeds

            nn.init.normal_(self.concepts, std = 0.02)

        self.concept_embed_ids = tuple(range(num_embeds, num_embeds + num_concepts))

    def parameters(self):
        return [self.concepts]

    @property
    def device(self):
        return self.concepts.device

    @beartype
    def forward(
        self,
        x: Union[Tensor, List[str]],
        concept_id: Optional[Union[int, Tuple[int, ...]]] = None,
        return_embed_with_superclass = True,
        clip_transformer_fn: Optional[Callable[[Tensor], Tensor]] = None
    ) -> EmbeddingReturn:

        assert not (self.training and self.num_concepts > 1), 'cannot train with multiple concepts'

        if self.training:
            concept_id = default(concept_id, 0)

        if exists(concept_id):
            if not isinstance(concept_id, tuple):
                concept_id = (concept_id,)

            assert not self.training or len(concept_id) == 1, 'can only train or inference on single concepts if passing in list of superclass strings'
            assert not self.training or self.num_concepts == 1

        if is_bearable(x, List[str]):
            inferred_concept_id = self.concept_embed_ids[0]

            x = self.tokenize(x)
            x = x.to(self.device)

            superclass_mask = x == self.superclass_embed_id
            assert superclass_mask.any(dim = -1).all(), 'superclass embed id must be present for all prompts'

            # automatically replace the superclass id with the concept id

            x = torch.where(superclass_mask, inferred_concept_id, x)

        # get the embedding mask, defined as not padding id
        # default to open clip tokenizer padding id of 0

        embed_mask = x != self.pad_id

        # get masks for all concepts (support for multi-concepts)

        concept_masks = tuple(concept_id == x for concept_id in self.concept_embed_ids)

        if exists(concept_id):
            assert is_all_unique(concept_id), 'concept ids must be all unique'
            assert all([cid < self.num_concepts for cid in concept_id])

            has_concept = tuple(concept_mask.any(dim = -1).all() for concept_mask in concept_masks)

            assert all(filter_tuple_indices(has_concept, concept_id)), f'concept ids {filter_tuple_indices(self.concept_embed_ids, concept_id)} not found in ids passed in'
            concept_masks = filter_tuple_indices(concept_masks, concept_id)

        # just fetch the first embedding, as the concept embeddings are kept external to nn.Embedding

        for concept_mask in concept_masks:
            x = x.masked_fill(concept_mask, 0)

        # get all the embeddings that are not the concept or superclass concept
        # rest of embeddings are also not learnable, only concept embedding

        with torch.no_grad():
            embeds = self.embed(x)
            embeds.detach_()

        # substitute the concept back into the embeddings

        for concept, concept_mask in zip(self.concepts, concept_masks):
            embeds = torch.where(
                rearrange(concept_mask, '... -> ... 1'),
                concept,
                embeds
            )

        # whether to return concept indices for the rank-1-edit modules

        concept_indices = None

        if self.training and exists(concept_id) and len(concept_id) == 1:
            concept_mask, = concept_masks
            concept_indices = (concept_mask.cumsum(dim = -1) == 0).sum(dim = -1).long()

        # if training, and superclass embed id given
        # also return embeddings with superclass, for deriving superclass_text_enc

        superclass_embeds = None

        if self.training and exists(self.superclass_embed_id) and return_embed_with_superclass:
            x = x.masked_fill(concept_masks[0], self.superclass_embed_id)

            with torch.no_grad():
                superclass_embeds = self.embed(x)

        # if the clip transformer function is passed in, transform the embeds and superclass_embeds into the text_enc and superclass_text_enc, to be forwarded by cross attentions into the Rank1EditModules

        if exists(clip_transformer_fn):
            with torch.no_grad():
                embeds = clip_transformer_fn(embeds)

                if exists(superclass_embeds):
                    superclass_embeds = clip_transformer_fn(superclass_embeds)

        # return tuple, with
        # 1. text embeds | encodings
        # 2. superclass text embeds | encoding
        # 3. text mask
        # 4. concept indices

        return EmbeddingReturn(embeds, superclass_embeds, embed_mask, concept_indices)

# a wrapper for clip
# that automatically wraps the token embedding with new concept
# and on forward, passes the concept embeddings + superclass concept embeddings through the text transformer + final layernorm
# as well as make the forward pass the ids and superclass_ids through the modified text encoder twice (will attempt to substitute the nn.Embedding with an nn.Identity)

class OpenClipEmbedWrapper(Module):
    @beartype
    def __init__(
        self,
        clip: Module,
        text_transformer_path = 'transformer',
        ln_final_path = 'ln_final',  # in CLIP, they had the final layernorm separate from the transformer
        **embedding_wrapper_kwargs
    ):
        super().__init__()
        self.wrapped_embed = EmbeddingWrapper(clip.token_embedding, **embedding_wrapper_kwargs)

        path_to_modules = dict([(path, mod) for path, mod in clip.named_modules()])

        assert text_transformer_path in path_to_modules

        text_transformer = path_to_modules[text_transformer_path]
        ln_final = path_to_modules.get(ln_final_path, nn.Identity())

        self.text_transformer = nn.Sequential(
            text_transformer,
            ln_final
        )

    def forward(
        self,
        x,
        **kwargs
    ) -> EmbeddingWrapper:
        text_embeds, superclass_text_embeds, text_mask, concept_indices = self.wrapped_embed(x, **kwargs)

        text_enc = self.text_transformer(text_embeds)

        superclass_text_enc = None

        if exists(superclass_text_embeds):
            superclass_text_enc = self.text_transformer(superclass_text_embeds)

        return EmbeddingReturn(text_enc, superclass_text_enc, text_mask, concept_indices)

# merging multiple embedding wrappers (with one concepts) into a merged embedding wrapper with multiple concepts

@beartype
def merge_embedding_wrappers(
    *embeds: EmbeddingWrapper
) -> EmbeddingWrapper:

    total_concepts = sum([embed.num_concepts for embed in embeds])

    assert len(set([tuple(embed.embed.weight.shape) for embed in embeds])) == 1

    embed = embeds[0].embed

    merged_concepts = EmbeddingWrapper(
        embed = embed,
        num_concepts = total_concepts
    )

    merged_concepts.eval()

    concepts = torch.cat(tuple(embed.concepts.data for embed in embeds), dim = 0)

    merged_concepts.concepts = nn.Parameter(concepts)

    return merged_concepts
