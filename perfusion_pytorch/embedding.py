import torch
from torch import nn
from torch.nn import Module

from beartype import beartype
from beartype.typing import Optional, Tuple, Union

from einops import rearrange

# helper functions

def exists(val):
    return val is not None

def is_all_unique(arr):
    return len(set(arr)) == len(arr)

def filter_tuple_indices(tup, indices):
    return tuple(tup[i] for i in indices)

# embedding wrapper class

class EmbeddingWrapper(Module):
    @beartype
    def __init__(
        self,
        embed: nn.Embedding,
        num_concepts = 1,
        superclass_embed_id: Optional[Union[int, Tuple[int, ...]]] = None
    ):
        super().__init__()
        self.embed = embed
        num_embeds, dim = embed.weight.shape

        self.num_embeds = num_embeds
        self.num_concepts = num_concepts
        self.concepts = nn.Parameter(torch.zeros(num_concepts, dim))

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

    def forward(
        self,
        x,
        concept_id: Optional[Union[int, Tuple[int, ...]]] = None,
        return_embed_with_superclass = True
    ):
        concept_masks = tuple(concept_id == x for concept_id in self.concept_embed_ids)

        if exists(concept_id):
            if not isinstance(concept_id, tuple):
                concept_id = (concept_id,)

            assert is_all_unique(concept_id), 'concept ids must be all unique'
            assert all([cid < self.num_concepts for cid in concept_id])

            has_concept = tuple(concept_mask.any(dim = -1).all() for concept_mask in concept_masks)

            assert all(filter_tuple_indices(has_concept, concept_id)), f'concept ids {filter_tuple_indices(self.concept_embed_ids, concept_id)} not found in ids passed in'
            concept_masks = filter_tuple_indices(concept_masks, concept_id)

        for concept_mask in concept_masks:
            x = x.masked_fill(concept_mask, 0)

        with torch.no_grad():
            embeds = self.embed(x)
            embeds.detach_()

        for concept, concept_mask in zip(self.concepts, concept_masks):
            embeds = torch.where(
                rearrange(concept_mask, '... -> ... 1'),
                concept,
                embeds
            )

        # if training, and superclass embed id given
        # also return embeddings with superclass, for deriving superclass_text_enc

        if self.training and exists(self.superclass_embed_id) and return_embed_with_superclass:
            x = x.masked_fill(concept_masks[0], self.superclass_embed_id)

            with torch.no_grad():
                superclass_embeds = self.embed(x)

            return embeds, superclass_embeds

        return embeds

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

    concepts = torch.cat(tuple(embed.concepts.data for embed in embeds), dim = 0)

    merged_concepts.concepts = nn.Parameter(concepts)

    return merged_concepts
