import torch
from torch import nn
from torch.nn import Module
from beartype import beartype

from einops import rearrange

def exists(val):
    return val is not None

class EmbeddingWrapper(Module):
    @beartype
    def __init__(
        self,
        embed: nn.Embedding,
        num_concepts = 1
    ):
        super().__init__()
        self.embed = embed
        num_embeds, dim = embed.weight.shape

        self.num_concepts = num_concepts
        self.concepts = nn.Parameter(torch.randn(num_concepts, dim))
        self.concept_ids = tuple(range(num_embeds, num_embeds + num_concepts))

    def parameters(self):
        return [self.concepts]

    def forward(self, x):
        concept_masks = tuple(concept_id == x for concept_id in self.concept_ids)

        for concept_mask in concept_masks:
            x = x.masked_fill(concept_mask, 0)

        with torch.no_grad():
            embeds = self.embed(x)
            embeds.detach_()

        for concept_id, concept, concept_mask in zip(self.concept_ids, self.concepts, concept_masks):
            embeds = torch.where(
                rearrange(concept_mask, '... -> ... 1'),
                concept,
                embeds
            )

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
