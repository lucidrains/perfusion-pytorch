<img src="./key-locked-rank-1-editing.png" width="450px"></img>

## Perfusion - Pytorch

Implementation of <a href="https://arxiv.org/abs/2305.01644">Key-Locked Rank One Editing</a>. <a href="https://research.nvidia.com/labs/par/Perfusion/">Project page</a>

The selling point of this paper is extremely low extra parameters per added concept, down to 100kb.

It seems they successfully applied the Rank-1 editing technique from a <a href="https://arxiv.org/abs/2202.05262">memory editing paper for LLM</a>, with a few improvements. They also identified that the keys determine the "where" of the new concept, while the values determine the "what", and propose local / global-key locking to a superclass concept (while learning the values).

For researchers out there, if this paper checks out, the tools in this repository should work for any other text-to-`<insert modality>` network using cross attention conditioning. Just a thought

## Appreciation

- <a href="https://stability.ai/">StabilityAI</a> for the generous sponsorship, as well as my other sponsors out there

- Yoad Tewel for the multiple code reviews and clarifying emails

- <a href="https://github.com/BradVidler">Brad Vidler</a> for precomputing the covariance matrix for the CLIP used in Stable Diffusion 1.5!

- All the maintainers at <a href="https://github.com/mlfoundations/open_clip">OpenClip</a>, for their SOTA open sourced contrastive learning text-image models

## Install

```bash
$ pip install perfusion-pytorch
```

## Usage

```python
import torch
from torch import nn

from perfusion_pytorch import Rank1EditModule

to_keys = nn.Linear(768, 320, bias = False)
to_values = nn.Linear(768, 320, bias = False)

wrapped_to_keys = Rank1EditModule(
    to_keys,
    is_key_proj = True
)

wrapped_to_values = Rank1EditModule(
    to_values
)

text_enc = torch.randn(4, 77, 768)                  # regular input
text_enc_with_superclass = torch.randn(4, 77, 768)  # init_input in algorithm 1, for key-locking
concept_indices = torch.randint(0, 77, (4,))

keys = wrapped_to_keys(
    text_enc,
    concept_indices = concept_indices,
    text_enc_with_superclass = text_enc_with_superclass,
)

values = wrapped_to_values(
    text_enc,
    concept_indices = concept_indices,
    text_enc_with_superclass = text_enc_with_superclass,
)

# after much training ...

wrapped_to_keys.eval()
wrapped_to_values.eval()

keys = wrapped_to_keys(text_enc)

values = wrapped_to_values(text_enc)

```

The repository also contains an `EmbeddingWrapper` that makes it easy to train on a new concept (and for eventual inference with multiple concepts)

```python
import torch
from torch import nn

from perfusion_pytorch import EmbeddingWrapper

embed = nn.Embedding(49407, 512) # open clip embedding, somewhere in the module tree of stable diffusion

# wrap it, and will automatically create a new concept for learning, based on the superclass embed string

wrapped_embed = EmbeddingWrapper(
    embed,
    superclass_string = 'dog'
)

# now just pass in your prompts with the superclass id

embeds_with_new_concept, embeds_with_superclass, embed_mask = wrapped_embed([
    'a portrait of dog',
    'dog running through a green field',
    'a man walking his dog'
]) # (3, 77, 512), (3, 77, 512), (3, 77)

# now pass both embeds through clip text transformer
# the embed_mask needs to be passed to the cross attention as key padding mask
```

## Todo

- [ ] wire up with SD 1.5, starting with xiao's dreambooth-sd
- [ ] show example in readme for inference with multiple concepts
- [ ] automatically infer where keys and values projection are if not specified for the `make_key_value_proj_rank1_edit_modules_` function

- [x] embedding wrapper should take care of substituting with super class token id and return embedding with super class
- [x] review multiple concepts - thanks to Yoad
- [x] offer a function that wires up the cross attention
- [x] handle multiple concepts in one prompt at inference - summation of the sigmoid term + outputs
    - [x] accept multiple concept indices
- [x] offer a way to combine separately learned concepts from multiple `Rank1EditModule` into one for inference
    - [x] offer function for merging `Rank1EditModule`s
- [x] add the zero-shot masking of concept proposed in paper
- [x] take care of the function that takes in the dataset and text encoder and precomputes the covariance matrix needed for the rank-1 update
- [x] instead of having the researcher worry about different learning rates, offer the fractional gradient trick from other paper (to learn the concept embedding)

## Citations

```bibtex
@article{Tewel2023KeyLockedRO,
    title   = {Key-Locked Rank One Editing for Text-to-Image Personalization},
    author  = {Yoad Tewel and Rinon Gal and Gal Chechik and Yuval Atzmon},
    journal = {ACM SIGGRAPH 2023 Conference Proceedings},
    year    = {2023},
    url     = {https://api.semanticscholar.org/CorpusID:258436985}
}
```

```bibtex
@inproceedings{Meng2022LocatingAE,
    title   = {Locating and Editing Factual Associations in GPT},
    author  = {Kevin Meng and David Bau and Alex Andonian and Yonatan Belinkov},
    booktitle = {Neural Information Processing Systems},
    year    = {2022},
    url     = {https://api.semanticscholar.org/CorpusID:255825985}
}
```

