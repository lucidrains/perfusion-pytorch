<img src="./key-locked-rank-1-editing.png" width="450px"></img>

## Perfusion - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2305.01644">Key-Locked Rank One Editing</a>. <a href="https://research.nvidia.com/labs/par/Perfusion/">Project page</a>

It seems they successfully applied the Rank-1 editing technique from a <a href="https://arxiv.org/abs/2202.05262">memory editing paper for LLM</a>, with a few improvements. They also identified that the keys determine the "where" of the new concept, while the values determine the "what", and propose local / global-key locking to a superclass concept (while learning the values).

## Appreciation

- <a href="https://stability.ai/">StabilityAI</a> for the generous sponsorship, as well as my other sponsors out there

## Install

```bash
$ pip install perfusion-pytorch
```

## Usage

```python
import torch
from perfusion_pytorch import Rank1EditModule
from torch import nn

to_keys = nn.Linear(512, 1024, bias = False)
to_values = nn.Linear(512, 1024, bias = False)

input_covariance = torch.randn(512, 512)

wrapped_to_keys = Rank1EditModule(
    to_keys,
    C = input_covariance,
    is_key_proj = True,
    num_finetune_prompts = 32
)

wrapped_to_values = Rank1EditModule(
    to_values,
    C = input_covariance,
    num_finetune_prompts = 32
)

text_enc = torch.randn(4, 256, 512)                  # regular input
text_enc_with_superclass = torch.randn(4, 256, 512)  # init_input in algorithm 1, for key-locking
concept_ids = torch.randint(0, 256, (4,))


keys = wrapped_to_keys(text_enc, text_enc_with_superclass, concept_ids)
values = wrapped_to_values(text_enc, text_enc_with_superclass, concept_ids)
```

## Todo

- [ ] take care of the function that takes in the dataset and text encoder and precomputes the covariance matrix needed for the rank-1 update
- [ ] instead of having the researcher worry about different learning rates, offer the fractional gradient trick from other paper (to learn the concept embedding)

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

