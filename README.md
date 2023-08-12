<img src="./key-locked-rank-1-editing.png" width="450px"></img>

## Perfusion - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2305.01644">Key-Locked Rank One Editing</a>. <a href="https://research.nvidia.com/labs/par/Perfusion/">Project page</a>

It seems they successfully applied the Rank-1 editing technique from a <a href="https://arxiv.org/abs/2202.05262">memory editing paper for LLM</a>, with a few improvements. They also identified that the keys determine the "where" of the new concept, while the values determine the "what", and propose local / global-key locking to a superclass concept (while learning the values).

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

