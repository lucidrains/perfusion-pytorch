from torch.nn import Module
from torch.optim import AdamW, Adam, Optimizer

from beartype import beartype

from perfusion_pytorch.embedding import EmbeddingWrapper
from perfusion_pytorch.perfusion import Rank1EditModule

# function that automatically finds all the parameters necessary for fine tuning

@beartype
def get_finetune_parameters(text_image_model: Module):
    params = []
    for module in text_image_model.modules():
        if isinstance(module, (EmbeddingWrapper, Rank1EditModule)):
            params.extend(module.parameters())

    return params

@beartype
def get_finetune_optimizer(
    text_image_model: Module,
    lr = 1e-4,
    wd = 1e-2,
    betas = (0.9, 0.99),
    eps = 1e-8,
    **kwargs
) -> Optimizer:
    params = get_finetune_parameters(text_image_model)

    assert len(params) > 0, 'no finetuneable parameters found'
    total_params = sum([p.numel() for p in params])
    print(f'optimizing {total_params} parameters')

    has_weight_decay = wd > 0
    adam_klass = AdamW if has_weight_decay else Adam
    adam_kwargs = dict(lr = lr, betas = betas, eps = eps)

    if has_weight_decay:
        adam_kwargs.update(weight_decay = wd)

    return adam_klass(params, **adam_kwargs, **kwargs)
