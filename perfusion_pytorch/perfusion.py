from math import ceil
from copy import deepcopy
from pathlib import Path

from beartype import beartype
from beartype.typing import Union, List, Optional, Tuple

import torch
from torch import nn, einsum, Tensor, IntTensor, LongTensor, FloatTensor
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange, reduce

from opt_einsum import contract as opt_einsum

from perfusion_pytorch.open_clip import OpenClipAdapter

# constants

IndicesTensor = Union[LongTensor, IntTensor]

# precomputed covariance paths
# will add for more models going forward, if the paper checks out

CURRENT_DIR = Path(__file__).parents[0]
DATA_DIR = CURRENT_DIR / 'data'

assert DATA_DIR.is_dir()

COVARIANCE_FILENAME_BY_TEXT_IMAGE_MODEL = dict(
    SD15 = DATA_DIR / 'covariance_CLIP_VIT-L-14.pt'
)

assert all([filepath.exists() for filepath in COVARIANCE_FILENAME_BY_TEXT_IMAGE_MODEL.values()])

# helpers

def exists(val):
    return val is not None

def is_all_unique(arr):
    return len(set(arr)) == len(arr)

# function for calculating C - input covariance

@beartype
@torch.no_grad()
def calculate_input_covariance(
    clip: OpenClipAdapter,
    texts: List[str],
    batch_size = 32,
    **cov_kwargs
):
    num_batches = ceil(len(texts) / batch_size)

    all_embeds = []

    length = len(texts)

    for batch_ind in range(num_batches):
        start_index = batch_ind * batch_size
        batch_texts = texts[start_index:(start_index + batch_size)]

        embeds, mask = clip.embed_texts(batch_texts)
        all_embeds.append(embeds[mask])

    all_embeds = torch.cat(all_embeds, dim = 0)

    return einsum('n d, n e -> d e', all_embeds, all_embeds) / length

@beartype
def find_first_index(
    indices: IndicesTensor,
    concept_or_superclass_id: int
):
    """
    for deriving the concept_indices to be passed into the Rank1EditModule
    """

    edge = (indices == concept_or_superclass_id).cumsum(dim = -1)  # [1, 3, 5, 4, 1, 1], 4 -> [0, 0, 0, 1, 0, 0, 0] -> [0, 0, 0, 1, 1, 1]
    return edge.sum(dim = -1)

@beartype
def return_text_enc_with_concept_and_superclass(
    text_ids: IndicesTensor,
    concept_id: int,
    superclass_id: int,
    clip: Optional[OpenClipAdapter] = None
):
    batch = text_ids.shape[0]
    batch_arange = torch.arange(batch, device = text_ids.device)
    concept_indices = find_first_index(text_ids, concept_id)
    text_ids_with_superclass = text_ids[batch_arange, concept_indices] = superclass_ids

    if not exists(clip):
        return text_ids, concept_indices, text_ids_with_superclass

    concept_text_enc = clip.embed_texts(text_ids)
    superclass_text_enc = clip.embed_texts(text_ids_with_superclass)

    return concept_text_enc, concept_indices, superclass_text_enc

# loss weighted by the mask

@beartype
def loss_fn_weighted_by_mask(
    pred: FloatTensor,
    target: FloatTensor,
    mask: FloatTensor,
    normalized_mask_min_value = 0.
):
    assert mask.shape[-2:] == pred.shape[-2:] == target.shape[-2:]
    assert mask.shape[0] == pred.shape[0] == target.shape[0]

    assert (mask.amin() >= 0.).all(), 'mask should not have values below 0'

    if mask.ndim == 4:
        assert mask.shape[1] == 1
        mask = rearrange(mask, 'b 1 h w -> b h w')

    loss = F.mse_loss(pred, target, reduction = 'none')
    loss = reduce(loss, 'b c h w -> b h w')

    # normalize mask by max

    normalized_mask = mask / mask.amax(dim = -1, keepdim = True).clamp(min = 1e-5)
    normalized_mask = normalized_mask.clamp(min = normalized_mask_min_value)

    loss = loss * normalized_mask

    return loss.mean()

# a module that wraps the keys and values projection of the cross attentions to text encodings

class Rank1EditModule(Module):

    @beartype
    def __init__(
        self,
        key_or_values_proj: nn.Linear,
        *,
        num_concepts: int = 1,
        C: Optional[Tensor] = None,          # covariance of input, precomputed from 100K laion text
        default_model = 'SD15',
        text_seq_len: int = 77,
        is_key_proj: bool = False,
        input_decay = 0.99,
        train_beta = 0.75,
        train_temperature = 0.1,
        eval_beta = 0.70,                    # in paper, specified a range (0.6 - 0.75) for local-key lock, and (0.4 -0.6) for global-key lock
        eval_temperature = 0.15,
        frac_gradient_concept_embed = 0.1,   # they use a slower learning rate for the embed - this can be achieved by a trick to reduce the gradients going backwards through an operation
        multi_concepts_use_cholesky = False  # use an approximated technique without Cholesky root for multiple concepts
    ):
        super().__init__()
        assert not exists(key_or_values_proj.bias), 'key value projection in attention should not have bias'

        self.num_concepts = num_concepts
        self.multi_concepts_use_cholesky = multi_concepts_use_cholesky

        self.weight = key_or_values_proj.weight
        dim_output, dim_input = self.weight.shape

        self.train_beta = train_beta
        self.train_temperature = train_temperature
        self.eval_beta = eval_beta
        self.eval_temperature = eval_temperature

        self.input_decay = input_decay

        self.text_seq_len = text_seq_len

        # for the lowered learning rate on the concept embed (0.006 vs 0.03 or something)

        assert 0 < frac_gradient_concept_embed <= 1.
        self.frac_gradient_concept_embed = frac_gradient_concept_embed

        # for exponentially smoothing the inputs
        # will smooth both concept and superclass token inputs

        self.register_buffer('initted', torch.zeros(num_concepts, 1).bool())
        self.register_buffer('ema_concept_text_encs', torch.zeros(num_concepts, dim_input))

        # concept outputs - only optimized for values, but not keys

        self.is_key_proj = is_key_proj # will lock the output to the super-class, and turn off gradients

        self.concept_outputs = nn.Parameter(torch.zeros(num_concepts, dim_output), requires_grad = not is_key_proj)

        # input covariance C in the paper, inverse precomputed
        # if covariance was not passed in, then use default for SD1.5, precomputed by @BradVidler

        if not exists(C):
            covariance_filepath = COVARIANCE_FILENAME_BY_TEXT_IMAGE_MODEL.get(default_model, None)

            assert exists(covariance_filepath), f'{default_model} not found in the list of precomputed covariances {tuple(COVARIANCE_FILENAME_BY_TEXT_IMAGE_MODEL.keys())}'

            C = torch.load(str(covariance_filepath))
            print(f'precomputed covariance loaded from {str(covariance_filepath)}')

        # calculate C_inv

        C_inv = torch.inverse(C)
        self.register_buffer('C_inv', C_inv)

    @property
    def num_concepts(self):
        return self._num_concepts

    @num_concepts.setter
    def num_concepts(self, value):
        self._num_concepts = value

        if value == 1 or not self.multi_concepts_use_cholesky:
            return

        # for multiple concepts
        # need cholesky decomposed L_t_inv
        # Appendix B

        try:
            L = torch.linalg.cholesky(self.C_inv)
        except:
            print('unable to perform cholesky. please make sure input covariance matrix is properly calculated')
            exit()

        L_T = L.T
        L_T_inv = torch.inverse(L_T)

        self.register_buffer('L_T', L_T, persistent = False)
        self.register_buffer('L_T_inv', L_T_inv, persistent = False)

    @property
    def device(self):
        return next(self.buffers()).device

    def parameters(self):
        if not self.is_key_proj:
            return []

        return [self.concept_outputs]

    @beartype
    def forward(
        self,
        text_enc: FloatTensor,
        *,
        concept_indices: Optional[IndicesTensor] = None,
        text_enc_with_superclass: Optional[FloatTensor] = None,
        concept_id: Union[int, Tuple[int, ...]] = 0
    ):
        assert text_enc.shape[-2] == self.text_seq_len, f'CLIP text sequence length is set to be {self.text_seq_len}, but received text encoding with length {text_enc.shape[-2]}'

        """
        following the pseudocode of Algorithm 1 in appendix

        einstein notation:
        b - batch
        n - sequence
        d - feature dimension
        i - input dimension
        o - output dimension
        c - concepts dimension (for multiple concepts)
        """

        batch, device = text_enc.shape[0], self.C_inv.device

        weights, decay, Ci = self.weight, self.input_decay, self.C_inv

        # reduce learning rate going back to the text encoder and into the concept embed

        text_enc = text_enc * self.frac_gradient_concept_embed + text_enc.detach() * (1 - self.frac_gradient_concept_embed)

        # beta and temperature depends on whether training or inference

        beta, temperature = (self.train_beta, self.train_temperature) if self.training else (self.eval_beta, self.eval_temperature)

        # determine whether it is single (for training) or multi-concept (only at inference)
        # may separate into different modules at a future date if too complex in one module

        is_multi_concepts = isinstance(concept_id, tuple)

        if is_multi_concepts:
            assert not self.training, 'multi concepts can only be done at inference'
            assert is_all_unique(concept_id)
            assert all([cid < self.num_concepts for cid in concept_id])

            concept_id_tensor = torch.tensor(concept_id, dtype = torch.long, device = self.device)
        else:
            assert concept_id < self.num_concepts
            concept_id_tensor = torch.tensor([concept_id], dtype = torch.long, device = self.device)

        # get the initialization state

        if self.training:
            initted = self.initted[concept_id].item()

            # extract the concept text encoding input

            batch_indices = torch.arange(batch, device = device)
            batch_indices = rearrange(batch_indices, 'b -> b 1')
            concept_indices = rearrange(concept_indices, 'b -> b 1')

            concept_text_enc = text_enc[batch_indices, concept_indices]
            concept_text_enc = reduce(concept_text_enc, 'b 1 d -> d', 'mean')

            # only if training
            # do exponential smoothing of the inputs, both concept and superclass

            if exists(text_enc_with_superclass):
                superclass_text_enc = text_enc_with_superclass[batch_indices, concept_indices]
                superclass_text_enc = reduce(superclass_text_enc, 'b 1 d -> d', 'mean')

                superclass_output = einsum('i, o i -> o', superclass_text_enc, weights)

            # store the superclass i* if not all initialized
            # else fetch it from the buffer

            if not initted:
                assert exists(superclass_output), 'text_enc_with_superclass must be passed in for the first batch'

                # init concept output with superclass output - fixed for keys, learned for values
                self.concept_outputs[concept_id].data.copy_(superclass_output)

            elif exists(superclass_output) and self.is_key_proj:
                # if text enc with superclass is passed in for more than 1 batch
                # just take the opportunity to exponentially average it a bit more for the keys, which have fixed concept output (to superclass)

                ema_concept_output = self.concept_outputs[concept_id] * decay + superclass_output * (1. - decay)
                self.concept_outputs[concept_id].data.copy_(ema_concept_output)

            # if any in the batch is not initialized, initialize

            if not initted:
                ema_concept_text_enc = concept_text_enc
            else:
                ema_concept_text_enc = self.ema_concept_text_encs[concept_id]

            # exponential moving average for concept input encoding

            ema_concept_text_enc = ema_concept_text_enc * decay + concept_text_enc * (1. - decay)

            # store

            if not initted:
                self.initted[concept_id].data.copy_(Tensor([True]))

            # update ema i_*

            self.ema_concept_text_encs[concept_id].data.copy_(ema_concept_text_enc)

        else:
            assert self.initted[concept_id_tensor].all(), 'you have not initialized or trained this module for the concepts id given'

        # make it easier to match with paper

        i, o, W = self.ema_concept_text_encs[concept_id_tensor], self.concept_outputs[concept_id_tensor], weights

        # if training, i* needs gradients. use straight-through?
        # check with author about this

        if self.training:
            i = i + concept_text_enc - concept_text_enc.detach()

        # main contribution eq (3)

        i_energy = opt_einsum('c o, o i, c i -> c', i, Ci, i)
        i_energy_inv = i_energy ** -1

        sim = opt_einsum('b n o, o i, c i -> c b n', text_enc, Ci, i)

        # calculate W_em_orthogonal_term - depends on single or multiple concepts

        if is_multi_concepts:
            if self.multi_concepts_use_cholesky:
                L_T, L_T_inv = self.L_T, self.L_T_inv

                # metric - metric space - variable with tilde in Appendix B

                # equation (6)

                i_metric = einsum('o i, c i -> c o', L_T, i)
                u_metric, _ = torch.linalg.qr(i_metric.T)
                u = einsum('o i, i c -> c o', L_T_inv, u_metric)

                # equation (10)

                em_orthogonal = text_enc - opt_einsum('c o, b n i, c i -> b n o', u, text_enc, u)

                W_em_orthogonal_term = einsum('b n i, o i -> b n o', em_orthogonal, W)
            else:
                # an approximated version, without Cholesky root
                # author says to use this preferentially, and fallback to Cholesky root if there are issues

                text_enc_output = einsum('b n i, o i -> b n o', text_enc, W)

                W_em_orthogonal_term = text_enc_output - opt_einsum('c b n, c i, o i, c -> b n o', sim, i, W, i_energy_inv)
        else:
            text_enc_output = einsum('b n i, o i -> b n o', text_enc, W)

            concept_output = einsum('c i, o i -> c o', i, W)

            W_em_orthogonal_term = text_enc_output - opt_einsum('c b n, c o, c -> b n o', sim, concept_output, i_energy_inv)

        # calculate sigmoid_term (gating)

        sim = rearrange(sim, 'c b n -> c b n 1')
        i_energy = rearrange(i_energy, 'c -> c 1 1 1')

        sigmoid_term = (((sim / i_energy) - beta) / temperature).sigmoid()

        gated_term = sigmoid_term * rearrange(o, 'c d -> c 1 1 d')
        gated_term = reduce(gated_term, 'c ... -> ...', 'sum')

        return W_em_orthogonal_term + gated_term

# for merging trained Rank1EditModule(s) above

@beartype
def merge_rank1_edit_modules(
    *modules: Rank1EditModule,
    use_cholesky = False
) -> Rank1EditModule:

    assert all([m.initted.all() for m in modules]), 'all modules must be initialized and ideally trained'
    assert len(set([m.concept_outputs.shape[-1] for m in modules])) == 1, 'concept output dimension must be the same'
    assert len(set([m.is_key_proj for m in modules])) == 1, 'all modules must be either for keys, or values. you cannot merge rank 1 edit modules of keys and values together'

    first_module = modules[0]
    merged_module = deepcopy(first_module)
    merged_module.multi_concepts_use_cholesky = use_cholesky

    total_concepts = sum([m.num_concepts for m in modules])
    merged_module.num_concepts = total_concepts

    concept_outputs = torch.cat(tuple(m.concept_outputs.data for m in modules), dim = 0)
    merged_module.concept_outputs = nn.Parameter(concept_outputs, requires_grad = not first_module.is_key_proj)

    ema_concept_text_encs = torch.cat(tuple(m.ema_concept_text_encs.data for m in modules), dim = 0)
    merged_module.register_buffer('ema_concept_text_encs',  ema_concept_text_encs)

    merged_module.register_buffer('initted', torch.ones(total_concepts, 1).bool())

    return merged_module

# function for wiring up the cross attention

@beartype
def make_key_value_proj_rank1_edit_modules_(
    cross_attention: nn.Module,
    *,
    input_covariance: Tensor,
    key_proj_name: str,
    value_proj_name: str,
    **rank1_edit_module_kwargs
):
    linear_key = getattr(cross_attention, key_proj_name, None)
    linear_values = getattr(cross_attention, value_proj_name, None)

    assert isinstance(linear_key, nn.Linear), f'{key_proj_name} must point to where the keys projection is (ex. self.to_keys = nn.Linear(in, out, bias = False) -> key_proj_name = "to_keys")'
    assert isinstance(linear_values, nn.Linear), f'{value_proj_name} must point to where the values projection is (ex. self.to_values = nn.Linear(in, out, bias = False) -> value_proj_name = "to_values")'

    rank1_edit_module_keys = Rank1EditModule(linear_key, input_covariance = input_covariance, is_key_proj = True, **rank1_edit_module_kwargs)
    rank1_edit_module_values = Rank1EditModule(linear_values, input_covariance = input_covariance, is_key_proj = False, **rank1_edit_module_kwargs)

    setattr(cross_attention, key_proj_name, rank1_edit_module_keys)
    setattr(cross_attention, value_proj_name, rank1_edit_module_values)
