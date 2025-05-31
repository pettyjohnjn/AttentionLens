import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_lens.lens.base import Lens
import math

class LensLR(Lens):
    def __init__(
        self,
        unembed: nn.Parameter,
        bias: nn.Paramter,
        n_layer: int,
        d_model: int,
        d_vocab: int,
        r: int = 8,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_merge_weights = merge_weights
    )