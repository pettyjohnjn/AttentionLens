import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_lens.lens.base import Lens
import math

class LensLR(Lens):
    def __init__(
        self,
        unembed: nn.Parameter,
        bias: nn.Parameter,
        n_layers: int,
        d_model: int,
        d_vocab: int,
        r: int = 8,
        lora_alpha: int = 1.0,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
    ):

        super().__init__(
            unembed,
            bias,
            n_layers,
            d_model,
            d_vocab,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.n_layers = n_layers
        self.d_model = d_model
        self.d_vocab = d_vocab
        self.r = r

        self.shared_unembed = unembed

        b_exp = bias.unsqueeze(0).expand(n_layers, -1).clone()
        self.bias = nn.Parameter(b_exp)

        self.W_A = nn.Parameter(torch.empty(n_layers, d_model, r, dtype=unembed.dtype))
        nn.init.kaiming_uniform_(self.W_A, a = math.sqrt(5))
        self.W_A.data.mul_(0.01)
        self.W_B = nn.Parameter(torch.empty(n_layers, r, d_vocab, dtype=unembed.dtype))
        nn.init.normal_(self.W_B, mean=0.0, std=0.02)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Use the buffer which should automatically be on the correct device.
        shared_unembed = self.shared_unembed

        batch_size, pos, n_layers, d_model = input_tensor.size()
        assert n_layers == self.n_layers, "Number of layers in input does not match LensLR."

        output_tensors = torch.zeros((batch_size, pos, self.d_vocab), device=input_tensor.device)

        for i in range(n_layers):
            input_layer = input_tensor[:, :, i, :].reshape(-1, d_model)
            logits_base = input_layer @ shared_unembed
            Z = input_layer @ self.W_A[i]
            logits_lr = Z @ self.W_B[i]
            logits = logits_base + logits_lr + self.bias[0]
            output_tensors += logits.view(batch_size, pos, self.d_vocab) / self.n_layers
        return output_tensors
