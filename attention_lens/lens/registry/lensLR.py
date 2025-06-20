import torch
import torch.nn as nn
import loralib as lora

from typing import Optional

from attention_lens.lens.base import Lens


class LensLR(Lens):
    def __init__(
        self,
        unembed: nn.Parameter,
        bias: nn.Parameter,
        n_layers: int,
        d_model: int,
        d_vocab: int,
        r: int = 11,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        super().__init__(
            unembed,
            bias,
            n_layers,
            d_model,
            d_vocab,
            r=8,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        # Create a single shared parameter that holds the transpose of unembed (frozen).
        #self.shared_unembed = nn.Parameter(self.unembed.clone().t(), requires_grad=False)
        #self.shared_unembed = self.unembed.t() # Create a reference to original unembedding matrix, avoiding duplication. 

        # Create LoRA-enhanced Linear layers for each head
        self.linears = nn.ModuleList(
            [
                lora.Linear(
                    in_features=self.d_model,
                    out_features=self.d_vocab,
                    r=11,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                )
                for _ in range(self.n_layers)
            ]
        )

        print("LoRA Approximation Rank set to: {self.r}")

        # Replace each linear's original weight with the single shared_unembed
        # and initialize its bias from the original bias
        for linear in self.linears:
            del linear.weight
            # detach so no grad, transpose and make contiguous
            w = self.unembed.detach().t().contiguous()
            linear.register_buffer("weight", w)
            linear.bias.data = self.bias.data.clone()

    # def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
    #     """
    #     Args:
    #         input_tensor (torch.Tensor): shape (batch_size, pos, n_layers, d_model)

    #     Returns:
    #         torch.Tensor: shape (batch_size, pos, d_vocab), sum of outputs
    #                       from all attention heads.
    #     """

    #     batch_size, pos, n_layers, d_model = input_tensor.size()
    #     assert n_layers == self.n_layers, "Number of layers in input does not match LensLR."

    #     # Accumulate outputs over all heads
    #     output_tensors = torch.zeros(
    #         (batch_size, pos, self.d_vocab), device=input_tensor.device
    #     )

    #     for i in range(n_layers):
    #         input_head = input_tensor[:, :, i, :]        # [batch_size, pos, d_model]
    #         input_flat = input_head.reshape(-1, d_model) # [batch_size * pos, d_model]
    #         output_flat = self.linears[i](input_flat)    # [batch_size * pos, d_vocab]
    #         output_head = output_flat.view(batch_size, pos, self.d_vocab)
    #         output_tensors += output_head

    #     return output_tensors

    def forward(
        self,
        input_tensor: torch.Tensor,      # [B, S, n_layers, D]
        mask: Optional[torch.Tensor] = None,  # [B, S], 1 for real tokens, 0 for pad
    ) -> torch.Tensor:                  # returns [B, S, V]
        B, S, n_layers, D = input_tensor.size()
        V = self.bias.numel()  # vocab size

        # If no mask, fall back to original full-compute:
        if mask is None:
            output = torch.zeros((B, S, V), device=input_tensor.device)
            for i in range(n_layers):
                head_i = input_tensor[:, :, i, :]            # [B, S, D]
                flat = head_i.reshape(-1, D)                # [B·S, D]
                out = self.linears[i](flat)                 # [B·S, V]
                output += out.view(B, S, V)
            return output

        # 1) flatten batch & seq, select real tokens only
        flat_cache = input_tensor.view(-1, n_layers, D)         # [B·S, n_layers, D]
        flat_mask  = mask.view(-1).bool()                       # [B·S]
        valid_cache = flat_cache[flat_mask]                     # [N, n_layers, D]

        # 2) run only on real tokens
        #    accumulate layer-wise
        valid_out = torch.zeros((valid_cache.size(0), V), device=input_tensor.device)
        for i in range(n_layers):
            head_i = valid_cache[:, i, :]                       # [N, D]
            valid_out += self.linears[i](head_i)                # [N, V]

        # 3) scatter back into full [B·S, V], zeros for pads
        flat_out = torch.zeros((B * S, V), device=input_tensor.device)
        flat_out[flat_mask] = valid_out                        # pads stay 0

        # 4) reshape to [B, S, V]
        return flat_out.view(B, S, V)