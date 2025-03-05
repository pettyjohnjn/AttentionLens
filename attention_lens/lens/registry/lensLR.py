import torch
import torch.nn as nn
import loralib as lora

from attention_lens.lens.base import Lens


class LensLR(Lens):
    def __init__(
        self,
        unembed: nn.Parameter,
        bias: nn.Parameter,
        n_head: int,
        d_model: int,
        d_vocab: int,
        r: int = 8,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
    ):
        super().__init__(
            unembed,
            bias,
            n_head,
            d_model,
            d_vocab,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        # Create a single shared parameter that holds the transpose of unembed (frozen).
        self.shared_unembed = nn.Parameter(self.unembed.clone().t(), requires_grad=False)

        # Create LoRA-enhanced Linear layers for each head
        self.linears = nn.ModuleList(
            [
                lora.Linear(
                    in_features=self.d_model,
                    out_features=self.d_vocab,
                    r=8,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                )
                for _ in range(self.n_head)
            ]
        )

        # Replace each linear's original weight with the single shared_unembed
        # and initialize its bias from the original bias
        for linear in self.linears:
            del linear.weight
            linear.register_parameter("weight", self.shared_unembed)
            linear.bias.data = self.bias.data.clone()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tensor (torch.Tensor): shape (batch_size, pos, n_head, d_model)

        Returns:
            torch.Tensor: shape (batch_size, pos, d_vocab), sum of outputs
                          from all attention heads.
        """
        batch_size, pos, n_head, d_model = input_tensor.size()
        assert n_head == self.n_head, "Number of heads in input does not match LensLR."

        # Accumulate outputs over all heads
        output_tensors = torch.zeros(
            (batch_size, pos, self.d_vocab), device=input_tensor.device
        )

        for i in range(n_head):
            input_head = input_tensor[:, :, i, :]        # [batch_size, pos, d_model]
            input_flat = input_head.reshape(-1, d_model) # [batch_size * pos, d_model]
            output_flat = self.linears[i](input_flat)    # [batch_size * pos, d_vocab]
            output_head = output_flat.view(batch_size, pos, self.d_vocab)
            output_tensors += output_head

        return output_tensors