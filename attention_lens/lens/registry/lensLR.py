import torch.nn as nn
import torch
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
        r: int = 4,                # LoRA rank
        lora_alpha: int = 1,       # LoRA scaling factor
        lora_dropout: float = 0.0, # LoRA dropout rate
        merge_weights: bool = True, # Whether to merge weights during inference
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

        # Initialize LoRA-enhanced Linear layers using loralib correctly
        self.linears = nn.ModuleList(
            [
                lora.Linear(  # Correct usage
                    in_features=self.d_model,
                    out_features=self.d_vocab,
                    r=self.r,
                    lora_alpha=self.lora_alpha,    # Correct keyword argument
                    lora_dropout=self.lora_dropout, # Correct keyword argument
                )
                for _ in range(self.n_head)
            ]
        )

        # Initialize the Linear layers with W_U^T and bias, then freeze original weights
        for linear in self.linears:
            with torch.no_grad():
                # Initialize the main weights with W_U^T
                linear.weight_copy = self.unembed.data.clone().t()  # Shape [d_vocab, d_model]
                linear.weight.requires_grad = False  # Freeze original weights

                # Initialize bias
                linear.bias.data = self.bias.data.clone()
                # Bias is trainable by default in loralib's Linear

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the LensLR model.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, pos, n_head, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, pos, d_vocab) after processing through
                          the linear layers and summing across the attention heads.
        """
        batch_size, pos, n_head, d_model = input_tensor.size()
        assert n_head == self.n_head, "Number of heads in input does not match LensLR."

        # Initialize an output tensor
        output_tensors = torch.zeros(
            (batch_size, pos, self.d_vocab), device=input_tensor.device
        )

        # Iterate over each head and apply the corresponding LoRA Linear layer
        for i in range(n_head):
            # Extract the input for the i-th head: shape [batch_size, pos, d_model]
            input_head = input_tensor[:, :, i, :]  # Shape: [batch_size, pos, d_model]

            # Reshape to [batch_size * pos, d_model] for linear layer
            input_flat = input_head.reshape(-1, d_model)  # Shape: [batch_size * pos, d_model]

            # Apply the LoRA Linear layer: output_flat shape [batch_size * pos, d_vocab]
            output_flat = self.linears[i](input_flat)  # LoRA handles merged weights

            # Reshape back to [batch_size, pos, d_vocab]
            output_head = output_flat.view(batch_size, pos, self.d_vocab)

            # Accumulate the outputs
            output_tensors += output_head

        return output_tensors