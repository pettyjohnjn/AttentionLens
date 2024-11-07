import torch.nn as nn
import torch
from torch.nn.init import xavier_normal_

from attention_lens.lens.base import Lens

# Similar to LensA, LensLR treats the Attention Lens as a low rank perturbation on top of the unembedding matrix, significantly the number of trainable parameters.
class OldLensLR(Lens):
    def __init__(self, unembed, bias, n_head, d_model, d_vocab, rank):
        super().__init__(unembed, bias, n_head, d_model, d_vocab)
        self.linears = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_vocab) for _ in range(self.n_head)]
        )

        # Freeze the original unembedding matrix U
        for i in self.linears:
            i.weight = nn.Parameter(unembed.T.clone())
            i.bias = nn.Parameter(bias.clone())
            i.weight.requires_grad = False  # Freeze U

        self.rank = rank

        #L = A @ B
        # Initialize the low-rank perturbations A and B for each head
        self.As = nn.ParameterList([
            nn.Parameter(torch.empty(d_vocab, rank))
            for _ in range(n_head)
        ])
        self.Bs = nn.ParameterList([
            nn.Parameter(torch.empty(rank, d_model))
            for _ in range(n_head)
        ])

        # Apply Xavier uniform initialization to As and Bs
        for A in self.As:
            xavier_normal_(A)
        for B in self.Bs:
            xavier_normal_(B)


    def forward(self, input_tensor):
        """
        Performs a forward pass through the LensA model with low-rank perturbations.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, pos, n_head, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, pos, d_vocab) after processing through
            the linear layers and summing across the attention heads.
        """
        batch_size, pos, n_head, d_model = input_tensor.size()
        output_tensors = torch.empty(
            (batch_size, pos, self.n_head, self.d_vocab), device=input_tensor.device
        )

        for i in range(self.n_head):
            output_pos = torch.empty(
                (batch_size, pos, self.d_vocab), device=input_tensor.device
            )

            U = self.linears[i].weight  # [d_vocab, d_model]
            bias = self.linears[i].bias  # [d_vocab]
            L_i = self.As[i] @ self.Bs[i]  # [d_vocab, d_model]
            W_i = U + L_i  # [d_vocab, d_model]

            for j in range(pos):
                input_pos = input_tensor[:, j, i, :]  # [batch_size, d_model]
                input_reshaped = input_pos.reshape(batch_size, d_model)  # [batch_size, d_model]

                # Compute output_reshaped: [batch_size, d_vocab]
                output_reshaped = input_reshaped @ W_i.T + bias

                output_pos[:, j, :] = output_reshaped

            output_tensors[:, :, i, :] = output_pos

        summed_output = output_tensors.sum(dim=2)
        return summed_output