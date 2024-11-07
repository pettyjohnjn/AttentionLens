import torch
import torch.nn as nn
import math
from attention_lens.lens.base import Lens

class LensLR(Lens):
    """
    Low-Rank Lens (LensLR) approximates the linear transformation A as UV,
    where U is [d_model, rank] and V is [rank, d_vocab]. This reduces the
    number of parameters by leveraging the low-rank structure of A - W_U.

    Args:
        unembed (torch.Tensor): The unembedding matrix of shape [d_vocab, d_model].
        bias (torch.Tensor): The bias vector of shape [d_vocab].
        n_head (int): Number of attention heads.
        d_model (int): Dimension of the model.
        d_vocab (int): Size of the vocabulary.
        rank (int): The rank for the low-rank approximation.
    """
    
    def __init__(self, unembed, bias, n_head, d_model, d_vocab, rank=10):
        super().__init__(unembed, bias, n_head, d_model, d_vocab)
        self.rank = rank

        # Initialize U and V as separate ParameterLists for each head
        self.U = nn.ParameterList([
            nn.Parameter(torch.empty(d_model, rank)) for _ in range(n_head)
        ])
        self.V = nn.ParameterList([
            nn.Parameter(torch.empty(rank, d_vocab)) for _ in range(n_head)
        ])

        # Initialize U and V parameters with Xavier normal initialization
        for u, v in zip(self.U, self.V):
            nn.init.xavier_normal_(u)
            nn.init.xavier_normal_(v)

        # Initialize separate bias vectors for each head
        self.head_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(d_vocab)) for _ in range(n_head)
        ])

        # Register W_U as a buffer since it's not trainable
        # Ensure W_U is [d_model, d_vocab]
        self.register_buffer('W_U', unembed.clone())  # [d_model, d_vocab]

        # Print shapes for debugging
        print(f"W_U shape: {self.W_U.shape}")  # Should be [d_model, d_vocab]

    def forward(self, input_tensor):
        """
        Performs a forward pass through the LensLR model.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, pos, n_head, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, pos, d_vocab) after processing
                          through the low-rank linear transformations and summing across the
                          attention heads.
        """
        batch_size, pos, n_head, d_model = input_tensor.size()
        device = input_tensor.device

        # Ensure n_head matches
        assert n_head == self.n_head, f"Expected {self.n_head} heads, but got {n_head}."

        # Reshape input to [B*P, H, D]
        input_reshaped = input_tensor.view(batch_size * pos, n_head, d_model)  # [B*P, H, D]

        # Compute W_U contribution
        # W_U is [d_model, d_vocab], so input_reshaped @ W_U -> [B*P, H, d_vocab]
        # To ensure correctness, verify the shape
        # Uncomment the following line for debugging:
        # print(f"input_reshaped shape: {input_reshaped.shape}, W_U shape: {self.W_U.shape}")
        wu = torch.matmul(input_reshaped, self.W_U)  # [B*P, H, d_vocab]

        # Initialize UV contribution
        uv = torch.zeros_like(wu, device=device)  # [B*P, H, d_vocab]

        # Compute UV for each head
        for i in range(n_head):
            # input_head: [B*P, D]
            input_head = input_reshaped[:, i, :]  # [B*P, D]

            # Compute U @ V: [B*P, D] @ [D, R] = [B*P, R]
            # Then [B*P, R] @ [R, V] = [B*P, V]
            uv_head = torch.matmul(input_head, self.U[i])  # [B*P, R]
            uv_head = torch.matmul(uv_head, self.V[i])    # [B*P, V]

            # Assign to the UV tensor
            uv[:, i, :] = uv_head  # [B*P, H, d_vocab]

        # Compute total contribution: UV + W_U + bias
        # head_bias: [H, V] needs to be broadcasted to [B*P, H, V]
        bias = torch.stack(self.head_bias, dim=0)  # [H, V]
        bias = bias.unsqueeze(0).expand(batch_size * pos, -1, -1)  # [B*P, H, V]

        output = uv + wu + bias  # [B*P, H, V]

        # Sum across heads: [B*P, V]
        output = output.sum(dim=1)  # [B*P, V]

        # Reshape back to [batch_size, pos, d_vocab]
        output = output.view(batch_size, pos, self.d_vocab)  # [B, P, V]

        return output