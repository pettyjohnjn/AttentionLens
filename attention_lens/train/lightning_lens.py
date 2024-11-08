# lightning_lens.py

# -*- coding: utf-8 -*-
from __future__ import annotations

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import transformers

from attention_lens.lens import Lens
from attention_lens.model.get_model import get_model

import torch
import psutil
import loralib as lora  # Ensure loralib is imported if used elsewhere


def save_memory_usage():
    # Open the file in append mode
    with open("memory_usage.txt", "a") as f:
        # GPU memory usage
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            allocated = torch.cuda.memory_allocated(i) / 1024 ** 2
            cached = torch.cuda.memory_reserved(i) / 1024 ** 2
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024 ** 2
            max_cached = torch.cuda.max_memory_reserved(i) / 1024 ** 2

            # Write GPU memory info to the file
            if max_allocated > 0:
                f.write(f"GPU {i} Max Allocated: {max_allocated:.2f} MB\n")
            if max_cached > 0:
                f.write(f"GPU {i} Max Cached: {max_cached:.2f} MB\n")


class LightningLens(pl.LightningModule):
    def __init__(
        self,
        model_name: str,      # Name of the transformer model
        lens_cls: type[Lens] | str,  # Lens class or its string identifier
        layer_num: int,       # Layer number to hook
        lr: float = 1e-4,     # Learning rate
        rank: int = 8,        # LoRA rank (equivalent to 'r')
        **kwargs,             # Additional arguments (ensure they are not LoRA-specific)
    ):
        """
        Initialize the LightningLens module.

        Args:
            model_name (str): Name of the transformer model.
            lens_cls (type[Lens] | str): Lens class or its string identifier.
            layer_num (int): Layer number to hook.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            rank (int, optional): LoRA rank. Defaults to 8.
            **kwargs: Additional keyword arguments for LightningModule (ensure no LoRA-specific keys).
        """
        # Remove LoRA-specific parameters from kwargs before passing to super().__init__()
        # This prevents passing unexpected parameters to pl.LightningModule
        super().__init__()
        
        self.model_name = model_name
        self.layer_num = layer_num
        self.lr = lr
        self.rank = rank

        # Initialize the model and tokenizer
        self.model, self.tokenizer = get_model(
            model_name=self.model_name, device=self.device
        )

        # Handle lens_cls being a string or a class
        if isinstance(lens_cls, str):
            lens_cls = Lens.get_lens(lens_cls)
        elif not issubclass(lens_cls, Lens):
            raise ValueError(
                "Argument `lens_cls` must be a subclass of `Lens` or its string identifier."
            )

        # Handle bias initialization
        if self.model.lm_head.bias is None:
            self.bias = torch.zeros(self.model.config.vocab_size).to(self.device)
            # Alternatively, load from a file if needed
            # self.bias = torch.load('b_U.pt').to(self.device)
        else:
            self.bias = self.model.lm_head.bias

        # Handle weights initialization
        # If weights are loaded from a file, uncomment the following line
        # self.weights = torch.load('W_U.pt').to(self.device)
        self.weights = self.model.lm_head.weight.T  # Shape: [d_vocab, d_model]

        # Initialize the attention lens with LoRA
        self.attn_lens = lens_cls(
            unembed=self.weights,
            bias=self.bias,
            n_head=self.model.config.num_attention_heads,
            d_model=self.model.config.hidden_size,
            d_vocab=self.model.config.vocab_size,
            r=self.rank  # Pass LoRA rank here
        )

    def kl_loss(self, logits, lens_logits) -> torch.Tensor:
        r"""
        Compute the Kullback-Leibler divergence between tensors.

        Quantifies the difference between the probability distribution of the model's
        output versus the probability distribution of the attention lens.

        $$
            D_{KL} (\text{logits} \Vert \text{lens\_logits})
        $$

        Args:
            logits (torch.Tensor[d_vocab]): A probability distribution of the model's outputs.
            lens_logits (torch.Tensor[d_vocab]): The output of the AttentionLens model
                acting on the entire layer from the attention mechanism.

        Returns:
            loss: (torch.Tensor[bsz]): Returns difference between logits and lens_logits
        """

        kldiv = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        k_logits = F.log_softmax(logits[:, -1, :], dim=-1)  # Shape: [batch_size, d_vocab]
        k_lens_out = F.log_softmax(lens_logits[:, -1, :], dim=-1)  # Shape: [batch_size, d_vocab]

        loss = kldiv(k_lens_out, k_logits)
        return loss

    def setup(self, stage) -> None:
        """
        Sets up the model and tokenizer during training setup.

        Args:
            stage: The stage of the training process.
        """
        # Re-initialize the model and tokenizer on CPU to save GPU memory during setup
        self.model, self.tokenizer = get_model(
            model_name=self.model_name,
            device=torch.device("cpu"),
        )

    def forward(self, cache) -> torch.Tensor:
        """
        Compute a forward pass through the Attention Lens

        Takes the hook information of an entire layer of the attention mechanism, and
        computes the forward pass through that layer of Transformer Lens models.

        Args:

            cache (torch.Tensor[bsz, q_len, d_model]): The hooked information of an

                entire layer of the attention mechanism.

        Returns:
            lens_out (torch.Tensor[bsz, d_vocab]): The prediction of the attention lens
                models for that layer.
        """
        # The original code processes a list, but it's simpler to pass the tensor directly
        return self.attn_lens(cache)

    def training_step(self, train_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Defines a single step in the training loop. Takes in an entire batch and computes
        the KL-loss for that batch.

        Args:
            train_batch (torch.Tensor): The batch (bsz) of data for the current training
            step.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss for the current training step. 
        """

        prompt = train_batch["text"]
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Assuming you have a hook that stores 'head_out' for the specified layer
            # Modify this part based on how you access the cached outputs
            cache = self.model.transformer.h[self.layer_num].attn.head_out  # Shape: [batch_size, pos, d_model]
            logits = outputs.logits  # Shape: [batch_size, pos, d_vocab]

        lens_logits = self.forward(cache)  # Shape: [batch_size, d_vocab]
        loss = self.kl_loss(logits, lens_logits)
        self.log("train_loss", loss, prog_bar=True)

        save_memory_usage()
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer for training.

        Returns:
            torch.optim.Optimizer: The optimizer for training.
        """

        print(f'Learning Rate: {self.lr}')

        optimizer = torch.optim.Adam(self.attn_lens.parameters(), lr=self.lr)
        return optimizer

    # TODO(MS): register an early stopping call back which quits training if the loss/some metric drops below a certain point
    # TODO(MS): when training quits, save a copy of the appropriately named lens
    # TODO(MS): test and make sure distributed training works across nodes