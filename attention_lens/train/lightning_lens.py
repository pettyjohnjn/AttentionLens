# lightning_lens.py

# -*- coding: utf-8 -*-
from __future__ import annotations

from functools import reduce

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
        lr: float = 1e-4,     # Learning rate
        r: int = 8,        # LoRA rank
        train_attn: bool = True,
        **kwargs,             # Additional arguments (ensure they are not LoRA-specific)
    ):
        """
        Initialize the LightningLens module.

        Args:
            model_name (str): Name of the transformer model.
            lens_cls (type[Lens] | str): Lens class or its string identifier.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            rank (int, optional): LoRA rank. Defaults to 8.
            **kwargs: Additional keyword arguments for LightningModule (ensure no LoRA-specific keys).
        """
        # Remove LoRA-specific parameters from kwargs before passing to super().__init__()
        # This prevents passing unexpected parameters to pl.LightningModule
        super().__init__(**kwargs)

        # flags
        self.train_attn = train_attn

        # core model
        self.model_name = model_name
        self.lr = lr
        self.r = r
        self.model, self.tokenizer = get_model(model_name=self.model_name, device=self.device)

        # shared unembed + bias
        unembed = self.model.lm_head.weight.T.clone().detach()
        self.register_buffer("shared_unembed", unembed)

        if self.model.lm_head.bias is not None:
            raw_bias = self.model.lm_head.bias.detach().clone()
        else:
            raw_bias = torch.zeros(self.model.config.vocab_size, device = self.device)

        # Resolve lens class
        if isinstance(lens_cls, str):
            lens_cls = Lens.get_lens(lens_cls)
        elif not issubclass(lens_cls, Lens):
            raise ValueError(
                "Argument `lens_cls` must be a subclass of `Lens` or its string identifier."
            )
        
        if self.train_attn:
            self.attn_cache = []
            self._register_attention_hooks()
            self.attn_lens = lens_cls(
                unembed=self.shared_unembed,
                bias=raw_bias,
                n_layers=self.model.config.num_hidden_layers,
                d_model=self.model.config.hidden_size,
                d_vocab=self.model.config.vocab_size,
                r=self.r
            )

    def on_train_start(self) -> None:
        # at this point Lightning/DeepSpeed has wrapped your model,
        # so hooks will stick to the real modules that actually run.
        if self.train_attn:
            # clear any spurious old handles
            for h in getattr(self, "_hook_handles", []):
                h.remove()
            self._hook_handles = []
            self._register_attention_hooks()

            self.attn_lens.shared_unembed = self.shared_unembed
            # sanity check
            if not self._hook_handles:
                raise RuntimeError("No attention hooks registered at train start!")


    def kl_loss(
        self,
        logits: torch.Tensor,
        lens_logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
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

        lengths = attention_mask.sum(dim=1).long()

        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device

        idx = (lengths - 1).clamp(min=0)

        batch_indices = torch.arange(batch_size, device=device)
        model_last_logits = logits[batch_indices, idx, :]
        lens_last_logits = lens_logits[batch_indices, idx, :]

        log_p_model = F.log_softmax(model_last_logits, dim = -1)
        log_p_lens = F.log_softmax(lens_last_logits, dim=-1)

        # print(f" Log_p_model: {log_p_model.shape}")
        # print(f" Log_p_lens: {log_p_lens.shape}")

        kldiv = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        loss = kldiv(log_p_lens, log_p_model)
        return loss



        # kldiv = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        # k_logits = F.log_softmax(logits[:, -1, :], dim=-1)  # Shape: [batch_size, d_vocab]
        # k_lens_out = F.log_softmax(lens_logits[:, -1, :], dim=-1)  # Shape: [batch_size, d_vocab]

        # loss = kldiv(k_lens_out, k_logits)
        # return loss

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
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        ).to(self.device)
        attention_mask = inputs["attention_mask"]

        # Clear Caches
        if self.train_attn: self.attn_cache.clear()

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            # Assuming you have a hook that stores 'head_out' for the specified layer
            # Modify this part based on how you access the cached outputs
            if self.train_attn: attn_cache = torch.stack(self.attn_cache, dim = 2)
            logits = outputs.logits  # Shape: [batch_size, pos, d_vocab]

        lens_logits = self.forward(attn_cache)  # Shape: [batch_size, d_vocab]
        loss = self.kl_loss(logits, lens_logits, attention_mask)
        self.log("train_loss", loss, prog_bar=True)

        # save_memory_usage()
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer for training.

        Returns:
            torch.optim.Optimizer: The optimizer for training.
        """

        print(f'Learning Rate: {self.lr}')

        no_decay, decay = set(), set()
        for n,p in self.attn_lens.named_parameters():
            (decay if "W_B" in n else no_decay).add(n)
        
        optimizer = torch.optim.AdamW([
            {"params":[p for n,p in self.attn_lens.named_parameters() if n in decay],
            "weight_decay":1e-2},
            {"params":[p for n,p in self.attn_lens.named_parameters() if n in no_decay],
            "weight_decay":0.0},
        ], lr=1e-4)

        # optimizer = torch.optim.Adam(self.attn_lens.parameters(), lr=self.lr)
        return optimizer

    # TODO(MS): register an early stopping call back which quits training if the loss/some metric drops below a certain point
    # TODO(MS): when training quits, save a copy of the appropriately named lens
    # TODO(MS): test and make sure distributed training works across nodes

    # Helper Functions

    def _get_attr_path(self, root, path: str):
        """
        Resolve a dotted attribute path (e.g. "transformer.h")
        on `root', or raise AttributeError if any step fails.
        """
        obj = root
        for name in path.split("."):
            obj = getattr(obj, name)
        return obj

    def _save_attn_output(self, module, input, output):
        """
        Hook callback to save the attention outputs.

        Args:
            module: The module for which the hook is registered.
            input: The input to the module.
            output: The output from the module.
        """
        # If output is a tuple, take its first element; otherwise, use the output directly.
        attn_out = output[0] if isinstance(output, tuple) else output
        # print(attn_out.shape)
        self.attn_cache.append(attn_out)

    def _register_attention_hooks(self):
        mtype = getattr(self.model.config, "model_type", None)

        # map model_type to (module_path, attn_attr)
        mapping = {
            "gpt2": ("transformer.h", "attn"),
            "llama": ("model.layers", "self_attn"),
        }

        handles: list[torch.utils.hooks.RemovableHandle] = []

        if mtype in mapping:
            module_path, attn_attr = mapping[mtype]

            try:
                layers = self._get_attr_path(self.model, module_path)
            except AttributeError:
                raise RuntimeError(
                    f"Found model_type={mtype!r} in mapping, but module path {module_path!r} "
                    "does not exist on this model."
                )

            for layer in layers:
                try:
                    attn_mod = reduce(getattr, attn_attr.split("."), layer)
                except AttributeError:
                    continue
                handles.append(attn_mod.register_forward_hook(self._save_attn_output))

            if not handles:
                raise RuntimeError(
                    f"model_type={mtype!r} is in mapping but no submodules at path "
                    f"{attn_attr!r} were found."
                )

        else:
            # generic fallback
            for module in self.model.modules():
                cname = module.__class__.__name__.lower()
                if "attention" in cname or isinstance(module, torch.nn.MultiheadAttention):
                    handles.append(module.register_forward_hook(self._save_attn_output))

            if not handles:
                raise RuntimeError(
                    f"modle_type={mtype!r} not in mapping and generic scan found no attention layers."
                )

        self._hook_handles = handles