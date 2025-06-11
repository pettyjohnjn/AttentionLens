from __future__ import annotations

import torch
import torch.nn.functional as F
import transformers
import lightning.pytorch as pl
from torch import nn
from typing import Optional
from itertools import chain

import math

from attention_lens.lens import Lens
from attention_lens.model.get_model import get_model

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
        model_name: str,
        lens_cls: type[Lens] | str,
        lr: float = 1e-4,
        r: int = 8,
        train_attn: bool = True,
        use_attention_lens: bool = True,
        use_mlp_lens: bool = False,
        use_residual_lens: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.lr = lr
        self.r = r
        self.train_attn = train_attn
        self.lens_cls = lens_cls
        self.use_attention_lens = use_attention_lens
        self.use_mlp_lens = use_mlp_lens
        self.use_residual_lens = use_residual_lens

        # Caches for different components
        self.attn_cache: list[torch.Tensor] = []
        self.mlp_cache: list[torch.Tensor] = []
        self.residual_cache: list[torch.Tensor] = []

        self.model: Optional[transformers.PreTrainedModel] = None
        self.tokenizer: Optional[transformers.PreTrainedTokenizer] = None

        self.attn_lens: Optional[Lens] = None
        self.mlp_lens: Optional[Lens] = None
        self.residual_lens: Optional[Lens] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" and self.train_attn:
            # Load model & tokenizer onto this rank's device
            self.model, self.tokenizer = get_model(
                model_name=self.model_name,
                device=self.device,
            )
            base_model = getattr(self.model, 'module', self.model)

            # Clone + transpose unembed once per GPU
            W = base_model.lm_head.weight.t().detach().clone().to(self.device)
            b0 = base_model.lm_head.bias
            b = b0.clone().to(self.device) if b0 is not None else torch.zeros(
                base_model.config.vocab_size, device=self.device
            )
            # Register shared buffers per rank
            self.register_buffer("unembed", W)
            self.register_buffer("bias", b)

            # Resolve lens class
            if isinstance(self.lens_cls, str):
                self.lens_cls = Lens.get_lens(self.lens_cls)
            elif not issubclass(self.lens_cls, Lens):
                raise ValueError(
                    "`lens_cls` must be a subclass of Lens or its string name"
                )

            # Instantiate lenses selectively, reusing same unembed/bias
            if self.use_attention_lens:
                self.attn_lens = self.lens_cls(
                    unembed=self.unembed,
                    bias=self.bias,
                    n_layers=base_model.config.n_layer,
                    d_model=base_model.config.hidden_size,
                    d_vocab=base_model.config.vocab_size,
                    r=self.r,
                )
            if self.use_mlp_lens:
                self.mlp_lens = self.lens_cls(
                    unembed=self.unembed,
                    bias=self.bias,
                    n_layers=base_model.config.n_layer,
                    d_model=base_model.config.hidden_size,
                    d_vocab=base_model.config.vocab_size,
                    r=self.r,
                )
            if self.use_residual_lens:
                self.residual_lens = self.lens_cls(
                    unembed=self.unembed,
                    bias=self.bias,
                    n_layers=base_model.config.n_layer,
                    d_model=base_model.config.hidden_size,
                    d_vocab=base_model.config.vocab_size,
                    r=self.r,
                )

            # Register hooks selectively
            if self.use_attention_lens:
                self._register_attention_hooks(base_model)
            if self.use_mlp_lens:
                self._register_mlp_hooks(base_model)
            if self.use_residual_lens:
                self._register_residual_hooks(base_model)

    def _make_attention_hook(self, layer_id: int):
        def _hook(module, inputs, output):
            attn_out = output[0] if isinstance(output, tuple) else output
            self.attn_cache.append(attn_out.detach())
        return _hook

    def _make_mlp_hook(self, layer_id: int):
        def _hook(module, inputs, output):
            mlp_out = output[0] if isinstance(output, tuple) else output
            self.mlp_cache.append(mlp_out.detach())
        return _hook

    def _make_residual_hook(self, layer_id: int):
        def _hook(module, inputs, output):
            res_out = output[0] if isinstance(output, tuple) else output
            self.residual_cache.append(res_out.detach())
        return _hook

    def _register_attention_hooks(self, base_model) -> None:
        if hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'h'):
            layers = base_model.transformer.h
            for idx, block in enumerate(layers):
                block.attn.register_forward_hook(self._make_attention_hook(idx))
        elif hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
            layers = base_model.model.layers
            for idx, block in enumerate(layers):
                block.self_attn.register_forward_hook(self._make_attention_hook(idx))
        else:
            raise ValueError("Unsupported architecture for attention hooks")

    def _register_mlp_hooks(self, base_model) -> None:
        if hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'h'):
            layers = base_model.transformer.h
            for idx, block in enumerate(layers):
                block.mlp.register_forward_hook(self._make_mlp_hook(idx))
        elif hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
            layers = base_model.model.layers
            for idx, block in enumerate(layers):
                block.mlp.register_forward_hook(self._make_mlp_hook(idx))
        else:
            raise ValueError("Unsupported architecture for MLP hooks")

    def _register_residual_hooks(self, base_model) -> None:
        if hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'h'):
            layers = base_model.transformer.h
            for idx, block in enumerate(layers):
                block.register_forward_hook(self._make_residual_hook(idx))
        elif hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
            layers = base_model.model.layers
            for idx, block in enumerate(layers):
                block.register_forward_hook(self._make_residual_hook(idx))
        else:
            raise ValueError("Unsupported architecture for residual hooks")
        
    def kl_loss(self, logits: torch.Tensor, lens_logits: torch.Tensor) -> torch.Tensor:
        kldiv = nn.KLDivLoss(reduction="batchmean", log_target=True)
        k_logits = F.log_softmax(logits[:, -1, :], dim=-1)
        k_lens = F.log_softmax(lens_logits[:, -1, :], dim=-1)
        return kldiv(k_lens, k_logits)
    
    # def kl_loss(self, logits: torch.Tensor, lens_logits: torch.Tensor) -> torch.Tensor:
    #     """
    #     logits:       (B, T, V)
    #     lens_logits:  (B, T, L, V)
    #     returns:      scalar KL averaged over layers and batch
    #     """
    #     B, T, V = logits.shape
    #     _, _, L, _ = lens_logits.shape
    #     kldiv = nn.KLDivLoss(reduction="batchmean", log_target=True)

    #     # 1) log-probs for the “true” model at last position
    #     p = F.log_softmax(logits[:, -1, :], dim=-1)         # (B, V)

    #     # 2) log-probs for each layer’s lens at last position
    #     q = F.log_softmax(lens_logits[:, -1, :, :], dim=-1) # (B, L, V)

    #     # 3) expand true log-probs to match shape
    #     p_exp = p.unsqueeze(1).expand(-1, L, -1)            # (B, L, V)

    #     # 4) flatten the layer‐batch dims so we get one big batch of size B*L
    #     p_flat = p_exp.reshape(-1, V)                       # (B*L, V)
    #     q_flat = q.reshape(-1, V)                           # (B*L, V)

    #     # 5) compute one KL over that big batch
    #     return kldiv(q_flat, p_flat)

    def forward(self, cache: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Use specific lens objects: attn_lens, mlp_lens, or residual_lens")

    def training_step(self, train_batch: dict, batch_idx: int) -> torch.Tensor:
        # Clear caches
        if self.use_attention_lens:
            self.attn_cache.clear()
        if self.use_mlp_lens:
            self.mlp_cache.clear()
        if self.use_residual_lens:
            self.residual_cache.clear()

        # Tokenize & run base model
        inputs = self.tokenizer(
            train_batch["text"], truncation=True, padding=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits

        losses = []
        # Attention lens
        if self.use_attention_lens:
            assert len(self.attn_cache) == self.attn_lens.n_layers, \
                f"Expected {self.attn_lens.n_layers} attn caches, got {len(self.attn_cache)}"
            attn_tensor = torch.stack(self.attn_cache, dim=0).permute(1, 2, 0, 3)
            attn_logits = self.attn_lens(attn_tensor)
            loss_attn = self.kl_loss(logits, attn_logits)
            self.log("loss_attn", loss_attn, prog_bar=True)
            losses.append(loss_attn)

        # MLP lens
        if self.use_mlp_lens:
            assert len(self.mlp_cache) == self.mlp_lens.n_layers, \
                f"Expected {self.mlp_lens.n_layers} mlp caches, got {len(self.mlp_cache)}"
            mlp_tensor = torch.stack(self.mlp_cache, dim=0).permute(1, 2, 0, 3)
            mlp_logits = self.mlp_lens(mlp_tensor)
            loss_mlp = self.kl_loss(logits, mlp_logits)
            self.log("loss_mlp", loss_mlp, prog_bar=True)
            losses.append(loss_mlp)

        # Residual lens
        if self.use_residual_lens:
            assert len(self.residual_cache) == self.residual_lens.n_layers, \
                f"Expected {self.residual_lens.n_layers} residual caches, got {len(self.residual_cache)}"
            res_tensor = torch.stack(self.residual_cache, dim=0).permute(1, 2, 0, 3)
            res_logits = self.residual_lens(res_tensor)
            loss_res = self.kl_loss(logits, res_logits)
            self.log("loss_res", loss_res, prog_bar=True)
            losses.append(loss_res)

        save_memory_usage()
        # Combine and log
        total_loss = torch.stack(losses).mean()
        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # Collect parameters from active lenses
        lens_modules = []
        if self.use_attention_lens:
            lens_modules.append(self.attn_lens)
        if self.use_mlp_lens:
            lens_modules.append(self.mlp_lens)
        if self.use_residual_lens:
            lens_modules.append(self.residual_lens)
        params = chain(*(lens.parameters() for lens in lens_modules if lens is not None))
        return torch.optim.Adam(params, lr=self.lr)
