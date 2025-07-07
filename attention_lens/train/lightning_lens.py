from __future__ import annotations

import torch
import torch.nn.functional as F
import transformers
import lightning.pytorch as pl
from torch import nn
from typing import Optional
from itertools import chain
from transformers import get_linear_schedule_with_warmup

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
        lens_cls: type[Lens] | str,
        lr: float = 1e-4,
        r: int = 8,
        train_attn: bool = True,
        use_attention_lens: bool = False,
        use_mlp_lens: bool = False,
        use_residual_lens: bool = True,
        sum_logits: bool = False,
        model_name: Optional[str] = None,
        model: Optional[transformers.PreTrainedModel] = None,
        tokenizer: Optional[transformers.PreTrainedTokenizer] = None,

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
        self.sum_logits = sum_logits

        # Caches for different components
        self.input_cache: Optional[torch.tensor] = None
        self.attn_cache: list[torch.Tensor] = []
        self.mlp_cache: list[torch.Tensor] = []
        self.residual_cache: list[torch.Tensor] = []

        self.model: Optional[transformers.PreTrainedModel] = model
        self.tokenizer: Optional[transformers.PreTrainedTokenizer] = tokenizer

        self.attn_lens: Optional[Lens] = None
        self.mlp_lens: Optional[Lens] = None
        self.residual_lens: Optional[Lens] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" and self.train_attn:
            # Load model & tokenizer onto this rank's device
            if self.model is None or self.tokenizer is None:
                if self.model_name is None:
                    raise ValueError("Either pass `model_name` or a `(model, tokenizer)` pair into LightningLens")
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

            self._register_input_hook(base_model)
            # Register hooks selectively
            if self.use_attention_lens:
                self._register_attention_hooks(base_model)
            if self.use_mlp_lens:
                self._register_mlp_hooks(base_model)
            if self.use_residual_lens:
                self._register_residual_hooks(base_model)
    
    def _make_input_hook(self):
        def _hook(module, inputs, output):
            self.input_cache = output.detach()
        return _hook

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
    
    def _register_input_hook(self, base_model):
        base_model.transformer.drop.register_forward_hook(self._make_input_hook())

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
        
    # def kl_loss(self, logits: torch.Tensor, lens_logits: torch.Tensor, mask=None) -> torch.Tensor:
    #     kldiv = nn.KLDivLoss(reduction="batchmean", log_target=True)
    #     k_logits = F.log_softmax(logits[:, -1, :], dim=-1)
    #     k_lens = F.log_softmax(lens_logits[:, -1, :], dim=-1)
    #     return kldiv(k_lens, k_logits)

    def kl_loss(self,
                logits:      torch.Tensor,  # [bsz, seq_len, vocab]
                lens_logits: torch.Tensor,  # [bsz, seq_len, vocab]
                mask:        torch.Tensor,  # [bsz, seq_len]
            ) -> torch.Tensor:
        # use sum-then-mean so we can mask explicitly
        kldiv = nn.KLDivLoss(reduction="sum", log_target=True)

        # log-probs
        log_p = F.log_softmax(logits,     dim=-1)  # [b,s,v]
        log_q = F.log_softmax(lens_logits, dim=-1)  # [b,s,v]

        b, s, v = log_p.shape
        # flatten to [b*s, v]
        log_p = log_p.view(-1, v)
        log_q = log_q.view(-1, v)
        mask  = mask.view(-1).bool()               # [b*s]

        # only keep real tokens
        log_p = log_p[mask]
        log_q = log_q[mask]

        # sum KL over all token positions, then average
        total_kl = kldiv(log_p, log_q)             # scalar
        return total_kl / mask.sum()

    def forward(self, cache: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Use specific lens objects: attn_lens, mlp_lens, or residual_lens")

    def training_step(self, train_batch: dict, batch_idx: int) -> torch.Tensor:
        # Clear caches
        self.input_cache = None
        if self.use_attention_lens:
            self.attn_cache.clear()
        if self.use_mlp_lens:
            self.mlp_cache.clear()
        if self.use_residual_lens:
            self.residual_cache.clear()

        # Tokenize & run base model
        # inputs = self.tokenizer(
        #     train_batch["text"], 
        #     truncation=True, 
        #     padding=True, 
        #     return_tensors="pt",
        #     max_length=1024,
        # ).to(self.device)

        # mask = inputs["attention_mask"]

        input_ids = train_batch["input_ids"].to(self.device)
        attention_mask = train_batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
        
        assert self.input_cache is not None, "input_cache was not set by the hook!"

        r0_logits = torch.einsum("bsd,dk->bsk", self.input_cache, self.unembed) + self.bias

        # losses = []

        # collect each len's predicted logits
        lens_logits = [r0_logits]
        # Attention lens
        if self.use_attention_lens:
            assert len(self.attn_cache) == self.attn_lens.n_layers, \
                f"Expected {self.attn_lens.n_layers} attn caches, got {len(self.attn_cache)}"
            attn_cache = torch.stack(self.attn_cache, dim=0).permute(1, 2, 0, 3)
            attn_logits = self.attn_lens(attn_cache, mask, sum_logits = self.sum_logits)
            # loss_attn = self.kl_loss(logits, attn_logits)
            # self.log("loss_attn", loss_attn, prog_bar=True)
            # losses.append(loss_attn)
            lens_logits.append(attn_logits)

        # MLP lens
        if self.use_mlp_lens:
            assert len(self.mlp_cache) == self.mlp_lens.n_layers, \
                f"Expected {self.mlp_lens.n_layers} mlp caches, got {len(self.mlp_cache)}"
            mlp_cache = torch.stack(self.mlp_cache, dim=0).permute(1, 2, 0, 3)
            mlp_logits = self.mlp_lens(mlp_cache, mask, sum_logits = self.sum_logits)
            # loss_mlp = self.kl_loss(logits, mlp_logits)
            # self.log("loss_mlp", loss_mlp, prog_bar=True)
            # losses.append(loss_mlp)
            lens_logits.append(mlp_logits)

        # Residual lens (Tuned Lens)
        if self.use_residual_lens:
            assert len(self.residual_cache) == self.residual_lens.n_layers, \
                f"Expected {self.residual_lens.n_layers} residual caches, got {len(self.residual_cache)}"
            res_cache = torch.stack(self.residual_cache, dim=0).permute(1, 2, 0, 3)
            res_logits = self.residual_lens(res_cache, mask, sum_logits = self.sum_logits)
            # loss_res = self.kl_loss(logits, res_logits)
            # self.log("loss_res", loss_res, prog_bar=True)
            # losses.append(loss_res)
            lens_logits.append(res_logits)

        if self.sum_logits:
            combined = torch.stack(lens_logits, dim=0).sum(dim=0)
            loss = self.kl_loss(combined, logits, mask) / math.log(2)
        else:
            losses: list[torch.Tensor] = []
            for ll in lens_logits:
                if ll.ndim == 4: # [B, S, n_layers, V]
                    for i in range(ll.size(2)):
                        losses.append(self.kl_loss(ll[:, :, i, :], logits, mask))
                else:
                    losses.append(self.kl_loss(ll, logits, mask))
            loss = torch.stack(losses).mean() / math.log(2)

        # save_memory_usage()
        # Combine and log
        # Sum all lens logits, compute one KL against the model's logits
        # lens_logits = torch.stack(lens_logits, dim=0).sum(dim=0)
        # total_loss = torch.stack(losses).mean()
        # loss = self.kl_loss(lens_logits, logits, mask) / math.log(2) # Convert nats to bits
        self.log("train_loss", 
                 loss, 
                 prog_bar=True,
                 on_step=True,
                 on_epoch=False,
                 )
        return loss

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
        # return torch.optim.AdamW(params, 
        #                          lr=self.lr,
        #                          betas=(0.9, 0.98),
        #                          eps=1e-6,
        #                          weight_decay=1e-2)

    # def configure_optimizers(self):

    #     # Collect parameters
    #     lens_modules = []
    #     if self.use_attention_lens:
    #         lens_modules.append(self.attn_lens)
    #     if self.use_mlp_lens:
    #         lens_modules.append(self.mlp_lens)
    #     if self.use_residual_lens:
    #         lens_modules.append(self.residual_lens)

    #     params = chain(*(lens.parameters() for lens in lens_modules if lens is not None))

    #     # Initialize AdamW with weight decay
    #     optimizer = torch.optim.AdamW(
    #         params,
    #         lr=self.lr,
    #         betas=(0.9, 0.98),
    #         eps=1e-6,
    #         weight_decay=1e-2,
    #     )

    #     # Linear warmup + decay over entire training run
    #     num_steps = self.trainer.estimated_stepping_batches
    #     num_warmup = max(1, num_steps // 20)
    #     print(f"{num_steps} estimated steps with {num_warmup} warmup")
    #     scheduler = get_linear_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=num_warmup,
    #         num_training_steps=num_steps,
    #     )

    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "interval": "step",
    #             "frequency": 1,
    #         },
    #     }

    # def configure_optimizers(self):
    #     lens_modules = []
    #     if self.use_attention_lens:
    #         lens_modules.append(self.attn_lens)
    #     if self.use_mlp_lens:
    #         lens_modules.append(self.mlp_lens)
    #     if self.use_residual_lens:
    #         lens_modules.append(self.residual_lens)

    #     params = chain(*(m.parameters() for m in lens_modules if m is not None))

    #     # Initialize momentum SGD + Nesterov
    #     optimizer = torch.optim.SGD(
    #         params,
    #         lr=self.lr,                # 1.0 or 0.25
    #         momentum=0.9,
    #         nesterov=True,
    #         weight_decay=1e-3,         # 1 × 10⁻³
    #     )

    #     # Linear decay to 0 over 250 steps 
    #     scheduler = torch.optim.lr_scheduler.LinearLR(
    #         optimizer,
    #         start_factor=0.25,
    #         end_factor=0.0,
    #         total_iters=500,
    #     )

    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "interval": "step",
    #             "frequency": 1,
    #         },
    #     }

    # def configure_optimizers(self):
        # # 1) Collect all lens parameters
        # lens_modules = []
        # if self.use_attention_lens:
        #     lens_modules.append(self.attn_lens)
        # if self.use_mlp_lens:
        #     lens_modules.append(self.mlp_lens)
        # if self.use_residual_lens:
        #     lens_modules.append(self.residual_lens)
        # params = chain(*(m.parameters() for m in lens_modules if m is not None))

        # # 2) Optimizer
        # optimizer = torch.optim.SGD(
        #     params,
        #     lr=self.lr,           # peak LR, e.g. 5e-3
        #     momentum=0.9,
        #     nesterov=True,
        #     weight_decay=1e-3,
        # )

        # # 3) Warmup via LambdaLR: factor = step/warmup_steps (clamped ≤1)
        # warmup_steps = getattr(self, 'warmup_steps', 100)
        # def warmup_fn(step):
        #     return min((step + 1) / warmup_steps, 1.0)
        # warmup_sched = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer,
        #     lr_lambda=warmup_fn
        # )

        # # 4) Indefinite damped cyclic schedule
        # cycle_up   = getattr(self, 'cycle_up_steps',   1000)
        # cycle_down = getattr(self, 'cycle_down_steps', 1000)
        # cyclic_sched = torch.optim.lr_scheduler.CyclicLR(
        #     optimizer,
        #     base_lr=self.lr * 0.1,  # floor at 10%
        #     max_lr=self.lr,         # peak
        #     step_size_up=cycle_up,
        #     step_size_down=cycle_down,
        #     mode='triangular2',     # halves amplitude each cycle
        #     cycle_momentum=False,
        # )

        # # 5) Chain: warmup first, then cyclic forever
        # scheduler = torch.optim.lr_scheduler.SequentialLR(
        #     optimizer,
        #     schedulers=[warmup_sched, cyclic_sched],
        #     milestones=[warmup_steps],
        # )

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "step",
        #         "frequency": 1,
        #     },
        # }