import lightning.pytorch as pl

from lightning import Callback
from typing import Optional, Union


from attention_lens.data.get_data_pl import DataModule
from attention_lens.train.config import TrainConfig
from attention_lens.train.lightning_lens import LightningLens

from lightning.pytorch.strategies import FSDPStrategy, DeepSpeedStrategy

def train_lens(
    lens: LightningLens,
    data_module: DataModule,
    config: TrainConfig,
    callbacks: Optional[Union[list[Callback], Callback]] = None,
):
    """
    Trains the given lens model using the provided data module and training configuration.

    Args:
        lens (LightningLens): The LightningLens model to be trained.
        data_module (DataModule): The DataModule providing the training and validation data.
        config (TrainConfig): The configuration settings for training.
        callbacks (Optional[Union[list[Callback], Callback]]): Optional list of callbacks or
        a single callback to be used during training.

    Notes:
        - The training precision is set to mixed precision (16-mixed) if config.mix_precision is True,
        otherwise 32-bit precision is used.
        - The training uses a distributed data parallel strategy with unused parameter detection
        enabled. (Necessary for GPU training, incompatible with CPU training.)
        - If no specific checkpoint to reload from is specified, the function searches for the most recent
        checkpoint in the checkpoint directory.

    Returns:
        trainer fits the lens according to data_module.

    Examples:
        >>> lens = LightningLens(config.model_name, "lensa", config.layer_number, config.lr)
        >>> data = DataModule()
        >>> train_lens(lens, data, config, callbacks=[checkpoint_callback, early_stop_callback])
    """

    # accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    
    #   The training precision is set to mixed preciision (16-mixed) if config.mix_precision is True,
    # otherwise 32-bit precision is used.
    training_precision = "16-mixed" if config.mixed_precision else 32

    if config.strategy == "deepspeed_stage_2":
        deepspeed_stage_2_config = {
            "zero_optimization": {
                "stage": 2,  # Using ZeRO Stage 2
                "model_persistence_threshold": 0,  # Custom threshold
            },
        }
        strategy = DeepSpeedStrategy(config=deepspeed_stage_2_config)
    elif config.strategy == "deepspeed_stage_3":
        deepspeed_stage_3_config = {
            "zero_optimization": {
                "stage": 3,
                "model_persistence_threshold": 0,
            }
        }
        strategy = DeepSpeedStrategy(config=deepspeed_stage_3_config)
    elif config.strategy == "fsdp":
        strategy = FSDPStrategy(use_orig_params=True)

    trainer = pl.Trainer(
        #   The training uses a distributed data parallel strategy with unused parameter detection
        #enabled. (Necessary for GPU training, incompatible for CPU training.)
        strategy=strategy,
        precision=training_precision,
        accelerator="auto",
        max_epochs=config.max_epochs,
        num_nodes=config.num_nodes,
        default_root_dir=config.checkpoint_dir,
        accumulate_grad_batches=config.accumulate_grad_batches,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        # callbacks=[early_stop_callback, logging_checkpoint, latest_checkpoint],
        # flush_logs_every_n_steps=100,
        #log_every_n_steps=50,
        # logger=csv_logger)
        # logger=wandb_logger)
        # TODO(MS): eventually use the profile to find bottlenecks: profiler='simple')
    )

    #   If no specific checkpoint to reload from is specified, the function searches for the most
    #recent checkpoint in the checkpoint directory.
    if config.reload_checkpoint is None:
        print(
            "Checkpoint directory to reload from is not specified, searching for existing checkpoint directory."
        )

        if config.checkpoint_dir.exists():
            files = list(config.checkpoint_dir.glob("*.ckpt"))
            if files:
                most_recent_file = max(files, key=lambda p: p.stat().st_ctime)
                config.reload_checkpoint = most_recent_file

    print("latest checkpoint file: ", config.reload_checkpoint)
    trainer.fit(lens, data_module, ckpt_path=config.reload_checkpoint)
