import argparse


def get_args() -> argparse.Namespace:
    # TODO (MS): clean up args and keep only relavent ones
    # NOTE(MS): I copied these args from a different project so they might not be relavent anymore
    #### SET UP USER ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--max_epochs", default=20, type=int)
    parser.add_argument(
        "--max_checkpoint_num",
        default=1,
        type=int,
        help="maximum number of ckpts to save",
    )
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument(
        "--model_name",
        default="gpt2",
        type=str,
        choices=["gpt2", "llama3_8b"],
    )
    parser.add_argument(
        "--mixed_precision",
        default=True,
        type=bool,
        help="whether to use mixed precision for training",
    )
    parser.add_argument(
        "--checkpoint_mode",
        default="step",
        type=str,
        choices=["step", "loss"],
        help="whether to checkpoint on train loss decrease or training step number",
    )
    parser.add_argument(
        "--num_steps_per_checkpoint",
        default=5,
        type=int,
        help="number of steps after which to checkpoint (only valid for checkpoint_mode='step')",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoint",
        type=str,
        help="directory to store checkpoint files in; if dir already has ckpt files and user didn't specity a differnt ckpt dir to reload from (via relaod_checkpoint), we will reload latest ckptfile in this dir",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        default=16,
        type=int,
        help="controls how many steps to accumulate gradients over",
    )
    parser.add_argument(
        "--reload_checkpoint",
        default=None,
        type=str,
        help="path to checkpoint file, if set training resumes using this checkpoint",
    )
    parser.add_argument(
        "--stopping_delta",
        default=1e-7,
        type=float,
        help="early stopping delta, if train loss decreases by <= delta we stop training",
    )
    parser.add_argument(
        "--stopping_patience",
        default=2,
        type=int,
        help="number of checks with no improvement after which to stop training",
    )
    parser.add_argument(
        "--lora_rank",
        default=8,
        type=int,
        help="rank of the LoRA approximation."
    )
    parser.add_argument(
        "--strategy",
        default="deepspeed_stage_2",
        type=str,
        help="training strategy to use (deepspeed_stage_2, deepspeed_stage_3, fdsp)"
    )
    return parser.parse_args()



    # Mapping aliases to full model names
    model_aliases = {
        "gpt2": "gpt2",
        "llama3_8b": "/grand/SuperBERT/aswathy/models/models--meta-llama--Meta-Llama-3-8B-Instruct",
    }
    args.model_name = model_aliases[args.model_name]

    return args