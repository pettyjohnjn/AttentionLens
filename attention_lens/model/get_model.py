import torch.types

from typing import Union
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes as bnb


def get_model(
    model_name: str = "gpt2", device: Union[str, torch.types.Device] = "cuda"
) -> AutoModelForCausalLM:
    """Loads and returns a model and tokenizer from the modified Hugging Face Transformers library.

    Args:
        model_name (str): The name of the pre-trained model.
        device (Union[str, torch.types.Device]): The device to train on.

    Examples:
        >>> model, tokenizer = get_model("gpt2")

    Returns:
        The light-weight hooked model and tokenizer.
    """

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,        # Enable 4-bit quantization
        bnb_4bit_use_double_quant=True,  # Use double quantization (optional, can improve accuracy)
        bnb_4bit_quant_type='nf4',        # Quantization type: 'nf4' or 'fp4'
        bnb_4bit_compute_dtype=torch.float16  # Compute dtype during inference
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager") # Force manual implementation of attention
    # model.to(device)

    model.eval()
    model.requires_grad_(False)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token


    print("Language model initialized on device: ", device)
    return model, tokenizer
