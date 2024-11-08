import torch.nn as nn


class Lens(nn.Module):
    """
    Base class for attention lenses with optional LoRA integration.
    Subclasses should implement specific lens behaviors.
    """

    registry = {}

    def __init__(
        self,
        unembed: nn.Parameter,
        bias: nn.Parameter,
        n_head: int,
        d_model: int,
        d_vocab: int,
        r: int = 0,               # LoRA rank
        lora_alpha: int = 1,      # LoRA scaling factor
        lora_dropout: float = 0.0, # LoRA dropout rate
        merge_weights: bool = True, # Whether to merge weights during inference
    ) -> None:
        """
        Args:
            unembed (nn.Parameter): Unembedding matrix \( W_U \).
            bias (nn.Parameter): Bias vector.
            n_head (int): Number of attention heads.
            d_model (int): Dimension of the model.
            d_vocab (int): Vocabulary size.
            r (int, optional): LoRA rank. Defaults to 0 (no LoRA).
            lora_alpha (int, optional): LoRA scaling factor. Defaults to 1.
            lora_dropout (float, optional): LoRA dropout rate. Defaults to 0.0.
            merge_weights (bool, optional): Whether to merge LoRA weights during inference. Defaults to True.
        """
        super().__init__()
        self.unembed = unembed
        self.bias = bias
        self.n_head = n_head
        self.d_model = d_model
        self.d_vocab = d_vocab
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.merge_weights = merge_weights

    @classmethod
    def get_lens(cls, name: str) -> type["Lens"]:
        """
        Retrieves a Lens subclass from the registry.

        Args:
            name (str): Name of the Lens subclass.

        Returns:
            type[Lens]: The corresponding Lens subclass.
        """
        name = name.lower()
        if name in cls.registry:
            return cls.registry[name]
        else:
            raise KeyError(
                f"Strategy name ({name=}) is not in the Strategy registry. Available ``Lens`` objects are: "
                f"{list(cls.registry.keys())}"
            )

    def __init_subclass__(cls, **kwargs) -> None:
        """
        Automatically registers Lens subclasses.
        """
        super().__init_subclass__(**kwargs)
        cls.registry[cls.__name__.lower()] = cls