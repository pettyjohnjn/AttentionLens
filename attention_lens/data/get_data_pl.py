import lightning.pytorch as pl
from datasets import load_from_disk, Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, PreTrainedTokenizer
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from .data import chunk_and_tokenize  # now importing from your local data.py


class DataModule(pl.LightningDataModule):
    def __init__(
        # dataset loading
        self,
        name: str = "/grand/SuperBERT/pettyjohnjn/cache/datasets/monology___pile-uncopyrighted",
        split: str = "train",
        sample_fraction: float = 0.1,
        seed: int = 42,
        load_from_cache_file: bool = True,

        # tokenization & chunking
        tokenizer: PreTrainedTokenizer | None = None,
        tokenizer_name_or_path: str = "gpt2",
        text_key: str = "text",
        max_seq_len: int = 2048,
        return_final_batch: bool = False,
        num_proc: int | None = None,

        # dataloader params
        batch_size: int = 16,
        num_workers: int = 16,
        pin_memory: bool = True,
        mlm: bool = False,  # for DataCollator
    ):
        super().__init__()
        # sampling
        self.name = name
        self.split = split
        self.sample_fraction = sample_fraction
        self.seed = seed
        self.load_from_cache_file = load_from_cache_file

        # tokenization / chunking
        self.tokenizer = tokenizer
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.text_key = text_key
        self.max_seq_len = max_seq_len
        self.return_final_batch = return_final_batch
        self.num_proc = num_proc or min(cpu_count() // 2, 8)

        # loader
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.mlm = mlm


    def setup(self, stage=None):
        # load & sample
        raw = load_dataset(self.name, split="train[:512000]")
        
        if isinstance(raw, DatasetDict):
            raw = raw[self.split]

        n = max(1, int(len(raw) * self.sample_fraction))
        raw = raw.shuffle(seed=self.seed).select(range(n))

        # prepare tokenizer
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # chunk & tokenize via your local data.py function
        tokenized, bpb = chunk_and_tokenize(
            raw,
            self.tokenizer,
            format="torch",
            num_proc=self.num_proc,
            text_key=self.text_key,
            max_seq_len=self.max_seq_len,
            return_final_batch=self.return_final_batch,
            load_from_cache_file=self.load_from_cache_file,
        )
        self.data = tokenized
        print(f"Dataset bits-per-byte ratio: {bpb:.3f}")


    def train_dataloader(self) -> DataLoader:
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=self.mlm,
        )
        return DataLoader(
            self.data,
            batch_size=self.batch_size,
            collate_fn=collator,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )