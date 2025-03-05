import lightning.pytorch as pl
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    """Initializes a DataLoader object for a dataset with chunking support."""

    def __init__(
        self,
        name: str = "/grand/SuperBERT/pettyjohnjn/cache/datasets/chunked_pile",
        split: str = "train",
        batch_size: int = 4,
        num_workers: int = 16,
        pin_memory: bool = True,
        chunk_size: int = 128,  # added chunk size parameter
        chunk: bool = False,
    ) -> None:
        super().__init__()
        self.name = name
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.chunk_size = chunk_size
        self.chunk = chunk

    def chunk_examples(self, examples):
        # Ensure the correct field name based on your dataset
        chunks = []
        for sentence in examples["text"]:  # Assume "text" is the correct column
            # Chunk the sentence based on the chunk_size
            chunks += [sentence[i:i+self.chunk_size] for i in range(0, len(sentence), self.chunk_size)]
        return {"text": chunks}  # Returning the chunks correctly

    def setup(self, stage) -> None:
        """Initializes a huggingface dataset and applies chunking."""
        # Load the dataset
        #self.data = load_dataset(self.name, split=self.split)
        self.data = load_from_disk(self.name)

        # Randomly sample xx% of the data before processing
        sample_size = max(1, int(len(self.data) * 0.1)) # 
        self.data = self.data.shuffle(seed=42).select(range(sample_size))

        # Apply the chunking function using `map`
        if self.chunk:
            self.data = self.data.map(
                self.chunk_examples,
                batched=True,
                remove_columns=self.data.column_names,
                num_proc=32
                )

    def train_dataloader(self) -> DataLoader:
        """Creates an instance of `DataLoader` with the processed (chunked) data."""
        print(f"Batch size: {self.batch_size}")
        return DataLoader(
            self.data,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )