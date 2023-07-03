import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader


def get_data(streaming=True, batch_size=32, dataset_name="c4"):
    # will start loading the data when iterated over
    c4  = load_dataset(dataset_name, 'en', split="train", streaming=streaming)  
    dataloader = DataLoader(c4, batch_size=batch_size)
    '''
    print('loaded c4')
    for example in c4:
        print(example['text'])
        break
    '''
    return dataloader