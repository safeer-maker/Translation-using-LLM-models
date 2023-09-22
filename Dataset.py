from datasets import load_dataset
from transformers import AutoTokenizer


def load_dataset_from_hf(dataset_name, lang1 = 'en', lang2 = 'fr' ):
    dataset = load_dataset(dataset_name, lang1 = lang1, lang2= lang2)
    dataset = dataset.train_test_split(test_size=0.1)
    dataset['validation'] = dataset.pop('test')
    return dataset


df = load_dataset_from_hf('kde4', lang1 = 'en', lang2 = 'fr' )
print (df)