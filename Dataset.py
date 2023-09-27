from datasets import load_dataset
from transformers import AutoTokenizer
import os

# Function to load the dataset from Hugging Face
def load_dataset_from_hf(dataset_name, lang1 = 'en', lang2 = 'fr' ):
    """
    load_dataset_from_hf: Function to load the dataset from Hugging Face
    param dataset_name: The name of the dataset to load
    param lang1: The language of the first column
    param lang2: The language of the second column
    """
    dataset = load_dataset(dataset_name, lang1 = lang1, lang2= lang2)
    dataset = dataset['train'].train_test_split(test_size=0.1, seed=11)
    dataset['validation'] = dataset.pop('test')
    return dataset

def tokenized_ds(dataset):
    """
    tokenized_ds: Function to tokenize the dataset
    param dataset: The dataset to tokenize
    """
    tokenizer_checkpoint = os.getenv("CHECKPOINT")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    eng = [en['en'] for en in dataset['translation']]
    fr  = [fr['fr'] for fr in dataset['translation']]
    token = tokenizer (eng, text_target=fr,max_length=128, truncation=True)
    return token

