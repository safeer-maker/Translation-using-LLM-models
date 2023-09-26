from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np
import evaluate


################### Global variables ###################
model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

metric = evaluate.load ("sacrebleu")


def load_dataset_from_hf(dataset_name, lang1 = 'en', lang2 = 'fr' ):
    dataset = load_dataset(dataset_name, lang1 = lang1, lang2= lang2)
    dataset = dataset['train'].train_test_split(test_size=0.1, seed=11)
    dataset['validation'] = dataset.pop('test')
    return dataset


def tokenized_ds(dataset):
    
    eng = [en['en'] for en in dataset['translation']]
    fr  = [fr['fr'] for fr in dataset['translation']]
    token = tokenizer (eng, text_target=fr,max_length=128, truncation=True)
    return token


ds = load_dataset_from_hf('kde4', lang1 = 'en', lang2 = 'fr' )

# Lets Reduce the Dataset side to 10000
ds['train'] = ds['train'].shuffle(seed=11).select(range(10000))
ds['validation'] = ds['validation'].shuffle(seed=11).select(range(1000))

tok_ds = ds.map ( tokenized_ds, batched=True, remove_columns=ds['train'].column_names )

print (tok_ds['train'][:3])


