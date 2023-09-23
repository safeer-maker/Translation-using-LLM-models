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

args = Seq2SeqTrainingArguments(
    f"marian-finetuned-kde4-en-to-fr",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    #predict_with_generate=True,
    #fp16=True,
)


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

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

ds = load_dataset_from_hf('kde4', lang1 = 'en', lang2 = 'fr' )

# Lets Reduce the Dataset side to 10000
ds['train'] = ds['train'].shuffle(seed=11).select(range(10000))
ds['validation'] = ds['validation'].shuffle(seed=11).select(range(1000))

tok_ds = ds.map ( tokenized_ds, batched=True, remove_columns=ds['train'].column_names )

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tok_ds["train"],
    eval_dataset=tok_ds["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print (tok_ds['train'][:3])

trainer.train()

trainer.save_model("marian-finetuned-kde4-en-to-fr")
