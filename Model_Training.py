import evaluate
import numpy as np
from transformers import AutoTokenizer,DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import os

################### Functions ###################

# Function to compute the metrics
def compute_metrics(eval_preds):
    """
    compute_metrics: Function to compute the metrics

    param eval_preds: The predictions from the model
    """
    preds, labels = eval_preds
    model_checkpoint = os.getenv("CHECKPOINT")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    metric = evaluate.load ("sacrebleu")
    
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

# Function to get the training arguments
def training_args(model_checkpoint, batch_size):
    """
    training_args: Function to get the training arguments

    param model_checkpoint: The model checkpoint to use for training
    param batch_size: The batch size to use for training
    """
    args = Seq2SeqTrainingArguments(
        model_checkpoint,
        evaluation_strategy="no",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=True,
    )

    return args

# Function to fine tune the model
def fine_tune_model (model_name, dataset_tokenized, batch_size, model_name_to_save):
    """
    fine_tune_model: Function to fine tune the model.

    param model_name: The model name to use for training
    param dataset_tokenized: The tokenized dataset to use for training
    param batch_size: The batch size to use for training
    param model_name_to_save: The model name to save the fine tuned model
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    args = training_args(model_name, batch_size)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorForSeq2Seq (tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=dataset_tokenized["train"],
        eval_dataset=dataset_tokenized["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Fine tune the model
    try:
        # If the training hits any error, the model is not saved
        trainer.train()
        trainer.save_model(model_name_to_save)
        print(f"Model saved successfully with name {model_name_to_save}")

        return model
    except:
        print("Error in training the model")
        return -1

    
