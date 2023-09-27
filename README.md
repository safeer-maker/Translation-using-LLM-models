# Translation-using-LLM-models

Demo link: 
Fine tune model link: https://huggingface.co/Safeer143/marian-finetuned-kde4-en-to-fr-4-epochs-10000-samples


this repo consist of four files
Datasets
Model_Training
FineTuning_the_model
Translating_with_model_EN_to_FR
Translation

Dataset:
dataset is loading the data set from hugging face and doing tokenization on dataset.
Function:
def load_dataset_from_hf(dataset_name, lang1 = 'en', lang2 = 'fr' ):
    """
    load_dataset_from_hf: Function to load the dataset from Hugging Face
    param dataset_name: The name of the dataset to load
    param lang1: The language of the first column
    param lang2: The language of the second column
    """

def tokenized_ds(dataset):
    """
    tokenized_ds: Function to tokenize the dataset
    param dataset: The dataset to tokenize
    """

Model_Training:
in this module I am fine=tuning the LLM model and saving it to the hub

def fine_tune_model (model_name, dataset_tokenized, batch_size, model_name_to_save):
    """
    fine_tune_model: Function to fine tune the model.

    param model_name: The model name to use for training
    param dataset_tokenized: The tokenized dataset to use for training
    param batch_size: The batch size to use for training
    param model_name_to_save: The model name to save the fine tuned model
    """

def compute_metrics(eval_preds):
    """
    compute_metrics: Function to compute the metrics

    param eval_preds: The predictions from the model
    """

def training_args(model_checkpoint, batch_size):
    """
    training_args: Function to get the training arguments

    param model_checkpoint: The model checkpoint to use for training
    param batch_size: The batch size to use for training
    """


FineTuning_the_model
this module is integrated with above module to get dataset, reduced to a sutable number for training
then tokenized the dataset to fine tuned the model.
that fine tuned model is avaible on hugging face link
that same model is also used to for demo purposes.


Translating_with_model_EN_to_FR
Loaded the fine tuned model and used gradio for demo purposes

def Translate_EN_to_FR (text):
    """Translate English to French
    Args:
        text (str): English text to translate
    """

Translation
This is  Jupyter notebook I have used to perform working on various aspects of model,
dataset and all the random testing needs to be done is performed on this notebook
