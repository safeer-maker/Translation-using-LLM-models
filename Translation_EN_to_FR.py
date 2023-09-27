import Model_Training
import Dataset
import os

# loading the token
Hugging_face_token = os.getenv("HUGGING_FACE_TOKEN")
# print (Hugging_face_token)

# defining the model checkpoint
CHECKPOINT = "Helsinki-NLP/opus-mt-en-fr"
os.environ ['CHECKPOINT'] = CHECKPOINT

# loading the dataset
ds = Dataset.load_dataset_from_hf('kde4', lang1 = 'en', lang2 = 'fr' )

# Lets Reduce the train Dataset side to 10000
sub_ds = ds
sub_ds['train'] = ds['train'].shuffle(seed=11).select(range(10000))
sub_ds['validation'] = ds['validation'].shuffle(seed=11).select(range(1000))

print (sub_ds)
# Lets Tokenize the complete dataset
ds_tokenized = sub_ds.map ( Dataset.tokenized_ds, batched=True, remove_columns=ds['train'].column_names)

print (ds_tokenized)

# Lets Train the model
Trained_model_name = "marian-finetuned-kde4-en-to-fr-4-epochs-10000-samples"

fine_tune_model = Model_Training.fine_tune_model (CHECKPOINT, ds_tokenized, 4, Trained_model_name)

