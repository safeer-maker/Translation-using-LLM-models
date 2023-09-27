from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

checkpoint = "marian-finetuned-kde4-en-to-fr-4-epochs-10000-samples"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# Translate English to French
sample_text = "I am learning Natural Language Processing with Hugging Face, Lets Tanslate into french"

input_ids = tokenizer(sample_text, return_tensors="pt")

fr_token = model.generate(**input_ids).squeeze()

french = tokenizer.decode(fr_token, skip_special_tokens=True)

print (french)
