from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

checkpoint = "marian-finetuned-kde4-en-to-fr-4-epochs-10000-samples"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# Translate English to French
def Translate_EN_to_FR (text):
    """Translate English to French
    Args:
        text (str): English text to translate
    """
    input_ids = tokenizer(text, return_tensors="pt")
    fr_token = model.generate(**input_ids).squeeze()
    french = tokenizer.decode(fr_token, skip_special_tokens=True)
    return french

# Lets use Gradio to create a simple UI for our model
input_text = gr.Textbox(lines=5, label="Input Text to Translate")
output_text = gr.Textbox(label="Output Text Translated into French")

demo = gr.Interface(
    Translate_EN_to_FR, 
    input_text, 
    output_text, 
    title="English to French Translator", 
    allow_flagging="never" )

demo.launch()
