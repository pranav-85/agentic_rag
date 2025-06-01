# summarizers.py

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from PIL import Image
import base64
import io

# === Load models ===
# Image captioning model (BLIP)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Table summarization model (FLAN-T5 base)
table_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
table_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# === IMAGE SUMMARIZATION ===
def summarize_image_from_data_uri(image_data_uri: str) -> str:
    try:
        if "," in image_data_uri:
            _, base64_str = image_data_uri.split(",", 1)
        else:
            base64_str = image_data_uri

        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        inputs = blip_processor(image, return_tensors="pt")
        out = blip_model.generate(**inputs, max_new_tokens=50)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)

        return caption
    except Exception as e:
        return f"[Image summarization failed: {str(e)}]"

# === TABLE SUMMARIZATION ===
def summarize_table(markdown_table: str) -> str:
    try:
        prompt = f"Summarize the following legal table:\n\n{markdown_table[:1500]}"
        inputs = table_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = table_model.generate(**inputs, max_new_tokens=64)
        summary = table_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        return f"[Table summarization failed: {str(e)}]"
