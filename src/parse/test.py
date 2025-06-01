from PIL import Image
import io
import base64
from pathlib import Path
from document_parser import parse_pdf
from transformers import AutoTokenizer

# Config
pdf_path = "sample.pdf"
output_dir = Path("output_images")
output_dir.mkdir(exist_ok=True)
embedding_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Parse PDF
result = parse_pdf(pdf_path, embedding_tokenizer)
images = result["images"]

# Resize factor (2x = 200% enlargement)
UPSCALE_FACTOR = 2

for img_doc in images:
    metadata = img_doc.metadata
    data_uri = str(metadata.get("image_data_uri", ""))

    if not data_uri.startswith("data:image/png;base64,"):
        print(f"[!] Skipping image {metadata.get('doc_id')} - invalid base64")
        continue

    # Decode base64
    base64_data = data_uri.split(",", 1)[1]
    image_bytes = base64.b64decode(base64_data)

    # Open and resize image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    new_size = (int(image.width * UPSCALE_FACTOR), int(image.height * UPSCALE_FACTOR))
    highres_image = image.resize(new_size, Image.BICUBIC)

    # Save high-res image
    filename = f"{metadata['filename'].replace('.pdf', '')}_page{metadata.get('page_number')}_fig{metadata.get('figure_number')}.png"
    output_path = output_dir / filename
    highres_image.save(output_path, format="PNG")

    print(f"[✓] Saved upscaled image to {output_path}")

print(f"\nSaved {len(images)} high-resolution images to: {output_dir.resolve()}")

# Output directory for tables
tables_dir = Path("output_tables")
tables_dir.mkdir(exist_ok=True)

# Extract table documents
tables = result["tables"]

for table_doc in tables:
    metadata = table_doc.metadata
    table_text = table_doc.page_content.strip()
    if not table_text or len(table_text.splitlines()) < 2:
        print(f"[!] Skipping table {metadata.get('doc_id')} - seems empty or malformed")
        continue

    # Save as .md for easy viewing
    filename = f"{metadata['filename'].replace('.pdf', '')}_page{metadata.get('ref').split('/')[-1]}.md"
    output_path = tables_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(table_text)

    print(f"[✓] Saved table to {output_path}")
