from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc.document import PictureItem
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc.document import TableItem, PictureItem
from docling_core.types.doc.labels import DocItemLabel
from langchain_core.documents import Document

import base64
import io
from PIL import Image, ImageOps
import os

# Configure docling converter
pdf_pipeline_options = PdfPipelineOptions(do_ocr=False, generate_picture_images=True, picture_dpi=300)
format_options = {InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)}
converter = DocumentConverter(format_options=format_options)

def parse_pdf(pdf_path: str, embeddings_tokenizer) -> dict:
    filename = os.path.basename(pdf_path)
    docling_document = converter.convert(source=pdf_path).document

    doc_id = 0
    text_chunks: list[Document] = []
    table_docs: list[Document] = []
    image_docs: list[Document] = []

    # --- Extract text chunks ---
    for chunk in HybridChunker(tokenizer=embeddings_tokenizer).chunk(docling_document):
        items = chunk.meta.doc_items
        if len(items) == 1 and isinstance(items[0], TableItem):
            continue  # skip tables here
        refs = " ".join([item.get_ref().cref for item in items])
        text = chunk.text
        doc_id += 1
        text_chunks.append(
            Document(
                page_content=text,
                metadata={
                    "doc_id": doc_id,
                    "type": "text",
                    "filename": filename,
                    "ref": refs
                }
            )
        )

    # --- Extract tables ---
    print(docling_document.tables[0])
    for table in docling_document.tables:
        if table.label == DocItemLabel.TABLE:
            ref = table.get_ref().cref
            text = table.export_to_markdown(doc=docling_document)
            doc_id += 1
            table_docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "doc_id": doc_id,
                        "type": "table",
                        "filename": filename,
                        "ref": ref
                    }
                )
            )

    # --- Extract images ---
    for fig_idx, img in enumerate(docling_document.pictures, start=1):
        if not isinstance(img, PictureItem):
            continue

        ref = img.get_ref().cref
        page_number = getattr(img, "page_number", None)
        bbox = img.bbox.model_dump() if hasattr(img, "bbox") else None
        image_uri = img.image.uri  # already base64 encoded

        doc_id += 1
        image_docs.append(
            Document(
                page_content="Image embedded as base64 string.",
                metadata={
                    "doc_id": doc_id,
                    "type": "image",
                    "filename": filename,
                    "ref": ref,
                    "page_number": page_number,
                    "figure_number": fig_idx,
                    "bbox": bbox,
                    "image_data_uri": image_uri
                }
            )
        )

    return {
        "texts": text_chunks,
        "tables": table_docs,
        "images": image_docs
    }
