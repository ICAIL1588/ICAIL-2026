"""
convert.py — FORCE OCR for every page of every PDF in ./pdfs to UTF-8 .txt in ./txt.

This is the correct approach for PDFs with broken font encodings (gibberish extraction).
It renders each page to an image and OCRs it with Tesseract.

macOS requirements:
  - brew install tesseract
Python (inside project venv):
  - python -m pip install pytesseract pypdfium2 Pillow

Run (from project root):
  source .venv/bin/activate
  python convert.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pypdfium2 as pdfium
import pytesseract


INPUT_DIR = Path("pdfs")
OUTPUT_DIR = Path("txt")

PAGE_SEPARATOR = "\n\n"
OCR_LANG = "eng"          # change to "eng+<lang>" if you have language packs installed
RENDER_SCALE = 6.0        # increase (e.g., 3.0) for better OCR if pages are small; slower


def require_venv() -> None:
    if "VIRTUAL_ENV" not in os.environ:
        raise RuntimeError("Activate venv first: source .venv/bin/activate")


def ensure_folders() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Missing input folder: {INPUT_DIR.resolve()}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(s: str) -> str:
    """
    Conservative normalization suitable for legal text:
    - Normalize line endings
    - Strip trailing whitespace per line
    - Preserve paragraph breaks (no aggressive reflow)
    """
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in s.split("\n")]
    return "\n".join(lines).strip()


def ocr_pdf_to_text(pdf_path: Path) -> str:
    doc = pdfium.PdfDocument(str(pdf_path))
    pages_out: list[str] = []

    try:
        for page_index in range(len(doc)):
            page = doc.get_page(page_index)
            try:
                pil_image = page.render(scale=RENDER_SCALE).to_pil()
            finally:
                page.close()

            text = pytesseract.image_to_string(pil_image, lang=OCR_LANG) or ""
            text = normalize_text(text)

            if not text:
                text = f"[Page {page_index + 1}: OCR produced no text]"

            pages_out.append(text)

    finally:
        doc.close()

    return PAGE_SEPARATOR.join(pages_out)


def main() -> None:
    require_venv()
    ensure_folders()

    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No .pdf files found in: {INPUT_DIR.resolve()}")

    for pdf_file in pdf_files:
        out_file = OUTPUT_DIR / f"{pdf_file.stem}.txt"
        text = ocr_pdf_to_text(pdf_file)
        out_file.write_text(text, encoding="utf-8")
        print(f"OCR converted: {pdf_file.name} -> {out_file.name}")


if __name__ == "__main__":
    main()