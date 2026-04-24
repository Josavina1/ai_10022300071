# data_loader.py
# Author:  - Josavina - 10022300071
# CS4241 - Introduction to Artificial Intelligence - 2026
# Part A: Data Engineering & Preparation

import pandas as pd
import pdfplumber
import requests
import os
import re
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── CSV Loader ────────────────────────────────────────────────────────────────

def load_election_csv(path_or_url: str) -> pd.DataFrame:
    """
    Load and clean the Ghana Election Results CSV.
    Accepts a local file path or a raw GitHub URL.
    """
    logger.info(f"Loading election CSV from: {path_or_url}")

    if path_or_url.startswith("http"):
        # Convert GitHub blob URL to raw URL if needed
        raw_url = path_or_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        df = pd.read_csv(raw_url)
    else:
        df = pd.read_csv(path_or_url)

    logger.info(f"Raw CSV shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    # ── Cleaning Steps ──────────────────────────────────────────────────────
    # 1. Strip whitespace from column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # 2. Drop fully empty rows
    df.dropna(how="all", inplace=True)

    # 3. Fill remaining NaN with empty string for text columns
    df.fillna("", inplace=True)

    # 4. Remove duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info(f"Removed {before - len(df)} duplicate rows.")

    # 5. Strip whitespace from all string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    logger.info(f"Cleaned CSV shape: {df.shape}")
    return df


def csv_to_text_chunks_raw(df: pd.DataFrame) -> list[dict]:
    """
    Convert each CSV row into a readable text string.
    Returns a list of dicts with 'text' and 'source' keys.
    This is the RAW text before chunking — chunker.py will handle splitting.
    """
    documents = []
    for idx, row in df.iterrows():
        # Build a natural-language sentence from each row
        text = " | ".join([f"{col}: {val}" for col, val in row.items() if val != ""])
        documents.append({
            "text": text,
            "source": "Ghana_Election_Results",
            "row_index": idx
        })
    logger.info(f"Converted {len(documents)} CSV rows to raw text documents.")
    return documents


# ── PDF Loader ────────────────────────────────────────────────────────────────

def load_budget_pdf(path_or_url: str, save_path: str = "data/budget.pdf") -> str:
    """
    Load and extract text from the Ghana 2025 Budget PDF.
    Downloads the file if a URL is given.
    Returns the full extracted text as a string.
    """
    # Download if URL
    if path_or_url.startswith("http"):
        logger.info(f"Downloading budget PDF from: {path_or_url}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        response = requests.get(path_or_url, timeout=60)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        logger.info(f"PDF saved to {save_path}")
        local_path = save_path
    else:
        local_path = path_or_url

    # Extract text page by page
    logger.info(f"Extracting text from PDF: {local_path}")
    full_text = ""
    with pdfplumber.open(local_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                full_text += f"\n[PAGE {i+1}]\n{page_text}"

    logger.info(f"Extracted {len(full_text)} characters from PDF.")
    return full_text


def clean_pdf_text(raw_text: str) -> str:
    """
    Clean extracted PDF text:
    - Remove excessive whitespace and newlines
    - Remove page headers/footers patterns
    - Normalize unicode characters
    """
    # Remove lines that are just page numbers or short header/footer artifacts
    lines = raw_text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are just numbers (page numbers)
        if re.match(r"^\d+$", stripped):
            continue
        # Skip very short lines likely to be artifacts (less than 4 chars)
        if len(stripped) < 4 and stripped not in ["", " "]:
            continue
        cleaned_lines.append(stripped)

    cleaned = "\n".join(cleaned_lines)

    # Collapse multiple blank lines into one
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # Fix hyphenated line breaks (e.g., "econ-\nomy" → "economy")
    cleaned = re.sub(r"-\n(\w)", r"\1", cleaned)

    # Normalize spaces
    cleaned = re.sub(r" {2,}", " ", cleaned)

    logger.info(f"PDF text cleaned. Final length: {len(cleaned)} characters.")
    return cleaned


# ── Combined Loader ───────────────────────────────────────────────────────────

def load_all_data(
    csv_url: str = "https://github.com/GodwinDansoAcity/acitydataset/blob/main/Ghana_Election_Result.csv",
    pdf_url: str = "https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-andEconomic-Policy_v4.pdf",
    pdf_save_path: str = "data/budget.pdf"
) -> dict:
    """
    Master loader — loads and cleans both datasets.
    Returns a dict with:
        'election_df'    : cleaned DataFrame
        'election_docs'  : list of raw text dicts from CSV rows
        'budget_text'    : cleaned full PDF text string
    """
    logger.info("=== Loading all data sources ===")

    # Load CSV
    election_df = load_election_csv(csv_url)
    election_docs = csv_to_text_chunks_raw(election_df)

    # Load PDF
    raw_pdf = load_budget_pdf("data/budget.pdf", save_path=pdf_save_path)
    budget_text = clean_pdf_text(raw_pdf)

    logger.info("=== All data loaded successfully ===")
    return {
        "election_df": election_df,
        "election_docs": election_docs,
        "budget_text": budget_text
    }


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_all_data()
    print("\n--- Election DF Sample ---")
    print(data["election_df"].head(3))
    print("\n--- Election Doc Sample ---")
    print(data["election_docs"][0])
    print("\n--- Budget Text Sample (first 500 chars) ---")
    print(data["budget_text"][:500])