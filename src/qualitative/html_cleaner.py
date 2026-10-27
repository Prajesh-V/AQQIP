from bs4 import BeautifulSoup
import re
import html2text

def clean_html(raw_html: str) -> str:
    """
    Strips HTML tags and normalizes whitespace from raw SEC filing HTML.
    Uses BeautifulSoup for initial parsing and html2text for markdown-like text extraction.
    """
    if not raw_html:
        return ""
    
    # Use BeautifulSoup to remove script and style elements
    soup = BeautifulSoup(raw_html, "html.parser")
    for script_or_style in soup(["script", "style", "table"]):  # Remove tables as they usually add noise to risk factors
        script_or_style.decompose()

    # Get text
    text = soup.get_text(separator="\n")
    
    # Normalize whitespace
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

def preprocess_text(text: str) -> str:
    """
    Preprocesses cleaned text (e.g., lowercasing, removing excess symbols) if necessary,
    but for SEC filings we mostly just want clean sentences.
    """
    # Just a basic normalization, as we want to preserve semantic meaning for embeddings.
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
