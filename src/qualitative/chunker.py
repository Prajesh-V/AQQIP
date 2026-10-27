import nltk
from typing import List

def chunk_text(text: str) -> List[str]:
    """
    Chunks text into sentences.
    NLTK's punkt tokenizer is required.
    """
    if not text:
        return []
    
    # We expect 'punkt_tab' or 'punkt' to be downloaded
    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        sentences = nltk.sent_tokenize(text)
        
    # Clean up empty or very short sentences
    valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return valid_sentences
