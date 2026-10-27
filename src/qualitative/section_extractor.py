import re
import logging
from typing import Optional
from src.qualitative.models import ExtractionState, ExtractedSection

logger = logging.getLogger(__name__)

MIN_RISK_FACTORS_LENGTH = 1000
MIN_MDA_LENGTH = 1000

def _extract_primary(text: str) -> Optional[str]:
    # Look for Item 1A to Item 1B
    pattern = re.compile(r'(?i)(?:item\s+1a\.?\s*risk\s+factors)(.*?)(?:item\s+1b\.?\s*unresolved\s+staff\s+comments|item\s+2\.?\s*properties)', re.DOTALL)
    matches = pattern.findall(text)
    
    if not matches:
        return None
        
    # Filings often have a TOC mention first, so the longest match is usually the actual section
    longest_match = max(matches, key=len)
    return longest_match.strip()

def _extract_fallback(text: str) -> Optional[str]:
    # Looser search strategy
    pattern = re.compile(r'(?i)(?:risk\s+factors)(.*?)(?:unresolved\s+staff\s+comments|properties)', re.DOTALL)
    matches = pattern.findall(text)
    
    if not matches:
        return None
        
    longest_match = max(matches, key=len)
    return longest_match.strip()

def extract_risk_factors(ticker: str, accession_number: str, clean_text: str) -> ExtractedSection:
    extracted_text = _extract_primary(clean_text)
    state = ExtractionState.SUCCESS
    
    if extracted_text is None or len(extracted_text) < MIN_RISK_FACTORS_LENGTH:
        extracted_text = _extract_fallback(clean_text)
        state = ExtractionState.FALLBACK_USED
        
    if extracted_text is None or len(extracted_text) < MIN_RISK_FACTORS_LENGTH:
        extracted_text = ""
        state = ExtractionState.EXTRACTION_FAILED
        logger.warning(f"Failed to extract Risk Factors for {ticker} (Accession: {accession_number})")
        
    return ExtractedSection(
        ticker=ticker,
        accession_number=accession_number,
        section_name="risk_factors",
        raw_text=extracted_text,
        state=state
    )

def _extract_mda_primary(text: str) -> Optional[str]:
    # Look for Item 7 Management's Discussion and Analysis to Item 7A or Item 8
    pattern = re.compile(r'(?i)(?:item\s+7\.?\s*management(?:[\'\’]|&#8217;)?s\s+discussion\s+and\s+analysis)(.*?)(?:item\s+7a\.?\s*quantitative|item\s+8\.?\s*financial\s+statements)', re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return None
    longest_match = max(matches, key=len)
    return longest_match.strip()

def _extract_mda_fallback(text: str) -> Optional[str]:
    pattern = re.compile(r'(?i)(?:discussion\s+and\s+analysis\s+of\s+financial\s+condition)(.*?)(?:quantitative\s+and\s+qualitative\s+disclosures|financial\s+statements)', re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return None
    longest_match = max(matches, key=len)
    return longest_match.strip()

def extract_mda(ticker: str, accession_number: str, clean_text: str) -> ExtractedSection:
    extracted_text = _extract_mda_primary(clean_text)
    state = ExtractionState.SUCCESS
    
    if extracted_text is None or len(extracted_text) < MIN_MDA_LENGTH:
        extracted_text = _extract_mda_fallback(clean_text)
        state = ExtractionState.FALLBACK_USED
        
    if extracted_text is None or len(extracted_text) < MIN_MDA_LENGTH:
        extracted_text = ""
        state = ExtractionState.EXTRACTION_FAILED
        logger.warning(f"Failed to extract MD&A for {ticker} (Accession: {accession_number})")
        
    return ExtractedSection(
        ticker=ticker,
        accession_number=accession_number,
        section_name="mda",
        raw_text=extracted_text,
        state=state
    )

