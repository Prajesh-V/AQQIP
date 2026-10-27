import logging
from typing import Optional

from src.qualitative.html_cleaner import clean_html, preprocess_text
from src.qualitative.section_extractor import extract_risk_factors
from src.qualitative.chunker import chunk_text
from src.qualitative.embeddings import EmbeddingModel
from src.qualitative.chroma_store import ChromaStore
from src.qualitative.semantic_diff import compute_semantic_diff
from src.qualitative.llm_provider import LangchainLLMProvider
from src.qualitative.diff_store import persist_semantic_diff
from src.qualitative.models import QualitativeSignal, ExtractionState

logger = logging.getLogger(__name__)

class QualitativePipeline:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.chroma_store = ChromaStore()
        # Initialize LLM only if we actually do a diff to save startup time
        self.llm_provider = None

    def process_filing(self, ticker: str, accession_number: str, year: int, raw_text: str) -> Optional[dict]:
        """
        End-to-end processing of a single filing: clean -> extract -> chunk -> embed -> persist.
        Returns the data dictionary if successful, or None if extraction failed.
        """
        clean_txt = clean_html(raw_html=raw_text)
        clean_txt = preprocess_text(clean_txt)
        
        extracted = extract_risk_factors(ticker, accession_number, clean_txt)
        
        if extracted.state == ExtractionState.EXTRACTION_FAILED:
            logger.error(f"Extraction failed for {ticker} - {accession_number}")
            return None
            
        sentences = chunk_text(extracted.raw_text)
        
        if not sentences:
            logger.warning(f"No sentences extracted for {ticker} - {accession_number}")
            return None
            
        embeddings = self.embedding_model.encode(sentences)
        
        self.chroma_store.persist_chunks(
            ticker=ticker,
            accession_number=accession_number,
            year=year,
            section="risk_factors",
            sentences=sentences,
            embeddings=embeddings.tolist()
        )
        
        return {
            "sentences": sentences,
            "embeddings": embeddings
        }

    def compute_shift(
        self, 
        ticker: str, 
        cik: str, 
        year_to: int, 
        accession_to: str, 
        raw_text_to: str, 
        year_from: int, 
        accession_from: str, 
        raw_text_from: str
    ) -> Optional[QualitativeSignal]:
        """
        Computes the semantic shift between two filings.
        """
        # 1. Check ChromaDB or process Year To
        data_to = self.chroma_store.retrieve_filing(ticker, accession_to, "risk_factors")
        if not data_to:
            data_to = self.process_filing(ticker, accession_to, year_to, raw_text_to)
            if not data_to:
                return None
                
        # 2. Check ChromaDB or process Year From
        data_from = self.chroma_store.retrieve_filing(ticker, accession_from, "risk_factors")
        if not data_from:
            data_from = self.process_filing(ticker, accession_from, year_from, raw_text_from)
            if not data_from:
                return None
                
        import numpy as np
        emb_to = np.array(data_to["embeddings"]) if isinstance(data_to["embeddings"], list) else data_to["embeddings"]
        emb_from = np.array(data_from["embeddings"]) if isinstance(data_from["embeddings"], list) else data_from["embeddings"]
        
        # 3. Compute Deterministic Diff
        diff_results = compute_semantic_diff(
            sentences_n=data_to["sentences"],
            embeddings_n=emb_to,
            sentences_n_1=data_from["sentences"],
            embeddings_n_1=emb_from,
            threshold_added=0.8,
            threshold_identical=0.95
        )
        
        # 4. LLM Interpretation
        if self.llm_provider is None:
            self.llm_provider = LangchainLLMProvider()
            
        interpretation = self.llm_provider.interpret_diff(
            added=diff_results["added"],
            deleted=diff_results["deleted"],
            shifted=diff_results["shifted"]
        )
        
        signal = QualitativeSignal(
            ticker=ticker,
            cik=cik,
            section="risk_factors",
            year_from=year_from,
            year_to=year_to,
            accession_from=accession_from,
            accession_to=accession_to,
            cosine_similarity=diff_results["cosine_similarity"],
            semantic_shift_score=diff_results["shift_score"],
            added_sentences=diff_results["added"],
            deleted_sentences=diff_results["deleted"],
            shifted_sentences=diff_results["shifted"],
            llm_summary=interpretation.summary,
            llm_provider=self.llm_provider.provider_name,
            llm_model=self.llm_provider.model_name
        )
        
        # 5. Persist Diff
        persist_semantic_diff(signal.model_dump(mode="json"))
        
        return signal
