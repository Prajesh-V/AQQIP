import chromadb
from typing import List, Dict, Any, Optional
from config.settings import get_settings

class ChromaStore:
    def __init__(self):
        settings = get_settings()
        self.client = chromadb.PersistentClient(path=str(settings.chroma.persist_path))
        
    def _get_collection_name(self, ticker: str) -> str:
        # ChromaDB requires valid collection names, e.g., lowercase alphanumeric
        clean_ticker = ''.join(c.lower() for c in ticker if c.isalnum())
        return f"filings_{clean_ticker}"
        
    def get_collection(self, ticker: str):
        collection_name = self._get_collection_name(ticker)
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
    def persist_chunks(
        self, 
        ticker: str, 
        accession_number: str, 
        year: int, 
        section: str, 
        sentences: List[str], 
        embeddings: List[List[float]]
    ) -> None:
        """
        Persists chunks and embeddings into ChromaDB for a specific filing.
        """
        if not sentences:
            return
            
        collection = self.get_collection(ticker)
        
        ids = []
        metadatas = []
        documents = []
        
        for i, sentence in enumerate(sentences):
            chunk_id = f"{accession_number}_{section}_{i}"
            ids.append(chunk_id)
            documents.append(sentence)
            metadatas.append({
                "ticker": ticker,
                "accession_number": accession_number,
                "year": year,
                "section": section,
                "chunk_index": i
            })
            
        # We process in batches to avoid ChromaDB limits if necessary, 
        # but for typical Risk Factors, ~500-2000 sentences easily fit in one go.
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def retrieve_filing(self, ticker: str, accession_number: str, section: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves all chunks and embeddings for a specific filing.
        """
        collection = self.get_collection(ticker)
        
        # Query by metadata
        results = collection.get(
            where={
                "$and": [
                    {"accession_number": accession_number},
                    {"section": section}
                ]
            },
            include=["documents", "embeddings", "metadatas"]
        )
        
        if not results or not results["ids"]:
            return None
            
        # Sort by chunk_index to reconstruct order
        combined = list(zip(results["ids"], results["documents"], results["embeddings"], results["metadatas"]))
        combined.sort(key=lambda x: x[3]["chunk_index"])
        
        return {
            "ids": [x[0] for x in combined],
            "sentences": [x[1] for x in combined],
            "embeddings": [x[2] for x in combined],
            "metadatas": [x[3] for x in combined]
        }
