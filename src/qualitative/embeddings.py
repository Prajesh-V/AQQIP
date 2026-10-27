import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from config.settings import get_settings

class EmbeddingModel:
    def __init__(self):
        settings = get_settings()
        self.model_name = settings.embedding.model_name
        self.device = settings.embedding.device
        self.model = SentenceTransformer(self.model_name, device=self.device)
        
    def encode(self, sentences: List[str]) -> np.ndarray:
        """
        Encodes a list of sentences into embeddings.
        Returns a numpy array of shape (len(sentences), embedding_dim).
        """
        if not sentences:
            return np.array([])
        settings = get_settings()
        embeddings = self.model.encode(
            sentences,
            batch_size=settings.embedding.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings
