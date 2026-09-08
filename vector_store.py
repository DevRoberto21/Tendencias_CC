import json
import os
from typing import Any, Dict, List, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticStore:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_dir: str = "faiss_index",
    ) -> None:
        """Gerenciador do banco vetorial FAISS com persistência de metadados local."""
        self.index_dir = index_dir
        self.index_file = os.path.join(index_dir, "index.faiss")
        self.meta_file = os.path.join(index_dir, "metadata.json")

        # Garante a existência do diretório de armazenamento
        os.makedirs(self.index_dir, exist_ok=True)

        # Inicialização do modelo de embeddings
        self.encoder = SentenceTransformer(model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()

        # Metadados e índice FAISS
        self.metadata: List[Dict[str, Any]] = []
        self.index: Optional[faiss.IndexFlatIP] = None

        # Carrega índice do disco ou inicializa novo
        self._load_or_create_index()

    def _load_or_create_index(self) -> None:
        """Carrega o índice e os metadados existentes, ou cria novas estruturas."""
        if os.path.exists(self.index_file) and os.path.exists(self.meta_file):
            try:
                self.index = faiss.read_index(self.index_file)
                with open(self.meta_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception:
                self._init_empty_index()
        else:
            self._init_empty_index()

    def _init_empty_index(self) -> None:
        """Inicializa um índice FAISS baseado em produto interno (Similaridade de Cosseno com vetores normalizados)."""
        # IndexFlatIP é ideal para busca por Cosseno quando os vetores são normalizados (L2)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.metadata = []
        self._save()

    def _save(self) -> None:
        """Persiste o índice FAISS e os metadados no disco."""
        if self.index is not None:
            faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    @property
    def total_documents(self) -> int:
        """Retorna o número total de documentos armazenados."""
        return self.index.ntotal if self.index else 0

    def add_documents(
        self,
        documents: List[str],
        extra_meta: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Gera embeddings e adiciona novos documentos à base vetorial."""
        if not documents:
            return

        # Normalização L2 para garantir que Produto Interno funcione como Similaridade de Cosseno
        embeddings = self.encoder.encode(
            documents, convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

        start_id = self.total_documents
        self.index.add(embeddings)

        for i, doc in enumerate(documents):
            meta = {
                "id": start_id + i,
                "text": doc,
            }
            if extra_meta and i < len(extra_meta):
                meta.update(extra_meta[i])
            self.metadata.append(meta)

        self._save()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Realiza a consulta semântica e retorna os top_k documentos mais similares."""
        if not query.strip() or self.total_documents == 0:
            return []

        # Limita top_k ao número de documentos disponíveis
        top_k = min(top_k, self.total_documents)

        # Encode da query com normalização L2
        query_vector = self.encoder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                doc_data = dict(self.metadata[idx])
                # Converte float32 nativo para float Python padrão
                doc_data["score"] = float(score)
                results.append(doc_data)

        return results

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Retorna a lista completa dos metadados de documentos."""
        return self.metadata

    def clear(self) -> None:
        """Reseta e limpa toda a base vetorial."""
        self._init_empty_index()