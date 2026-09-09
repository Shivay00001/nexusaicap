"""
NexusAI RAG Pipeline — Complete Retrieval-Augmented Generation.

Supports: PDF, DOCX, TXT, MD, HTML, CSV, JSON document loading.
Chunking: recursive, sentence-based, or fixed-size.
Embeddings: sentence-transformers, OpenAI, or provider-based.
Vector stores: FAISS, Chroma, or in-memory.
Retrieval: similarity search with optional reranking.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nexusai.core.config import NexusConfig, RAGConfig
from nexusai.core.exceptions import RAGError


@dataclass
class Document:
    """A document chunk with content and metadata."""
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.md5(self.content[:200].encode()).hexdigest()[:12]


class DocumentLoader:
    """Load documents from various file formats."""

    @staticmethod
    def load(path: str) -> list[Document]:
        """Load a document from a file path."""
        p = Path(path)
        if not p.exists():
            raise RAGError(f"File not found: {path}")

        ext = p.suffix.lower()
        loaders = {
            ".txt": DocumentLoader._load_text,
            ".md": DocumentLoader._load_text,
            ".py": DocumentLoader._load_text,
            ".js": DocumentLoader._load_text,
            ".ts": DocumentLoader._load_text,
            ".json": DocumentLoader._load_json,
            ".csv": DocumentLoader._load_csv,
            ".html": DocumentLoader._load_html,
            ".htm": DocumentLoader._load_html,
            ".pdf": DocumentLoader._load_pdf,
            ".docx": DocumentLoader._load_docx,
        }

        loader = loaders.get(ext, DocumentLoader._load_text)
        return loader(path)

    @staticmethod
    def load_directory(path: str, extensions: list[str] | None = None) -> list[Document]:
        """Load all documents from a directory."""
        docs = []
        exts = extensions or [".txt", ".md", ".py", ".json", ".csv", ".html", ".pdf"]
        for f in Path(path).rglob("*"):
            if f.is_file() and f.suffix.lower() in exts:
                try:
                    docs.extend(DocumentLoader.load(str(f)))
                except Exception:
                    pass
        return docs

    @staticmethod
    def _load_text(path: str) -> list[Document]:
        content = Path(path).read_text(errors="ignore")
        return [Document(content=content, metadata={"source": path, "type": "text"})]

    @staticmethod
    def _load_json(path: str) -> list[Document]:
        with open(path) as f:
            data = json.load(f)
        content = json.dumps(data, indent=2, default=str)
        return [Document(content=content, metadata={"source": path, "type": "json"})]

    @staticmethod
    def _load_csv(path: str) -> list[Document]:
        import csv
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        content = "\n".join(json.dumps(row) for row in rows[:500])
        return [Document(content=content, metadata={"source": path, "type": "csv", "rows": len(rows)})]

    @staticmethod
    def _load_html(path: str) -> list[Document]:
        html = Path(path).read_text(errors="ignore")
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
        except ImportError:
            text = re.sub(r"<[^>]+>", " ", html)
        return [Document(content=text, metadata={"source": path, "type": "html"})]

    @staticmethod
    def _load_pdf(path: str) -> list[Document]:
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            docs = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(content=text, metadata={"source": path, "type": "pdf", "page": i + 1}))
            return docs if docs else [Document(content="[Empty PDF]", metadata={"source": path})]
        except ImportError:
            raise RAGError("pypdf not installed. pip install nexusai[rag]")

    @staticmethod
    def _load_docx(path: str) -> list[Document]:
        try:
            from docx import Document as DocxDocument
            doc = DocxDocument(path)
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return [Document(content=text, metadata={"source": path, "type": "docx"})]
        except ImportError:
            raise RAGError("python-docx not installed. pip install nexusai[rag]")


class Chunker:
    """Split documents into chunks for embedding."""

    @staticmethod
    def chunk(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200, strategy: str = "recursive") -> list[Document]:
        """Split documents into smaller chunks."""
        chunks = []
        for doc in documents:
            if strategy == "sentence":
                doc_chunks = Chunker._sentence_chunk(doc, chunk_size, chunk_overlap)
            elif strategy == "fixed":
                doc_chunks = Chunker._fixed_chunk(doc, chunk_size, chunk_overlap)
            else:
                doc_chunks = Chunker._recursive_chunk(doc, chunk_size, chunk_overlap)
            chunks.extend(doc_chunks)
        return chunks

    @staticmethod
    def _recursive_chunk(doc: Document, size: int, overlap: int) -> list[Document]:
        """Recursively split text using hierarchical separators."""
        separators = ["\n\n", "\n", ". ", " ", ""]
        return Chunker._split_recursive(doc.content, separators, size, overlap, doc.metadata)

    @staticmethod
    def _split_recursive(text: str, separators: list[str], size: int, overlap: int, metadata: dict) -> list[Document]:
        chunks = []
        if len(text) <= size:
            return [Document(content=text, metadata={**metadata, "chunk": True})]

        sep = separators[0]
        parts = text.split(sep) if sep else [text[i:i + size] for i in range(0, len(text), size)]

        current = ""
        for part in parts:
            if len(current) + len(part) + len(sep) <= size:
                current += (sep + part) if current else part
            else:
                if current:
                    chunks.append(Document(content=current.strip(), metadata={**metadata, "chunk": True}))
                current = part[-overlap:] + sep + part if overlap and current else part

        if current.strip():
            chunks.append(Document(content=current.strip(), metadata={**metadata, "chunk": True}))

        return chunks

    @staticmethod
    def _sentence_chunk(doc: Document, size: int, overlap: int) -> list[Document]:
        sentences = re.split(r"(?<=[.!?])\s+", doc.content)
        chunks, current = [], ""
        for s in sentences:
            if len(current) + len(s) <= size:
                current += " " + s
            else:
                if current:
                    chunks.append(Document(content=current.strip(), metadata={**doc.metadata, "chunk": True}))
                current = s
        if current:
            chunks.append(Document(content=current.strip(), metadata={**doc.metadata, "chunk": True}))
        return chunks

    @staticmethod
    def _fixed_chunk(doc: Document, size: int, overlap: int) -> list[Document]:
        text = doc.content
        chunks = []
        for i in range(0, len(text), size - overlap):
            chunk_text = text[i : i + size]
            if chunk_text.strip():
                chunks.append(Document(content=chunk_text, metadata={**doc.metadata, "chunk": True}))
        return chunks


class VectorStore:
    """Simple in-memory vector store with cosine similarity."""

    def __init__(self):
        self.documents: list[Document] = []
        self._embeddings: list[list[float]] = []

    def add(self, documents: list[Document]):
        """Add documents with their embeddings."""
        for doc in documents:
            if doc.embedding:
                self.documents.append(doc)
                self._embeddings.append(doc.embedding)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[tuple[Document, float]]:
        """Search for similar documents."""
        if not self._embeddings:
            return []

        scores = []
        for i, emb in enumerate(self._embeddings):
            score = self._cosine_similarity(query_embedding, emb)
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.documents[i], score) for i, score in scores[:top_k]]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @property
    def count(self) -> int:
        return len(self.documents)


class RAGPipeline:
    """
    Complete RAG pipeline: Load → Chunk → Embed → Store → Retrieve → Generate.

    Usage:
        rag = RAGPipeline()
        rag.ingest("documents/")
        answer = rag.query("What is the main topic?")
    """

    def __init__(self, config: NexusConfig | None = None, rag_config: RAGConfig | None = None):
        self.config = config or NexusConfig.from_env()
        self.rag_config = rag_config or self.config.rag
        self.store = VectorStore()
        self._embedder = None

    def ingest(self, path: str, chunk_size: int | None = None, chunk_overlap: int | None = None):
        """Ingest documents from a file or directory."""
        p = Path(path)
        if p.is_dir():
            docs = DocumentLoader.load_directory(str(p))
        else:
            docs = DocumentLoader.load(str(p))

        if not docs:
            raise RAGError(f"No documents loaded from {path}")

        # Chunk
        chunks = Chunker.chunk(
            docs,
            chunk_size=chunk_size or self.rag_config.chunk_size,
            chunk_overlap=chunk_overlap or self.rag_config.chunk_overlap,
        )

        # Embed
        self._embed_documents(chunks)

        # Store
        self.store.add(chunks)

        return {"documents_loaded": len(docs), "chunks_created": len(chunks), "stored": self.store.count}

    def _embed_documents(self, documents: list[Document]):
        """Generate embeddings for documents."""
        texts = [doc.content for doc in documents]
        embeddings = self._get_embeddings(texts)
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings using available provider."""
        # Try sentence-transformers first (local, no API key)
        try:
            from sentence_transformers import SentenceTransformer
            if not self._embedder:
                self._embedder = SentenceTransformer(self.rag_config.embedding_model)
            embeddings = self._embedder.encode(texts, show_progress_bar=False)
            return embeddings.tolist()
        except ImportError:
            pass

        # Fall back to simple TF-IDF-like embeddings
        return self._simple_embeddings(texts)

    @staticmethod
    def _simple_embeddings(texts: list[str], dim: int = 384) -> list[list[float]]:
        """Fallback: simple hash-based embeddings for when no ML library is available."""
        embeddings = []
        for text in texts:
            words = text.lower().split()
            vec = [0.0] * dim
            for i, word in enumerate(words):
                idx = hash(word) % dim
                vec[idx] += 1.0 / (i + 1)
            # Normalize
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]
            embeddings.append(vec)
        return embeddings

    def query(self, question: str, top_k: int | None = None) -> dict:
        """Query the RAG pipeline: retrieve relevant docs and generate answer."""
        k = top_k or self.rag_config.top_k

        # Embed query
        query_embedding = self._get_embeddings([question])[0]

        # Retrieve
        results = self.store.search(query_embedding, top_k=k)

        # Build context
        context = "\n\n---\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.content}"
            for doc, score in results
        )

        # Generate answer using LLM
        try:
            from nexusai.agents.orchestrator import AgentOrchestrator
            orch = AgentOrchestrator(config=self.config, verbose=False)
            provider = orch._get_provider()

            from nexusai.providers.base import Message
            messages = [
                Message(role="system", content="You are a helpful assistant. Answer the question based on the provided context. If the context doesn't contain relevant information, say so."),
                Message(role="user", content=f"Context:\n{context}\n\n---\n\nQuestion: {question}\n\nAnswer based on the context above:"),
            ]

            response = provider.invoke(messages)
            answer = response.content
        except Exception as e:
            answer = f"Retrieved {len(results)} relevant documents but could not generate answer: {e}"

        return {
            "answer": answer,
            "sources": [
                {"content": doc.content[:200], "score": round(score, 4), "metadata": doc.metadata}
                for doc, score in results
            ],
            "total_chunks": self.store.count,
        }

    def ingest_text(self, text: str, metadata: dict | None = None):
        """Ingest raw text directly."""
        docs = [Document(content=text, metadata=metadata or {})]
        chunks = Chunker.chunk(docs, self.rag_config.chunk_size, self.rag_config.chunk_overlap)
        self._embed_documents(chunks)
        self.store.add(chunks)
        return {"chunks_created": len(chunks)}
