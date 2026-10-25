"""Document loaders and metadata extractors for multi-format technical documents."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional
from core.models import Document, DocumentMetadata


class DocumentLoader:
    """Loads technical documents from local files and directories."""

    @staticmethod
    def _extract_markdown_title(content: str) -> Optional[str]:
        """Extract first # header as title if available."""
        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return None

    @classmethod
    def load_file(cls, file_path: str | Path) -> Document:
        """Load a single document from disk."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        stem = path.stem
        ext = path.suffix.lower()

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        title = cls._extract_markdown_title(content) or stem.replace("_", " ").title()
        doc_type = "technical_doc"
        domain = "engineering"
        tags = []

        # Parse frontmatter if present
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter_text = parts[1]
                content = parts[2].strip()
                for line in frontmatter_text.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        k = k.strip().lower()
                        v = v.strip().strip('"').strip("'")
                        if k == "title":
                            title = v
                        elif k == "doc_type":
                            doc_type = v
                        elif k == "domain":
                            domain = v
                        elif k == "tags":
                            tags = [t.strip() for t in v.split(",") if t.strip()]

        metadata = DocumentMetadata(
            doc_id=stem,
            title=title,
            doc_type=doc_type,
            domain=domain,
            tags=tags,
            source_path=str(path.resolve()),
        )

        return Document(id=stem, content=content, metadata=metadata)

    @classmethod
    def load_directory(
        cls, directory_path: str | Path, extensions: Optional[List[str]] = None
    ) -> List[Document]:
        """Load all matching documents from a directory recursively."""
        dir_path = Path(directory_path)
        if not dir_path.exists():
            return []

        if extensions is None:
            extensions = [".md", ".txt", ".json"]

        documents: List[Document] = []
        for file_path in dir_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                try:
                    if file_path.suffix.lower() == ".json":
                        # Attempt to load json document or list of documents
                        with open(file_path, "r", encoding="utf-8") as jf:
                            data = json.load(jf)
                            if isinstance(data, list):
                                for item in data:
                                    if "content" in item:
                                        meta = item.get("metadata", {})
                                        doc_id = item.get("id") or meta.get("doc_id") or f"{file_path.stem}_{len(documents)}"
                                        doc_metadata = DocumentMetadata(
                                            doc_id=doc_id,
                                            title=meta.get("title", doc_id),
                                            doc_type=meta.get("doc_type", "technical_doc"),
                                            domain=meta.get("domain", "general"),
                                            tags=meta.get("tags", []),
                                            source_path=str(file_path),
                                        )
                                        documents.append(Document(id=doc_id, content=item["content"], metadata=doc_metadata))
                            elif isinstance(data, dict) and "content" in data:
                                meta = data.get("metadata", {})
                                doc_id = data.get("id") or meta.get("doc_id") or file_path.stem
                                doc_metadata = DocumentMetadata(
                                    doc_id=doc_id,
                                    title=meta.get("title", doc_id),
                                    doc_type=meta.get("doc_type", "technical_doc"),
                                    domain=meta.get("domain", "general"),
                                    tags=meta.get("tags", []),
                                    source_path=str(file_path),
                                )
                                documents.append(Document(id=doc_id, content=data["content"], metadata=doc_metadata))
                    else:
                        documents.append(cls.load_file(file_path))
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")

        return documents
