"""Context-aware chunking engine for technical documentation."""

from __future__ import annotations

import re
from typing import List, Tuple
from core.models import Chunk, ChunkMetadata, Document


class ContextAwareChunker:
    """
    Context-aware chunking that parses structural document hierarchies (headers, sections, code blocks),
    preserves breadcrumbs (e.g. Document > Section > Subsection), and prepends contextual headers
    to prevent semantic drift in dense vector retrieval.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64, min_chunk_size: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        # Regex to match markdown headings #, ##, ###, ####
        self.header_regex = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

    def _extract_sections(self, content: str) -> List[Tuple[str, str]]:
        """
        Split document by markdown headings, tracking active heading breadcrumbs.
        Returns list of (section_breadcrumb, section_text).
        """
        lines = content.splitlines()
        sections: List[Tuple[str, List[str]]] = []
        current_breadcrumb = ["Introduction"]
        current_lines: List[str] = []

        header_stack = {}  # level -> title

        for line in lines:
            header_match = re.match(r"^(#{1,4})\s+(.+)$", line.strip())
            if header_match:
                # Flush previous lines if any
                if current_lines:
                    sections.append((" > ".join(current_breadcrumb), current_lines))
                    current_lines = []

                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                # Update heading stack
                header_stack[level] = title
                # Remove deeper levels
                levels_to_remove = [lvl for lvl in header_stack.keys() if lvl > level]
                for lvl in levels_to_remove:
                    del header_stack[lvl]

                current_breadcrumb = [header_stack[lvl] for lvl in sorted(header_stack.keys())]
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((" > ".join(current_breadcrumb), current_lines))

        return [(bc, "\n".join(lns).strip()) for bc, lns in sections if "\n".join(lns).strip()]

    def _split_text_with_overlap(self, text: str, max_chars: int, overlap: int) -> List[str]:
        """Split text into overlapping blocks at sentence or paragraph boundaries."""
        if len(text) <= max_chars:
            return [text]

        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_len = len(para)

            # If single paragraph exceeds max_chars, split by sentences
            if para_len > max_chars:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    if current_len + len(sentence) > max_chars and current_chunk:
                        chunk_str = "\n".join(current_chunk)
                        chunks.append(chunk_str)
                        # Keep overlap from the end
                        overlap_target = overlap
                        overlap_parts = []
                        running_overlap = 0
                        for part in reversed(current_chunk):
                            overlap_parts.insert(0, part)
                            running_overlap += len(part)
                            if running_overlap >= overlap_target:
                                break
                        current_chunk = overlap_parts
                        current_len = sum(len(p) for p in current_chunk)

                    current_chunk.append(sentence)
                    current_len += len(sentence)
            else:
                if current_len + para_len > max_chars and current_chunk:
                    chunk_str = "\n\n".join(current_chunk)
                    chunks.append(chunk_str)
                    overlap_parts = []
                    running_overlap = 0
                    for part in reversed(current_chunk):
                        overlap_parts.insert(0, part)
                        running_overlap += len(part)
                        if running_overlap >= overlap:
                            break
                    current_chunk = overlap_parts
                    current_len = sum(len(p) for p in current_chunk)

                current_chunk.append(para)
                current_len += para_len

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return [c for c in chunks if len(c) >= self.min_chunk_size]

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Deconstruct a document into context-aware chunks with prepended document metadata
        and hierarchical section breadcrumbs.
        """
        doc_meta = document.metadata
        sections = self._extract_sections(document.content)

        # Fallback if no markdown sections found
        if not sections:
            sections = [("Overview", document.content)]

        raw_chunks: List[Tuple[str, str]] = []  # (breadcrumb, text)

        for breadcrumb, section_text in sections:
            sub_chunks = self._split_text_with_overlap(
                section_text,
                max_chars=self.chunk_size,
                overlap=self.chunk_overlap,
            )
            for sub_chunk in sub_chunks:
                raw_chunks.append((breadcrumb, sub_chunk))

        total_chunks = len(raw_chunks)
        chunks: List[Chunk] = []

        for idx, (breadcrumb, raw_text) in enumerate(raw_chunks):
            chunk_id = f"{doc_meta.doc_id}_c{idx:03d}"

            # Context-aware prefix: prepending high-signal provenance
            context_header = (
                f"[DocID: {doc_meta.doc_id}] "
                f"[Document: {doc_meta.title}] "
                f"[Domain: {doc_meta.domain} | Type: {doc_meta.doc_type}] "
                f"[Section: {breadcrumb}]"
            )
            enriched_content = f"{context_header}\n{raw_text}"

            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                doc_id=doc_meta.doc_id,
                doc_title=doc_meta.title,
                doc_type=doc_meta.doc_type,
                domain=doc_meta.domain,
                tags=doc_meta.tags,
                section_path=breadcrumb,
                chunk_index=idx,
                total_chunks=total_chunks,
                char_count=len(enriched_content),
                token_count=len(enriched_content.split()),
            )

            chunk = Chunk(
                id=chunk_id,
                content=enriched_content,
                raw_content=raw_text,
                metadata=metadata,
            )
            chunks.append(chunk)

        return chunks
