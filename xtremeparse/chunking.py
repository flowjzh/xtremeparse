"""Deterministic three-tier chunker: markdown structure, sentence
punctuation, character windows.

The same input always yields the same chunks, every chunk is an exact
substring of the (newline-normalized) input in reading order, and any
input — including a single line without punctuation — yields chunks.
Chunk ids are list positions.
"""

from __future__ import annotations

import re

_HEADING = re.compile(r'^#{1,6}\s')
_LIST_ITEM = re.compile(r'^\s*(?:[-*+]+\s+|•\s*|\d+[、)]\s*|\d+\.\s+)')
_TABLE_ROW = re.compile(r'^\s*\|')
# Sentence enders, CJK and latin; '.' only when not followed by a digit so
# dates like 2015.3 never split.
_SENTENCE_END = re.compile(r'(?<=[。．！？；!?;])|(?<=\.)(?!\d)')

MAX_CHARS = 250


def normalize_newlines(text: str) -> str:
    """Owned at extraction entry: chunks and the shared prompt prefix
    must agree on newlines."""
    return text.replace('\r\n', '\n').replace('\r', '\n')


def chunk_text(text: str, *, max_chars: int = MAX_CHARS) -> list[str]:
    """Split text into chunks, numbered by position.

    Tier 1 splits on structure: headings open blocks that absorb the plain
    lines after them, list items and table rows stand alone, blank lines
    end paragraphs. Tier 2 sentence-splits oversized blocks and greedily
    packs fragments up to ``max_chars``. Tier 3 hard-slices whatever still
    exceeds the ceiling (dense text without punctuation). Every chunk is
    at most ``max_chars`` long.
    """
    if not (text := normalize_newlines(text)).strip():
        return []
    return [c for block in _blocks(text) for c in _pack(_SENTENCE_END.split(block), max_chars)]


def _blocks(text: str) -> list[str]:
    blocks, para = [], []

    def flush():
        if para:
            blocks.append('\n'.join(para))
            para.clear()

    for line in text.split('\n'):
        if _HEADING.match(line):
            flush()
            para.append(line)
        elif not line.strip():
            flush()
        elif _LIST_ITEM.match(line) or _TABLE_ROW.match(line):
            flush()
            blocks.append(line)
        else:
            para.append(line)
    flush()
    return blocks


def _pack(pieces: list[str], max_chars: int) -> list[str]:
    chunks, current = [], ''
    for piece in pieces:
        if len(piece) > max_chars:
            if current:
                chunks.append(current)
                current = ''
            chunks.extend(piece[i:i + max_chars] for i in range(0, len(piece), max_chars))
        elif len(current) + len(piece) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current += piece
    if current:
        chunks.append(current)
    return chunks
