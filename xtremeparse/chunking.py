"""Deterministic three-tier chunker: markdown structure, sentence
punctuation, character windows — plus runs of parallel entries kept
unmerged so each instance gets its own chunk.

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
_CLAUSE_END = '；;'  # the ender enumerated lists chain with — the entry
# run's signal; one inventory shared by the sentence splitter below
# and the entry detector in _pack_entries
_CLAUSE_END_TUPLE = tuple(_CLAUSE_END)
# Sentence enders, CJK and latin; '.' only before whitespace or end so
# tickers like 06228.HK never split (dates like 2015.3 still hold).
_SENTENCE_END = re.compile(rf'(?<=[。．！？{_CLAUSE_END}!?;])|(?<=\.)(?=\s|$)')
# Parallel entries a converter left mid-line: symbol bullets (C1 controls,
# •) it failed to lift to line starts, and the clause ender above.
_ENTRY = re.compile(r'^\s*[\x80-\x9f•]')

MAX_CHARS = 250


def normalize_newlines(text: str) -> str:
    """Owned at extraction entry: chunks and the shared prompt prefix
    must agree on newlines."""
    return text.replace('\r\n', '\n').replace('\r', '\n')


def chunk_text(text: str, *, max_chars: int = MAX_CHARS) -> list[str]:
    """Split text into chunks, numbered by position.

    Tier 1 splits on structure: headings open blocks that absorb the plain
    lines after them, list items and table rows stand alone, blank lines
    end paragraphs. Tier 2 sentence-splits oversized blocks, keeps runs of
    parallel entries (bullet-prefixed or ;-chained clauses) unmerged so
    each stands as its own chunk, and greedily packs the rest up to
    ``max_chars``. Tier 3 hard-slices whatever still exceeds the ceiling
    (dense text without punctuation). Every chunk is at most ``max_chars``
    long.
    """
    if not (text := normalize_newlines(text)).strip():
        return []
    return [c for block in _blocks(text)
            for c in _pack_entries(_SENTENCE_END.split(block), max_chars)]


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


def _pack_entries(pieces: list[str], max_chars: int) -> list[str]:
    """Pack pieces, but keep runs of parallel entries unmerged — the
    router maps items at chunk granularity, so a chunk holding several
    instances can only be extracted whole. Fragments a sentence end cut
    out of an unclosed bracket are healed back into their clause first.
    A piece is an entry when it starts with a symbol bullet or ends
    with a clause ender; inside a run of two or more, each stands
    alone, a lone one packs as usual."""
    healed = _heal(pieces)
    entry = [bool(_ENTRY.match(p)) or p.rstrip().endswith(_CLAUSE_END_TUPLE)
             for p in healed]
    groups, current = [], []

    def flush():
        if current:
            groups.append(current[:])
            current.clear()

    for piece, run in zip(healed, _in_run(entry)):
        if run:
            flush()
            groups.append([piece])
        else:
            current.append(piece)
    flush()
    return [c for group in groups for c in _pack(group, max_chars)]


def _heal(pieces: list[str]) -> list[str]:
    """Re-join fragments a sentence end inside a bracket cut off — an
    unclosed '(' means the clause continues in the next piece
    ('某公司(CHA.' | ' US)；'), whose entry mark would otherwise give
    the fragment its own chunk and, in an entry run, its own item
    with a nonsense budget."""
    out = []
    for piece in pieces:
        if out and not _ENTRY.match(piece) and _unclosed(out[-1]):
            out[-1] += piece
        else:
            out.append(piece)
    return out


def _unclosed(piece: str) -> bool:
    return piece.count('(') + piece.count('（') > piece.count(')') + piece.count('）')


def _in_run(entry: list[bool]) -> list[bool]:
    """Where the entry marks form a run of two or more."""
    padded = [False, *entry, False]
    return [m and (l or r) for m, l, r in zip(entry, padded, padded[2:])]


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
