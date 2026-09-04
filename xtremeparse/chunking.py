"""Deterministic chunker: newlines are layout truth — chunks respect
them unless they provably carry no breaks — with a three-tier fallback
(markdown structure, sentence punctuation, character windows) for the
shapes where they don't.

The same input always yields the same chunks, every chunk is an exact
substring of the (newline-normalized) input in reading order, and any
input — including a single line without punctuation — yields chunks.
Chunk ids are list positions.
"""

from __future__ import annotations

import re
from statistics import fmean, pstdev

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

    OCR and converted text break lines at layout boundaries, so a line
    is a chunk as it stands — the band a heading opens stays
    addressable on its own (measured: 250-char windowing buried a
    skills band inside another unit's chunk and the router could not
    point at it). Shapes whose newlines carry no meaning fall to
    ``_breaks_meaningless`` and the tiered path; an oversized line in a
    respected text runs that path within itself. Every chunk is at
    most ``max_chars`` long.
    """
    if not (text := normalize_newlines(text)).strip():
        return []
    lines = [l for l in text.split('\n') if l.strip()]
    if _breaks_meaningless(lines, max_chars):
        return _tiers(text, max_chars)
    return [c for line in lines for c in _tiers(line, max_chars)]


def _breaks_meaningless(lines: list[str], max_chars: int) -> bool:
    """Whether the newlines in ``lines`` carry no layout meaning — the
    two shapes where respecting them would split sentences or join
    nothing: a mean length past the ceiling (prose dumped without
    breaks — a lone line included), and box-width wrapping (near-zero
    length variance, every line a full measure, a mid-sentence cut)."""
    lengths = [len(l) for l in lines]
    mean = fmean(lengths)
    if mean > max_chars:
        return True
    return pstdev(lengths) <= max(2.0, 0.1 * mean)


def _tiers(text: str, max_chars: int) -> list[str]:
    """The tiered split of text whose newlines carry no meaning:
    markdown structure, sentence pieces, entry runs kept unmerged,
    greedy packing."""
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
