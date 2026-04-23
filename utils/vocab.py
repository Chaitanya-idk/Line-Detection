"""
utils/vocab.py
==============
Character-level vocabulary for CTC-based handwritten text recognition.

Index convention
----------------
  0          → CTC blank token (never assigned to a real character)
  1 … N      → sorted unique characters found in training transcriptions

Example
-------
>>> vocab = Vocab(["hello", "world"])
>>> vocab.encode("hello")
[4, 3, 6, 6, 7]          # indices depend on sorted order
>>> vocab.decode([4, 4, 3, 0, 6, 6, 7])   # CTC greedy collapse
'hello'
"""

from __future__ import annotations

from typing import List


class Vocab:
    """
    Character vocabulary built from a list of transcript strings.

    Attributes
    ----------
    char2idx : dict[str, int]  — maps character → integer index (1-based)
    idx2char : dict[int, str]  — maps integer index → character
    size     : int             — number of real characters (excluding blank)
    """

    BLANK_IDX: int = 0   # CTC blank token is always at index 0

    def __init__(self, transcripts: List[str]) -> None:
        """
        Build the vocabulary from a list of raw transcription strings.

        Parameters
        ----------
        transcripts : list of str
            All training transcriptions.  Characters are collected,
            de-duplicated, sorted, then assigned indices starting at 1.
        """
        # Collect every unique character across all transcriptions
        unique_chars = set()
        for text in transcripts:
            unique_chars.update(text)

        # Sorted for reproducibility
        sorted_chars = sorted(unique_chars)

        # Index 0 is reserved for the CTC blank token
        # Real characters start at index 1
        self.char2idx: dict[str, int] = {
            ch: idx + 1 for idx, ch in enumerate(sorted_chars)
        }
        self.idx2char: dict[int, str] = {
            idx: ch for ch, idx in self.char2idx.items()
        }

        # Size = number of real characters (does NOT count blank)
        self.size: int = len(sorted_chars)

    # ------------------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        """
        Convert a string to a list of integer indices.

        Unknown characters (not seen during training) are silently skipped.

        Parameters
        ----------
        text : str

        Returns
        -------
        list[int]  — length may be ≤ len(text) if unknowns are dropped
        """
        return [self.char2idx[ch] for ch in text if ch in self.char2idx]

    # ------------------------------------------------------------------
    def decode(self, indices: List[int], remove_blanks: bool = True) -> str:
        """
        CTC greedy decode: collapse consecutive duplicate indices,
        then optionally remove blank tokens.

        Parameters
        ----------
        indices       : list[int]   — raw argmax output from the model
        remove_blanks : bool        — if True, strip blank (index 0) tokens

        Returns
        -------
        str — decoded character string
        """
        # Step 1: Remove consecutive duplicates (CTC merge rule)
        collapsed: List[int] = []
        prev: int = -1
        for idx in indices:
            if idx != prev:
                collapsed.append(idx)
                prev = idx

        # Step 2: Remove blank tokens
        if remove_blanks:
            collapsed = [idx for idx in collapsed if idx != self.BLANK_IDX]

        # Step 3: Map indices back to characters; skip unknown indices
        return "".join(
            self.idx2char[idx] for idx in collapsed if idx in self.idx2char
        )

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"Vocab(size={self.size}, chars={sorted(self.char2idx.keys())!r})"
