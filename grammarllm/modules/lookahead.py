"""
lookahead.py
============
Token-boundary lookahead (g_t_r): vocab trie + DFS over PDA fragment
transitions, so the model can emit merged tokens that span grammar-terminal
boundaries or end mid-terminal.

Spec: docs/superpowers/specs/2026-07-08-token-boundary-lookahead-design.md
Regex-terminal crossing is out of scope (L2) — see
docs/superpowers/specs/2026-07-08-regex-lookahead-future-work.md
"""

import logging

# Reserved key in map_terminal_tokens carrying the NAMES of regex terminals.
# Written by generate_token_maps, read (and skipped) by PushdownAutomaton.
REGEX_TERMINALS_KEY = "__regex_terminal_names__"


class VocabTrie:
    """Character trie over the tokenizer's raw vocabulary strings
    (surface alphabet: byte-level BPE forms Ġ/Ċ/…)."""

    __slots__ = ("children", "token_id")

    def __init__(self):
        self.children = {}
        self.token_id = None

    def child(self, ch):
        return self.children.get(ch)

    @classmethod
    def from_tokenizer(cls, tokenizer):
        root = cls()
        for tok_str, tok_id in tokenizer.get_vocab().items():
            node = root
            for ch in tok_str:
                nxt = node.children.get(ch)
                if nxt is None:
                    nxt = cls()
                    node.children[ch] = nxt
                node = nxt
            node.token_id = tok_id
        return root


_TRIE_CACHE: dict = {}


def get_vocab_trie(tokenizer):
    """One trie per tokenizer, built once (~1s for 150k vocab) and cached."""
    key = getattr(tokenizer, "name_or_path", None) or id(tokenizer)
    trie = _TRIE_CACHE.get(key)
    if trie is None:
        trie = VocabTrie.from_tokenizer(tokenizer)
        _TRIE_CACHE[key] = trie
        logging.info(f"VocabTrie built for {key}")
    return trie
