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


def lookahead_paths(pda, trie):
    """
    g_t_r: DFS over PDA fragment transitions pruned by the vocab trie.

    Returns { token_id: [ (fragments, chars_into_last), ... ] } — for every
    vocabulary token realizable from the current (stack, residue) state, ALL
    the grammar paths that token is compatible with. Includes merged tokens
    spanning terminal boundaries and tokens ending mid-terminal. Depth-0 exact
    matches reproduce the legacy mask, so the legacy valid set is always a
    subset of this one.

    Why a LIST of paths and not one
    -------------------------------
    The same token can be compatible with several places in the grammar. Real
    case, children 'osteoarthritis' and 'osteoporosis':

        token 'oste'  ->  ('ost','eo') cut at 1  ->  residue 'o'   [osteoarthritis]
                      ->  ('oste',)   cut at 4   ->  residue ''    [osteoporosis]

    The previous version kept only the FIRST path found (`results.setdefault`,
    scan order) and discarded the rest. The consequence was not a mis-ranking
    but a silent hijack: the model emitted 'oste' — the canonical first token
    of 'osteoporosis' — the grammar routed it to 'osteoarthritis' because that
    tag came first in scan order, and every following token was then forced to
    spell out the wrong word. The model never got to choose.

    Returning every path lets the caller keep all compatible states alive and
    mask on their UNION, so the token stream itself disambiguates: it is the
    model that picks the branch, by writing, never the grammar by guessing.

    Regex terminals (pda.regex_terminals) are yielded whole at depth 0 and
    never crossed (spec L2 — see the regex-lookahead future-work doc).
    """
    results: dict[int, list] = {}

    def dfs(state, node, consumed):
        if state.residue:
            fragments = [state.residue]
            from_residue = True
        else:
            fragments = state.recursive_get_tokens(list(state.stack))
            from_residue = False

        for frag in fragments:
            if not from_residue and frag in state.regex_terminals:
                if not consumed:
                    # depth 0: regex class participates as whole tokens,
                    # replayed as a plain terminal consumption
                    path = ((frag,), len(frag))
                    for token_id in state.map_terminals_tokens.get(frag, []):
                        bucket = results.setdefault(token_id, [])
                        if path not in bucket:
                            bucket.append(path)
                continue

            n = node
            alive = True
            for i, ch in enumerate(frag):
                n = n.child(ch)
                if n is None:
                    alive = False
                    break
                if n.token_id is not None:
                    path = (tuple(consumed) + (frag,), i + 1)
                    bucket = results.setdefault(n.token_id, [])
                    if path not in bucket:
                        bucket.append(path)
            if alive:
                child = state.clone()
                if from_residue:
                    child.residue = ""
                else:
                    child.next_state_terminal(frag)
                dfs(child, n, consumed + [frag])

    dfs(pda, trie, [])
    return results
