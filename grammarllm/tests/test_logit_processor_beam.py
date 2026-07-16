"""
test_logit_processor_beam.py
============================
Tests for the beam-search cache behavior of StatelessLogitsProcessor,
parametrized over BOTH masking engines (legacy boundary-strict and
token-boundary lookahead) via the `engine` fixture in conftest.py.

Real dependencies throughout: real torch tensors, real transformers
tokenizer (session fixture), real PushdownAutomaton. Synthetic grammars
map each terminal to its REAL vocabulary id (the lookahead engine derives
ids by spelling terminals through the vocab trie, so fake ids would
diverge between engines).
"""

import pytest
import torch

from automaton import PushdownAutomaton
from logits_processor import StatelessLogitsProcessor, PdaSet, _MAX_CACHE_SIZE
from lookahead import REGEX_TERMINALS_KEY
from conftest import scores_for, vocab


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def vocab_id(tokenizer, s):
    v = vocab(tokenizer)
    assert s in v, f"test requires vocab token {s!r}"
    return v[s]


def make_pda(tokenizer, engine=False, grammar=None, terminals=("a", "b")):
    if grammar is None:
        grammar = {'S*': {t: [t] for t in terminals}}
    map_tt = {t: [vocab_id(tokenizer, t)] for t in terminals}
    map_tt[REGEX_TERMINALS_KEY] = []
    pda = PushdownAutomaton(grammar=grammar, startSymbol='S*', map=map_tt)
    pda.lookahead = engine
    return pda


def make_proc(tokenizer, engine=False, prompt_len=1, **pda_kw):
    return StatelessLogitsProcessor(
        tokenizer=tokenizer,
        base_pdas=[make_pda(tokenizer, engine, **pda_kw)],
        sequences_per_prompt=1,
        prompt_len=prompt_len,
    )


def seed_cache(proc, tokenizer, prefix_tokens, prompt_idx=0, engine=False):
    """Pre-populate cache with a valid PDA state for prefix_tokens."""
    pda = make_pda(tokenizer, engine)
    for t in prefix_tokens:
        pda.next_state(t)
    proc.pda_cache[(prompt_idx, tuple(prefix_tokens))] = pda
    return pda


# ─────────────────────────────────────────────────────────────────────────────
# Cache hit must return a clone, not the shared object
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheHitClone:

    def test_cache_hit_pda_is_clone_not_same_object(self, tokenizer, engine):
        proc = make_proc(tokenizer, engine)
        seed_cache(proc, tokenizer, prefix_tokens=[], engine=engine)
        key = (0, ())

        _cached = proc.pda_cache.pop(key)
        proc.pda_cache[key] = _cached   # LRU reinsert
        returned = _cached.clone()

        assert returned is not _cached, \
            "Cache hit must return a clone, not the stored object"

    def test_two_clones_from_same_key_are_independent(self, tokenizer, engine):
        proc = make_proc(tokenizer, engine)
        seed_cache(proc, tokenizer, prefix_tokens=[], engine=engine)
        stored = proc.pda_cache[(0, ())]

        clone_a = stored.clone()
        clone_b = stored.clone()
        clone_a.current_terminals = ["CORRUPTED"]

        assert clone_b.current_terminals != ["CORRUPTED"], \
            "Clones must be fully independent"

    def test_lru_key_moves_to_end_on_access(self, tokenizer, engine):
        proc = make_proc(tokenizer, engine)
        for i in range(3):
            proc.pda_cache[(0, (i,))] = make_pda(tokenizer, engine)

        first_key = list(proc.pda_cache.keys())[0]
        _cached = proc.pda_cache.pop(first_key)
        proc.pda_cache[first_key] = _cached

        keys = list(proc.pda_cache.keys())
        assert keys[-1] == first_key
        assert keys[0] != first_key


# ─────────────────────────────────────────────────────────────────────────────
# get_pda_for_sequence must not silently bypass invalid tokens
# ─────────────────────────────────────────────────────────────────────────────

class TestGetPdaForSequence:

    def test_valid_sequence_returns_correct_state(self, tokenizer, engine):
        proc = make_proc(tokenizer, engine)
        pda = proc.get_pda_for_sequence([vocab_id(tokenizer, "a")], prompt_idx=0)
        assert pda.eos(), "After consuming 'a', stack must be empty"

    def test_invalid_token_raises(self, tokenizer, engine):
        proc = make_proc(tokenizer, engine)
        bad = vocab_id(tokenizer, "Z")     # real token, not derivable here
        with pytest.raises(ValueError):
            proc.get_pda_for_sequence([bad], prompt_idx=0)

    def test_invalid_second_token_raises(self, tokenizer, engine):
        """No silent break on the second token."""
        grammar = {'S*': {'x': ['x', 'y']}}
        proc = make_proc(tokenizer, engine, grammar=grammar, terminals=("x", "y"))
        bad = vocab_id(tokenizer, "Z")
        with pytest.raises(ValueError):
            proc.get_pda_for_sequence([vocab_id(tokenizer, "x"), bad])


# ─────────────────────────────────────────────────────────────────────────────
# LRU eviction preserves recently-used (short-prefix) ancestors
# ─────────────────────────────────────────────────────────────────────────────

class TestLRUEviction:

    def test_eviction_removes_from_front(self, tokenizer, engine):
        proc = make_proc(tokenizer, engine)
        for i in range(_MAX_CACHE_SIZE):
            proc.pda_cache[(0, (i,))] = make_pda(tokenizer, engine)

        oldest_key = (0, (0,))
        newest_key = (0, (_MAX_CACHE_SIZE - 1,))

        evict_count = _MAX_CACHE_SIZE // 4
        for lru_key in list(proc.pda_cache.keys())[:evict_count]:
            del proc.pda_cache[lru_key]

        assert oldest_key not in proc.pda_cache
        assert newest_key in proc.pda_cache

    def test_recently_accessed_old_entry_survives_eviction(self, tokenizer, engine):
        proc = make_proc(tokenizer, engine)
        for i in range(_MAX_CACHE_SIZE):
            proc.pda_cache[(0, (i,))] = make_pda(tokenizer, engine)

        oldest_key = (0, (0,))
        _cached = proc.pda_cache.pop(oldest_key)
        proc.pda_cache[oldest_key] = _cached

        evict_count = _MAX_CACHE_SIZE // 4
        for lru_key in list(proc.pda_cache.keys())[:evict_count]:
            del proc.pda_cache[lru_key]

        assert oldest_key in proc.pda_cache


# ─────────────────────────────────────────────────────────────────────────────
# reset() clears score histories, PDA cache, mask cache
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreHistoryReset:

    def test_reset_clears_both_histories(self, tokenizer, engine):
        proc = make_proc(tokenizer, engine)
        proc.original_scores_history = [torch.zeros(2)]
        proc.filtered_scores_history = [torch.zeros(2)]
        proc.reset()
        assert proc.original_scores_history == []
        assert proc.filtered_scores_history == []

    def test_reset_clears_pda_and_mask_caches(self, tokenizer, engine):
        proc = make_proc(tokenizer, engine)
        seed_cache(proc, tokenizer, [], engine=engine)
        if engine:
            proc._valid_ids(PdaSet.from_pda(make_pda(tokenizer, engine)))   # populate mask cache
            assert len(proc.mask_cache) > 0
        assert len(proc.pda_cache) > 0
        proc.reset()
        assert proc.pda_cache == {}
        assert proc.mask_cache == {}

    def test_reset_resets_log_counter(self, tokenizer, engine):
        proc = make_proc(tokenizer, engine)
        proc.log_counter = 42
        proc.reset()
        assert proc.log_counter == 0


# ─────────────────────────────────────────────────────────────────────────────
# Real __call__ with real tensors: masking, forced EOS, history gating
# ─────────────────────────────────────────────────────────────────────────────

class TestCallRealTensors:

    def test_call_masks_all_but_grammar_tokens(self, tokenizer, engine):
        proc = make_proc(tokenizer, engine)
        out = proc(torch.tensor([[5]]), scores_for(tokenizer))
        finite = set(torch.isfinite(out[0]).nonzero().flatten().tolist())
        base = {vocab_id(tokenizer, 'a'), vocab_id(tokenizer, 'b')}
        if engine:
            assert base <= finite, "lookahead mask must be a superset of legacy"
        else:
            assert finite == base, f"legacy mask must be exactly {base}, got {finite}"

    def test_call_forces_eos_when_grammar_complete(self, tokenizer, engine):
        proc = make_proc(tokenizer, engine)
        a = vocab_id(tokenizer, 'a')
        out = proc(torch.tensor([[5, a]]), scores_for(tokenizer))
        finite = torch.isfinite(out[0]).nonzero().flatten().tolist()
        assert finite == [tokenizer.eos_token_id], \
            f"Only EOS must survive after grammar completion, got {finite}"

    def test_call_populates_cache_and_case_a_advance(self, tokenizer, engine):
        proc = make_proc(tokenizer, engine)
        a = vocab_id(tokenizer, 'a')
        proc(torch.tensor([[5]]), scores_for(tokenizer))
        assert (0, ()) in proc.pda_cache
        proc(torch.tensor([[5, a]]), scores_for(tokenizer))
        assert (0, (a,)) in proc.pda_cache

    def test_history_not_tracked_by_default(self, tokenizer, engine):
        proc = make_proc(tokenizer, engine)
        proc(torch.tensor([[5]]), scores_for(tokenizer))
        assert proc.original_scores_history == []
        assert proc.filtered_scores_history == []

    def test_history_tracked_when_enabled(self, tokenizer, engine):
        proc = StatelessLogitsProcessor(
            tokenizer=tokenizer,
            base_pdas=[make_pda(tokenizer, engine)],
            sequences_per_prompt=1,
            prompt_len=1,
            track_score_history=True,
        )
        proc(torch.tensor([[5]]), scores_for(tokenizer))
        assert len(proc.original_scores_history) == 1
        assert len(proc.filtered_scores_history) == 1


# ─────────────────────────────────────────────────────────────────────────────
# HF fills its beam candidate pool with grammar-masked (-inf) tokens
# ─────────────────────────────────────────────────────────────────────────────

class TestRetiredBeams:
    """
    Regression: `_beam_search` always returns `beams_to_keep` candidates, even
    when the mask leaves fewer valid continuations than that. The surplus slots
    are filled with tokens this processor set to -inf, and when fewer than
    `num_beams` candidates carry a finite score those -inf beams are promoted to
    running beams. The next step then replays a history holding a token the
    grammar never allowed.

    Such a beam scores -inf: it can never win nor be returned, and its spurious
    token is already committed to running_sequences. It must be retired, not
    crash the generation.
    """

    def test_call_retires_beam_instead_of_raising(self, tokenizer, engine):
        proc = make_proc(tokenizer, engine)
        bad = vocab_id(tokenizer, "Z")          # real token, never derivable here

        out = proc(torch.tensor([[5, bad]]), scores_for(tokenizer))

        assert proc.retired_beams == 1
        # retired -> only EOS survives the mask
        alive = (out[0] > -float("inf")).nonzero().flatten().tolist()
        assert alive == [tokenizer.eos_token_id]

    def test_retired_beam_stays_dead_for_its_children(self, tokenizer, engine):
        """A retired beam keeps getting extended by HF; it must not re-raise."""
        proc = make_proc(tokenizer, engine)
        bad = vocab_id(tokenizer, "Z")

        proc(torch.tensor([[5, bad]]), scores_for(tokenizer))
        # HF extends the dead beam again — replay must short-circuit, not raise
        out = proc(torch.tensor([[5, bad, vocab_id(tokenizer, "a")]]), scores_for(tokenizer))

        alive = (out[0] > -float("inf")).nonzero().flatten().tolist()
        assert alive == [tokenizer.eos_token_id]

    def test_valid_beam_unaffected_by_sibling_retirement(self, tokenizer, engine):
        """Retiring beam 1 must not relax the grammar on the healthy beam 0."""
        # 'x y': after 'x' the healthy beam is mid-parse and must still be
        # pinned to 'y' — a satisfied grammar would only prove EOS-forcing.
        proc = StatelessLogitsProcessor(
            tokenizer=tokenizer,
            base_pdas=[make_pda(tokenizer, engine,
                                grammar={'S*': {'x': ['x', 'y']}},
                                terminals=("x", "y"))],
            sequences_per_prompt=2,
            prompt_len=1,
        )
        bad = vocab_id(tokenizer, "Z")
        ids = torch.tensor([[5, vocab_id(tokenizer, "x")],   # beam 0: legal, mid-parse
                            [5, bad]])                        # beam 1: HF filler
        out = proc(ids, scores_for(tokenizer, batch=2))

        assert proc.retired_beams == 1
        # beam 0 is still mid-parse: grammar pins it to 'y', nothing else
        assert (out[0] > -float("inf")).nonzero().flatten().tolist() == [vocab_id(tokenizer, "y")]
        # beam 1 was retired: EOS only
        assert (out[1] > -float("inf")).nonzero().flatten().tolist() == [tokenizer.eos_token_id]
