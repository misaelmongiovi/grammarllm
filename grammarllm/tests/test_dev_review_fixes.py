"""
test_dev_review_fixes.py
========================
Regression tests for the dev-branch review fixes:

1. PushdownAutomaton.recursive_get_tokens — iterative FIRST-of-stack scan
   (nullable-NT chains resolved exactly, no visited-set heuristics).
2. PushdownAutomaton.reset() — recomputes current_terminals so reused
   base PDAs never hand out clones with a stale empty terminal cache.
3. StatelessLogitsProcessor._advance_token — replay stops cleanly on a
   forced (dead-end) EOS instead of raising, while a genuine grammar EOS
   terminal is consumed normally and true grammar violations still raise.
4. Score history is gated behind track_score_history.

Real dependencies throughout: real transformers tokenizer (session fixture
in conftest.py) supplying the real special-token IDs. The grammar tables
are synthetic test data.
"""

import pytest
from conftest import vocab

from automaton import PushdownAutomaton
from logits_processor import StatelessLogitsProcessor, PdaSet


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: grammar  S* → A 'x' B A ;  A → 'a' | ε ;  B → 'b' | ε
# expressed directly as an LL(1) parsing table.
# ─────────────────────────────────────────────────────────────────────────────

A_ID, B_ID, X_ID, HELLO_ID = 100, 101, 102, 103

NULLABLE_TABLE = {
    'S*': {'a': ['A', 'x', 'B', 'A'], 'x': ['A', 'x', 'B', 'A']},
    'A':  {'a': ['a'], 'x': []},          # ε under FOLLOW(A) ∋ 'x'
    'B':  {'b': ['b'], 'a': []},          # ε under FOLLOW(B) ∋ 'a'
}
NULLABLE_MAP = {'a': [A_ID], 'b': [B_ID], 'x': [X_ID]}


def make_nullable_pda():
    return PushdownAutomaton(grammar=NULLABLE_TABLE, startSymbol='S*', map=NULLABLE_MAP)


# ─────────────────────────────────────────────────────────────────────────────
# 1. FIRST-of-stack scan
# ─────────────────────────────────────────────────────────────────────────────

class TestFirstOfStackScan:
    def test_initial_state(self):
        pda = make_nullable_pda()
        assert set(pda.get_tokens()) == {A_ID, X_ID}

    def test_nullable_chain_is_scanned_through(self):
        # After consuming 'x' with A→ε, stack is [A, B] (B on top).
        # Valid continuations: 'b' (B→b) or 'a' (B→ε then A→a).
        pda = make_nullable_pda()
        pda.next_state(X_ID)
        assert pda.stack == ['A', 'B']
        assert set(pda.get_tokens()) == {B_ID, A_ID}

    def test_full_sentence_reaches_eos(self):
        # "x a" : A→ε, x, B→ε, A→a  → empty stack.
        pda = make_nullable_pda()
        pda.next_state(X_ID)
        pda.next_state(A_ID)
        assert pda.eos()

    def test_scan_stops_at_non_nullable(self):
        # "a" → stack [A, B, x]; top 'x' is a terminal → only X_ID valid.
        pda = make_nullable_pda()
        pda.next_state(A_ID)
        assert pda.stack == ['A', 'B', 'x']
        assert set(pda.get_tokens()) == {X_ID}

    def test_no_duplicate_terminals(self):
        # Same terminal reachable from multiple stack positions must not
        # trigger the disjointness ValueError in get_tokens().
        table = {
            'S*': {'a': ['A', 'x', 'A']},
            'A':  {'a': ['a'], 'x': []},
        }
        pda = PushdownAutomaton(grammar=table, startSymbol='S*',
                                map={'a': [A_ID], 'x': [X_ID]})
        pda.next_state(A_ID)          # stack [A, x]
        assert set(pda.get_tokens()) == {X_ID}

    def test_deterministic_and_linear_terminals(self):
        pda = make_nullable_pda()
        terminals = pda.recursive_get_tokens(list(pda.stack))
        assert terminals == pda.recursive_get_tokens(list(pda.stack))
        assert len(terminals) == len(set(terminals))


# ─────────────────────────────────────────────────────────────────────────────
# 2. reset() recomputes current_terminals
# ─────────────────────────────────────────────────────────────────────────────

class TestResetRecomputesTerminals:
    def test_reset_restores_initial_terminals(self):
        pda = make_nullable_pda()
        initial_terminals = list(pda.current_terminals)
        pda.next_state(X_ID)
        pda.reset()
        assert pda.stack == ['S*']
        assert set(pda.current_terminals) == set(initial_terminals)
        assert pda.current_terminals != []

    def test_clone_after_reset_can_advance_immediately(self):
        # This was the failure mode: base PDA reset by the streamer,
        # then cloned by the processor and advanced without get_tokens().
        pda = make_nullable_pda()
        pda.next_state(X_ID)
        pda.reset()
        clone = pda.clone()
        clone.next_state(X_ID)   # must not raise "0 matching terminals"
        assert clone.stack == ['A', 'B']


# ─────────────────────────────────────────────────────────────────────────────
# 3. _advance_token replay semantics (real tokenizer special-token IDs)
# ─────────────────────────────────────────────────────────────────────────────

from lookahead import REGEX_TERMINALS_KEY

# A terminal containing U+FFFF is unspellable: no vocabulary token contains
# that character, so BOTH engines see a dead end (the lookahead engine would
# otherwise spell an ordinary string like 'x' through the trie even when its
# token map is empty).
UNSPELLABLE = "\uffff"


def make_dead_end_processor(tokenizer, engine=False):
    """Grammar whose DEAD terminal is unspellable and maps to no vocab tokens
    → dead end after 'hello'. EOS terminal maps to the REAL tokenizer EOS id;
    'hello' maps to its REAL vocab id so both engines agree on the mask."""
    eos_token = tokenizer.eos_token          # e.g. '<|im_end|>'
    hello_id = vocab(tokenizer)["hello"]
    grammar = {
        'S*':   {'hello': ['hello', 'DEAD'], eos_token: [eos_token]},
        'DEAD': {UNSPELLABLE: [UNSPELLABLE]},
    }
    map_tt = {'hello': [hello_id], UNSPELLABLE: [],
              eos_token: [tokenizer.eos_token_id],
              REGEX_TERMINALS_KEY: []}
    base = PushdownAutomaton(grammar=grammar, startSymbol='S*', map=map_tt)
    base.lookahead = engine
    return StatelessLogitsProcessor(tokenizer=tokenizer, base_pdas=[base]), base


class TestAdvanceToken:
    def test_forced_eos_on_dead_end_stops_without_raising(self, tokenizer, engine):
        proc, base = make_dead_end_processor(tokenizer, engine)
        pda = PdaSet.from_pda(base.clone())
        hello_id = vocab(tokenizer)["hello"]
        assert proc._advance_token(pda, hello_id) is True
        assert proc._valid_ids(pda)[0] == []             # dead end (both engines)
        # The processor force-emitted EOS here; replay must stop, not crash.
        assert proc._advance_token(pda, tokenizer.eos_token_id) is False
        assert pda.representative().stack == ['DEAD']

    def test_genuine_grammar_eos_is_consumed(self, tokenizer, engine):
        proc, base = make_dead_end_processor(tokenizer, engine)
        pda = PdaSet.from_pda(base.clone())
        assert proc._advance_token(pda, tokenizer.eos_token_id) is True
        assert pda.eos()

    def test_replay_stops_after_grammar_complete(self, tokenizer, engine):
        proc, base = make_dead_end_processor(tokenizer, engine)
        pda = PdaSet.from_pda(base.clone())
        proc._advance_token(pda, tokenizer.eos_token_id)
        hello_id = vocab(tokenizer)["hello"]
        assert proc._advance_token(pda, hello_id) is False   # nothing after completion

    def test_special_tokens_are_skipped(self, tokenizer, engine):
        proc, base = make_dead_end_processor(tokenizer, engine)
        pda = PdaSet.from_pda(base.clone())
        if tokenizer.pad_token_id is not None:
            assert proc._advance_token(pda, tokenizer.pad_token_id) is True
        if tokenizer.bos_token_id is not None:
            assert proc._advance_token(pda, tokenizer.bos_token_id) is True
        assert pda.representative().stack == ['S*']

    def test_truly_invalid_token_still_raises(self, tokenizer, engine):
        proc, base = make_dead_end_processor(tokenizer, engine)
        pda = PdaSet.from_pda(base.clone())
        with pytest.raises(ValueError):
            proc._advance_token(pda, 999)

    def test_get_pda_for_sequence_survives_forced_eos(self, tokenizer, engine):
        proc, _ = make_dead_end_processor(tokenizer, engine)
        hello_id = vocab(tokenizer)["hello"]
        pda = proc.get_pda_for_sequence([hello_id, tokenizer.eos_token_id],
                                        prompt_idx=0)
        assert pda.stack == ['DEAD']                    # stopped at forced EOS


# ─────────────────────────────────────────────────────────────────────────────
# 4. Score history gating
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreHistoryGating:
    def test_default_does_not_track(self, tokenizer, engine):
        proc, _ = make_dead_end_processor(tokenizer, engine)
        assert proc.track_score_history is False

    def test_flag_is_stored(self, tokenizer, engine):
        _, base = make_dead_end_processor(tokenizer, engine)
        proc = StatelessLogitsProcessor(
            tokenizer=tokenizer, base_pdas=[base], track_score_history=True
        )
        assert proc.track_score_history is True
