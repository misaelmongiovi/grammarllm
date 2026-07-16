"""
Tests for the token-boundary lookahead engine (g_t_r).
Spec: docs/superpowers/specs/2026-07-08-token-boundary-lookahead-design.md
Real tokenizer via the session fixture in conftest.py — no mocks.
"""
import pytest
from conftest import vocab as vocab_of

from lookahead import VocabTrie, get_vocab_trie, REGEX_TERMINALS_KEY


class TestVocabTrie:
    def test_contains_known_tokens(self, tokenizer):
        trie = get_vocab_trie(tokenizer)
        vocab = vocab_of(tokenizer)
        for tok_str in ("a", "Ġhappy", "{\""):
            if tok_str not in vocab:
                continue
            node = trie
            for ch in tok_str:
                node = node.child(ch)
                assert node is not None, f"trie lost {tok_str!r} at {ch!r}"
            assert node.token_id == vocab[tok_str]

    def test_prefix_nodes_have_no_token_id_unless_token(self, tokenizer):
        trie = get_vocab_trie(tokenizer)
        vocab = vocab_of(tokenizer)
        # walk a long token; every strict prefix must carry either None
        # or the id of a shorter REAL token, never a wrong id
        long_tok = max(vocab, key=len)
        node = trie
        for i, ch in enumerate(long_tok):
            node = node.child(ch)
            prefix = long_tok[: i + 1]
            if node.token_id is not None:
                assert vocab.get(prefix) == node.token_id

    def test_cache_returns_same_instance(self, tokenizer):
        assert get_vocab_trie(tokenizer) is get_vocab_trie(tokenizer)

    def test_missing_child_is_none(self, tokenizer):
        trie = get_vocab_trie(tokenizer)
        assert trie.child("￿") is None


from automaton import PushdownAutomaton


def tid(tokenizer, tok_str):
    """Real vocab id of a token string; fails loudly if absent."""
    v = vocab_of(tokenizer)
    assert tok_str in v, f"test needs vocab token {tok_str!r}"
    return v[tok_str]


def brace_grammar(tokenizer):
    """S* → '{' ' ' 'ciao'  as three exact terminals (surface forms)."""
    grammar = {'S*': {'{': ['{', 'Ġ', 'ciao']}}
    map_tt = {
        '{': [tid(tokenizer, '{')],
        'Ġ': [tid(tokenizer, 'Ġ')],
        'ciao': [tid(tokenizer, 'ciao')] if 'ciao' in vocab_of(tokenizer) else [],
        REGEX_TERMINALS_KEY: [],
    }
    return PushdownAutomaton(grammar=grammar, startSymbol='S*', map=map_tt)


class TestResidueState:
    def test_defaults(self, tokenizer):
        pda = brace_grammar(tokenizer)
        assert pda.residue == ""
        assert pda.lookahead is False
        assert pda.regex_terminals == set()

    def test_regex_terminal_names_channel(self, tokenizer):
        grammar = {'S*': {'digit': ['digit']}}
        map_tt = {'digit': [tid(tokenizer, '4')], REGEX_TERMINALS_KEY: ['digit']}
        pda = PushdownAutomaton(grammar=grammar, startSymbol='S*', map=map_tt)
        assert pda.regex_terminals == {'digit'}
        # the sentinel key must NOT leak into token→terminal inversion
        assert all(not (isinstance(t, str) and t == REGEX_TERMINALS_KEY)
                   for terms in pda.map_tokens_terminals.values() for t in terms)

    def test_eos_requires_empty_residue(self, tokenizer):
        pda = brace_grammar(tokenizer)
        pda.stack = []
        pda.residue = "ao"
        assert not pda.eos()
        pda.residue = ""
        assert pda.eos()

    def test_clone_and_reset_carry_residue(self, tokenizer):
        pda = brace_grammar(tokenizer)
        pda.lookahead = True
        pda.residue = "ao"
        c = pda.clone()
        assert c.residue == "ao" and c.lookahead is True
        pda.reset()
        assert pda.residue == "" and pda.stack == ['S*']


class TestApplyLookaheadPath:
    def test_full_span_two_terminals(self, tokenizer):
        # token "{ " consumes '{' and 'Ġ' fully → stack ['ciao'], no residue
        pda = brace_grammar(tokenizer)
        pda.apply_lookahead_path(('{', 'Ġ'), 1)
        assert pda.stack == ['ciao']
        assert pda.residue == ""

    def test_mid_terminal_cut_sets_residue(self, tokenizer):
        # token "{ ci": '{' + 'Ġ' full, 'ciao' cut after 2 chars → residue 'ao'
        pda = brace_grammar(tokenizer)
        pda.apply_lookahead_path(('{', 'Ġ', 'ciao'), 2)
        assert pda.stack == []
        assert pda.residue == "ao"
        assert not pda.eos()

    def test_residue_consumed_next_step(self, tokenizer):
        pda = brace_grammar(tokenizer)
        pda.apply_lookahead_path(('{', 'Ġ', 'ciao'), 2)   # residue 'ao'
        pda.apply_lookahead_path(('ao',), 2)              # spell the rest
        assert pda.residue == "" and pda.eos()

    def test_partial_residue_consumption(self, tokenizer):
        pda = brace_grammar(tokenizer)
        pda.apply_lookahead_path(('{', 'Ġ', 'ciao'), 1)   # residue 'iao'
        pda.apply_lookahead_path(('iao',), 1)             # consume 'i' only
        assert pda.residue == "ao"


from lookahead import lookahead_paths


class TestLookaheadTokens:
    def test_old_subset_of_new(self, tokenizer):
        pda = brace_grammar(tokenizer)
        legacy = set(pda.get_tokens())
        mask = lookahead_paths(pda, get_vocab_trie(tokenizer))
        assert legacy <= set(mask), "legacy mask must be a subset of lookahead mask"

    def test_merged_span_token_present(self, tokenizer):
        # "{Ġ" is the surface form of "{ " — present in Qwen's vocab
        merged = vocab_of(tokenizer).get("{Ġ")
        if merged is None:
            pytest.skip("tokenizer has no '{ ' merged token")
        pda = brace_grammar(tokenizer)
        mask = lookahead_paths(pda, get_vocab_trie(tokenizer))
        assert merged in mask
        # lookahead_paths ritorna TUTTI i percorsi compatibili: quello atteso
        # dev'essere fra questi (qui la grammatica e' non ambigua -> uno solo).
        assert (('{', 'Ġ'), 1) in mask[merged]

    def test_mid_terminal_cut_path(self, tokenizer):
        # any vocab token that spells '{' + 'Ġ' + a strict prefix of 'ciao'
        vocab = vocab_of(tokenizer)
        candidates = {s: i for s, i in vocab.items()
                      if s.startswith("{Ġci") and len(s) < len("{Ġciao")}
        if not candidates:
            pytest.skip("no mid-cut token in this vocab")
        s, i = next(iter(candidates.items()))
        pda = brace_grammar(tokenizer)
        mask = lookahead_paths(pda, get_vocab_trie(tokenizer))
        assert i in mask
        assert (('{', 'Ġ', 'ciao'), len(s) - 2) in mask[i]   # chars inside 'ciao'

    def test_path_replay_reaches_consistent_state(self, tokenizer):
        pda = brace_grammar(tokenizer)
        mask = lookahead_paths(pda, get_vocab_trie(tokenizer))
        inv = {i: s for s, i in vocab_of(tokenizer).items()}
        for token_id, paths in list(mask.items())[:20]:
            # OGNI percorso compatibile deve compitare esattamente la stringa
            # del token: nessun ramo tenuto vivo puo' produrre altro testo.
            for frags, chars in paths:
                clone = pda.clone()
                clone.apply_lookahead_path(frags, chars)
                spelled = "".join(frags[:-1]) + frags[-1][:chars]
                assert inv[token_id] == spelled

    def test_regex_terminal_depth0_only(self, tokenizer):
        # S* → '"' digit : digit tokens appear whole; nothing crosses '"'→digit
        grammar = {'S*': {'"': ['"', 'digit']}, }
        four = tid(tokenizer, '4')
        map_tt = {'"': [tid(tokenizer, '"')], 'digit': [four],
                  REGEX_TERMINALS_KEY: ['digit']}
        pda = PushdownAutomaton(grammar=grammar, startSymbol='S*', map=map_tt)
        mask = lookahead_paths(pda, get_vocab_trie(tokenizer))
        # the quote itself is offered
        assert tid(tokenizer, '"') in mask
        # merged '"4' must NOT be offered (would cross into the regex class)
        merged = vocab_of(tokenizer).get('"4')
        if merged is not None:
            assert merged not in mask
        # and at the NEXT state, the regex token is offered whole
        pda.next_state_terminal('"'); pda.get_tokens()
        mask2 = lookahead_paths(pda, get_vocab_trie(tokenizer))
        assert four in mask2
        assert (('digit',), len('digit')) in mask2[four]

    def test_dfs_from_residue(self, tokenizer):
        pda = brace_grammar(tokenizer)
        pda.apply_lookahead_path(('{', 'Ġ', 'ciao'), 2)   # residue 'ao'
        mask = lookahead_paths(pda, get_vocab_trie(tokenizer))
        ao = vocab_of(tokenizer).get("ao")
        if ao is None:
            pytest.skip("no 'ao' token")
        assert ao in mask and (('ao',), 2) in mask[ao]


import torch
from logits_processor import StatelessLogitsProcessor, PdaSet


def make_processor(tokenizer, pda, prompt_len=1):
    return StatelessLogitsProcessor(
        tokenizer=tokenizer, base_pdas=[pda],
        sequences_per_prompt=1, prompt_len=prompt_len,
    )


def scores_for(tokenizer):
    return torch.zeros((1, max(vocab_of(tokenizer).values()) + 1))


class TestProcessorRouting:
    def test_legacy_flag_off_matches_get_tokens(self, tokenizer):
        pda = brace_grammar(tokenizer)          # lookahead defaults False
        proc = make_processor(tokenizer, pda)
        ids, paths = proc._valid_ids(PdaSet.from_pda(pda.clone()))
        assert set(ids) == set(pda.get_tokens())
        assert paths is None

    def test_lookahead_flag_widens_mask(self, tokenizer):
        pda = brace_grammar(tokenizer)
        pda.lookahead = True
        proc = make_processor(tokenizer, pda)
        ids, paths = proc._valid_ids(PdaSet.from_pda(pda.clone()))
        assert set(pda.get_tokens()) <= set(ids)
        assert paths is not None

    def test_mask_cache_hit(self, tokenizer):
        pda = brace_grammar(tokenizer)
        pda.lookahead = True
        proc = make_processor(tokenizer, pda)
        proc._valid_ids(PdaSet.from_pda(pda.clone()))
        digest = frozenset({(tuple(pda.stack), pda.residue)})
        assert digest in proc.mask_cache
        proc.reset()
        assert proc.mask_cache == {}

    def test_advance_merged_token_via_call(self, tokenizer):
        merged = vocab_of(tokenizer).get("{Ġ")
        if merged is None:
            pytest.skip("no '{ ' merged token")
        pda = brace_grammar(tokenizer)
        pda.lookahead = True
        proc = make_processor(tokenizer, pda)
        out = proc(torch.tensor([[5]]), scores_for(tokenizer))
        assert torch.isfinite(out[0, merged]).item()
        # feed the merged token back: re-simulation must accept it
        out2 = proc(torch.tensor([[5, merged]]), scores_for(tokenizer))
        cached = proc.pda_cache[(0, (merged,))]
        assert cached.stack == ['ciao'] and cached.residue == ""

    def test_invalid_token_still_raises_in_lookahead(self, tokenizer):
        pda = brace_grammar(tokenizer)
        pda.lookahead = True
        proc = make_processor(tokenizer, pda)
        with pytest.raises(ValueError):
            proc.get_pda_for_sequence([999999999 % len(vocab_of(tokenizer))])


class TestDefaultOn:
    def test_generate_grammar_parameters_default_lookahead(self, tokenizer):
        from grammarllm.generate_with_constraints import generate_grammar_parameters
        grammar = {'S*': {'a': ['a']}}
        map_tt = {'a': [tid(tokenizer, 'a')], REGEX_TERMINALS_KEY: []}
        pdas, _ = generate_grammar_parameters(tokenizer, grammar, map_tt)
        assert all(p.lookahead for p in pdas)
        pdas_off, _ = generate_grammar_parameters(
            tokenizer, grammar, map_tt, token_lookahead=False)
        assert all(not p.lookahead for p in pdas_off)


# ─────────────────────────────────────────────────────────────────────────────
# Task 6: canonical-tokenization acceptance + walks + perf budget
# ─────────────────────────────────────────────────────────────────────────────

import json
import time
from typing import Literal, Optional
from pydantic import BaseModel
from grammarllm.utils.pydantic_to_grammar import pydantic_to_productions
from grammarllm.generate_with_constraints import (
    get_parsing_table_and_map_tt, generate_grammar_parameters,
)


class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]


def build(tokenizer, model, token_lookahead=True):
    prods, rgx = pydantic_to_productions(model)
    pt, mtt = get_parsing_table_and_map_tt(tokenizer, prods, rgx)
    pdas, _ = generate_grammar_parameters(tokenizer, pt, mtt,
                                          token_lookahead=token_lookahead)
    proc = StatelessLogitsProcessor(tokenizer=tokenizer, base_pdas=pdas)
    return proc, pdas[0]


CANONICAL = '{"label": "positive"}'


@pytest.mark.integration
class TestCanonicalTokenization:
    def test_lookahead_accepts_canonical_encoding(self, tokenizer):
        """HEADLINE: the model's natural tokenization must replay through
        the masks token-by-token. The legacy engine fails this by design."""
        proc, base = build(tokenizer, Sentiment, token_lookahead=True)
        pda = PdaSet.from_pda(base.clone())
        for tid_ in tokenizer.encode(CANONICAL, add_special_tokens=False):
            valid, paths = proc._valid_ids(pda)
            assert tid_ in valid, (
                f"canonical token {tid_} ({tokenizer.decode([tid_])!r}) rejected; "
                f"states={list(pda.states)}"
            )
            proc._advance(pda, tid_, paths)
        assert pda.eos()

    def test_legacy_rejects_canonical_encoding(self, tokenizer):
        """A/B counterpart: proves the boundary limitation exists in the
        legacy engine (if this ever passes, the A/B story changes)."""
        proc, base = build(tokenizer, Sentiment, token_lookahead=False)
        pda = PdaSet.from_pda(base.clone())
        rejected = False
        for tid_ in tokenizer.encode(CANONICAL, add_special_tokens=False):
            valid, paths = proc._valid_ids(pda)
            if tid_ not in valid:
                rejected = True
                break
            proc._advance(pda, tid_, paths)
        assert rejected, "legacy engine unexpectedly accepted the canonical encoding"


@pytest.mark.integration
class TestLookaheadWalks:
    @pytest.mark.parametrize("seed", range(5))
    def test_walk_round_trip_under_lookahead(self, tokenizer, seed):
        import random
        proc, base = build(tokenizer, Sentiment, token_lookahead=True)
        rng = random.Random(seed)
        pda = PdaSet.from_pda(base.clone())
        ids = []
        for _ in range(200):
            if pda.eos():
                break
            valid, paths = proc._valid_ids(pda)
            choices = [t for t in valid if t != tokenizer.eos_token_id] or valid
            t = rng.choice(choices)
            proc._advance(pda, t, paths)
            ids.append(t)
        else:
            raise AssertionError("walk did not terminate")
        data = json.loads(tokenizer.decode(ids))
        Sentiment.model_validate(data)


@pytest.mark.integration
class TestPerfBudget:
    def test_cold_and_hot_mask_cost(self, tokenizer):
        proc, base = build(tokenizer, Sentiment, token_lookahead=True)
        pda = PdaSet.from_pda(base.clone())
        t0 = time.perf_counter()
        proc._valid_ids(pda)
        cold = time.perf_counter() - t0
        t0 = time.perf_counter()
        proc._valid_ids(pda)
        hot = time.perf_counter() - t0
        # spec target ~5ms typical; 20ms ceiling absorbs CI variance
        assert cold < 0.020, f"cold mask {cold*1e3:.1f}ms over budget"
        assert hot < 0.001, f"cache hit {hot*1e3:.2f}ms not O(1)"


@pytest.mark.e2e
def test_preserved_mass_improves_with_lookahead(tokenizer):
    transformers = pytest.importorskip("transformers")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct")
    from grammarllm.generate_with_constraints import generate_text
    from grammarllm.utils.generation_analysis import compute_generation_analysis

    prods, rgx = pydantic_to_productions(Sentiment)
    pt, mtt = get_parsing_table_and_map_tt(tokenizer, prods, rgx)
    prompt = 'Classify: "I love sunny days." Reply as JSON:'

    results = {}
    for flag in (False, True):
        pdas, streamer = generate_grammar_parameters(
            tokenizer, pt, mtt, token_lookahead=flag)
        r = generate_text(model, tokenizer, prompt, pdas, streamer,
                          max_new_tokens=20, output_scores=True)
        results[flag] = compute_generation_analysis(
            r, tokenizer, label="lookahead" if flag else "legacy")

    legacy, lookahead = results[False], results[True]
    print(f"\npreserved mass: legacy={legacy.mean_preserved_mass:.4f} "
          f"lookahead={lookahead.mean_preserved_mass:.4f} "
          f"delta={lookahead.mean_preserved_mass - legacy.mean_preserved_mass:+.4f}")
    # generated paths differ, so per-step dominance is not guaranteed;
    # the aggregate must not regress (small tolerance for path effects)
    assert lookahead.mean_preserved_mass >= legacy.mean_preserved_mass - 0.02


class TestAmbiguousTokenKeepsAllBranches:
    """
    Regression: a token compatible with SEVERAL grammar paths must keep them ALL
    alive. The grammar never picks a branch — the model does, by writing.

    The old engine kept only the first path found (`results.setdefault`, scan
    order) and killed the rest. With children 'osteoarthritis' / 'osteoporosis':

        token 'oste'  ->  ('ost','eo') cut 1  ->  residue 'o'   [osteoarthritis]
                      ->  ('oste',)   cut 4   ->  residue ''    [osteoporosis]

    'oste' is the canonical first token of 'osteoporosis'. Scan order routed it
    to 'osteoarthritis' anyway, and every later token was then forced to spell
    out the wrong word: the model emitted the right token and got the wrong
    answer, with no way back.
    """

    def two_children(self, tokenizer):
        # scan order puts 'osteoarthritis' FIRST — the branch that used to win
        tags = ['osteoarthritis', 'osteoporosis']
        grammar = {'S*': {tokenizer.tokenize(t)[0]: list(tokenizer.tokenize(t))
                          for t in tags}}
        map_tt = {tk: [vocab_of(tokenizer)[tk]]
                  for t in tags for tk in tokenizer.tokenize(t)}
        map_tt[REGEX_TERMINALS_KEY] = []
        pda = PushdownAutomaton(grammar=grammar, startSymbol='S*', map=map_tt)
        pda.lookahead = True
        return pda

    def test_ambiguous_token_yields_several_paths(self, tokenizer):
        pda = self.two_children(tokenizer)
        oste = vocab_of(tokenizer).get('oste')
        if oste is None:
            pytest.skip("tokenizer has no 'oste' token")
        paths = lookahead_paths(pda, get_vocab_trie(tokenizer))
        assert len(paths[oste]) >= 2, "both branches must be reported, not just the first"

    def test_both_branches_stay_alive_and_mask_is_their_union(self, tokenizer):
        pda = self.two_children(tokenizer)
        oste = vocab_of(tokenizer).get('oste')
        if oste is None:
            pytest.skip("tokenizer has no 'oste' token")
        proc = make_processor(tokenizer, pda)

        S = PdaSet.from_pda(pda.clone())
        proc._advance(S, oste)
        assert len(S) >= 2, "the ambiguous token must not collapse the state set"

        # The mask must be exactly the UNION of what each live state admits —
        # nothing dropped because another branch didn't offer it.
        ids, _ = proc._valid_ids(S)
        union = set()
        for key, single in S.states.items():
            sub, _ = proc._valid_ids(PdaSet.from_pda(single.clone()))
            union |= set(sub)
        assert set(ids) == union, "the mask is not the union of the live states"

        # Concretely: 'oste' left one branch mid-terminal (residue 'o' -> continues
        # 'osteo|arthritis') and the other on a clean boundary (-> 'opor'/'osis').
        # BOTH continuations must be offered; the old engine offered only the first.
        offered = {tokenizer.convert_ids_to_tokens([i])[0] for i in ids}
        assert any(t.startswith('op') for t in offered), "the osteoporosis branch was killed"
        assert any(t.startswith('o') and not t.startswith('op') for t in offered), \
            "the osteoarthritis branch was killed"

    @pytest.mark.parametrize("word", ['osteoporosis', 'osteoarthritis'])
    def test_canonical_tokenization_of_each_child_is_reachable(self, tokenizer, word):
        """The model must be able to spell EITHER child, its own way."""
        pda = self.two_children(tokenizer)
        if vocab_of(tokenizer).get('oste') is None:
            pytest.skip("tokenizer has no 'oste' token")
        proc = make_processor(tokenizer, pda)

        S = PdaSet.from_pda(pda.clone())
        for tid_ in tokenizer.encode(word, add_special_tokens=False):
            valid, paths = proc._valid_ids(S)
            assert tid_ in valid, f"canonical token of {word} rejected"
            proc._advance(S, tid_, paths)
        assert S.eos()
