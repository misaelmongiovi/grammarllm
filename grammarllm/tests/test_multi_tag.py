"""
test_multi_tag.py
=================
Tests for multi-tag ordering in process_full_grammar.

The core invariant: given  S* → <<a>> A <<b>>,  the generated grammar must
enforce that 'a' is always generated BEFORE 'b', never the reverse.

Uses the REAL Qwen tokenizer (session fixture in conftest.py). Expected
terminal sequences are computed through the tokenizer itself, so the tests
hold regardless of how a tag happens to split into subword tokens.
"""

from itertools import permutations

import pytest

from grammar_generation import ProductionRuleProcessor


@pytest.fixture()
def processor(tokenizer):
    return ProductionRuleProcessor(tokenizer=tokenizer)


def toks(tokenizer, *tags):
    """Concatenated real tokenization of the given tag strings."""
    out = ()
    for tag in tags:
        out += tuple(tokenizer.tokenize(tag))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Helper: collect all terminal sequences reachable from S* in the grammar
# ─────────────────────────────────────────────────────────────────────────────

def reachable_terminal_sequences(grammar, start="S*", max_depth=20):
    """
    Enumerate all sequences of terminals reachable from `start`.
    Returns a set of tuples, each tuple being one possible terminal sequence.
    Only works for small finite grammars (no recursion).
    """
    plain = {}
    for key, prods in grammar.items():
        nt = key[0] if isinstance(key, tuple) else key
        plain.setdefault(nt, [])
        for prod in prods:
            plain[nt].append(prod if isinstance(prod, list) else [prod])

    results = set()

    def expand(symbols, depth):
        if depth > max_depth:
            return
        if not symbols:
            results.add(())
            return
        head, *tail = symbols
        if head in plain:
            for prod in plain[head]:
                expand(list(prod) + tail, depth + 1)
        else:
            for rest in _expand_all(tail, depth):
                results.add((head,) + rest)

    def _expand_all(symbols, depth):
        if not symbols:
            return {()}
        head, *tail = symbols
        out = set()
        if head in plain:
            for prod in plain[head]:
                for rest in _expand_all(list(prod) + tail, depth + 1):
                    out.add(rest)
        else:
            for rest in _expand_all(tail, depth + 1):
                out.add((head,) + rest)
        return out

    expand([start], 0)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSingleTagPerProduction:
    """Existing behaviour must not regress: single tag per production."""

    def test_single_tag_alternatives(self, processor, tokenizer):
        """S* → <<a>> | <<b>> | <<c>>  — three alternatives, each one tag."""
        grammar, _ = processor.process_full_grammar({
            'S*': ["<<a>>", "<<b>>", "<<c>>"]
        })
        seqs = reachable_terminal_sequences(grammar)
        assert toks(tokenizer, 'a') in seqs
        assert toks(tokenizer, 'b') in seqs
        assert toks(tokenizer, 'c') in seqs
        assert len(seqs) == 3

    def test_tag_then_nonterminal(self, processor, tokenizer):
        """S* → <<pos>> A | <<neg>> A  (tag then NT, two alternatives)."""
        grammar, _ = processor.process_full_grammar({
            'S*': ["<<pos>> A", "<<neg>> A"],
            'A':  ["<<happy>>", "<<sad>>"],
        })
        seqs = reachable_terminal_sequences(grammar)
        assert toks(tokenizer, 'pos', 'happy') in seqs
        assert toks(tokenizer, 'pos', 'sad') in seqs
        assert toks(tokenizer, 'neg', 'happy') in seqs
        assert toks(tokenizer, 'neg', 'sad') in seqs


class TestMultipleTagsInOneProduction:
    """Core fix: multiple <<>> tags in the same production must preserve order."""

    def test_two_tags_order_enforced(self, processor, tokenizer):
        """
        S* → <<a>> <<b>>
        Only the sequence a·b must be reachable, never b·a.
        """
        grammar, _ = processor.process_full_grammar({
            'S*': ["<<a>> <<b>>"]
        })
        seqs = reachable_terminal_sequences(grammar)
        assert toks(tokenizer, 'a', 'b') in seqs, f"Expected a·b in {seqs}"
        assert toks(tokenizer, 'b', 'a') not in seqs, f"b·a must NOT be reachable: {seqs}"

    def test_tag_nt_tag_order_enforced(self, processor, tokenizer):
        """
        S* → <<a>> A <<b>>   (tag, NT, tag)
        Both tags must appear and in order a…b.
        """
        grammar, _ = processor.process_full_grammar({
            'S*': ["<<a>> A <<b>>"],
            'A':  ["<<x>>"],
        })
        seqs = reachable_terminal_sequences(grammar)
        assert toks(tokenizer, 'a', 'x', 'b') in seqs, f"Expected a·x·b in {seqs}"
        assert toks(tokenizer, 'b', 'x', 'a') not in seqs

    def test_three_tags_order_enforced(self, processor, tokenizer):
        """S* → <<a>> <<b>> <<c>>  — three tags, strict order."""
        grammar, _ = processor.process_full_grammar({
            'S*': ["<<a>> <<b>> <<c>>"]
        })
        seqs = reachable_terminal_sequences(grammar)
        assert toks(tokenizer, 'a', 'b', 'c') in seqs
        for perm in permutations(['a', 'b', 'c']):
            if perm != ('a', 'b', 'c'):
                assert toks(tokenizer, *perm) not in seqs, \
                    f"Unexpected permutation {perm} in {seqs}"


class TestAlternativesWithMultipleTags:
    """Mix of alternatives where some productions have multiple tags."""

    def test_two_alternatives_each_with_two_tags(self, processor, tokenizer):
        """
        S* → <<a>> <<x>>  |  <<b>> <<y>>
        Reachable: a·x and b·y. Cross products a·y and b·x must NOT appear.
        """
        grammar, _ = processor.process_full_grammar({
            'S*': ["<<a>> <<x>>", "<<b>> <<y>>"]
        })
        seqs = reachable_terminal_sequences(grammar)
        assert toks(tokenizer, 'a', 'x') in seqs
        assert toks(tokenizer, 'b', 'y') in seqs
        assert toks(tokenizer, 'a', 'y') not in seqs, f"Cross-product a·y must not appear: {seqs}"
        assert toks(tokenizer, 'b', 'x') not in seqs, f"Cross-product b·x must not appear: {seqs}"

    def test_shared_second_tag_across_alternatives(self, processor, tokenizer):
        """
        S* → <<a>> A <<z>>  |  <<b>> A <<z>>
        Both a·x·z and b·x·z must be reachable, and nothing else.
        """
        grammar, _ = processor.process_full_grammar({
            'S*': ["<<a>> A <<z>>", "<<b>> A <<z>>"],
            'A':  ["<<x>>"],
        })
        seqs = reachable_terminal_sequences(grammar)
        expected = {toks(tokenizer, 'a', 'x', 'z'), toks(tokenizer, 'b', 'x', 'z')}
        assert expected <= seqs
        assert seqs == expected, f"Unexpected extra sequences: {seqs - expected}"


class TestRegressionNoTagProductions:
    """Productions without any tag must still work correctly."""

    def test_no_tag_plain_terminals(self, processor, tokenizer):
        """S* → A B  (no <<>> tags, only NTs)."""
        grammar, _ = processor.process_full_grammar({
            'S*': ["A B"],
            'A':  ["<<x>>"],
            'B':  ["<<y>>"],
        })
        seqs = reachable_terminal_sequences(grammar)
        assert toks(tokenizer, 'x', 'y') in seqs


class TestSharedTagPrefixKeepsContinuation:
    """
    Regression: tags at the same position that share a TOKEN prefix but are
    followed by DIFFERENT continuations must not be collapsed into one shared
    NT chain.

    The chain branches internally (prefix left-factoring), so sharing it across
    productions with different continuations loses the tag->continuation link.
    Two failure modes were possible:
      - the child NTs' FIRST sets collide -> ValueError, rejecting a grammar
        that IS LL(1);
      - the FIRST sets are disjoint -> NO error, and the grammar silently stops
        enforcing the hierarchy (it accepted parent=cs with a child of ece).

    Grouping by (position, continuation) keeps each tag bound to its own
    continuation; the shared token prefix is then factored by step 4 into
    S* -> '{"' … S*_FACT ; S*_FACT -> 'cs' … C_J | 'ece' … D_J  (LL(1)).
    """

    def _grammar(self, processor, tokenizer, c_kids, d_kids):
        prods = {
            'S*':  ['<<{"parent": "cs", "child": ">> C_J',
                    '<<{"parent": "ece", "child": ">> D_J'],
            'C_J': [f'<<{k}">>' for k in c_kids],
            'D_J': [f'<<{k}">>' for k in d_kids],
        }
        grammar, _ = processor.process_full_grammar(prods)
        return reachable_terminal_sequences(grammar)

    def _seq(self, tokenizer, parent, child):
        return toks(tokenizer, f'{{"parent": "{parent}", "child": "', f'{child}"')

    def test_disjoint_first_sets_still_enforce_hierarchy(self, processor, tokenizer):
        """The silent-corruption case: no conflict, so nothing used to complain."""
        langs = self._grammar(processor, tokenizer, ['cryptography'], ['electricity'])

        assert self._seq(tokenizer, 'cs', 'cryptography') in langs
        assert self._seq(tokenizer, 'ece', 'electricity') in langs
        # cross products must NOT be derivable
        assert self._seq(tokenizer, 'cs', 'electricity') not in langs
        assert self._seq(tokenizer, 'ece', 'cryptography') not in langs

    def test_colliding_first_sets_build_and_enforce_hierarchy(self, processor, tokenizer):
        """The hard-error case: 'operating systems' / 'operational amplifier'."""
        langs = self._grammar(processor, tokenizer,
                              ['operating systems'], ['operational amplifier'])

        assert self._seq(tokenizer, 'cs', 'operating systems') in langs
        assert self._seq(tokenizer, 'ece', 'operational amplifier') in langs
        assert self._seq(tokenizer, 'cs', 'operational amplifier') not in langs
        assert self._seq(tokenizer, 'ece', 'operating systems') not in langs

    def test_same_continuation_is_unchanged(self, processor, tokenizer):
        """Common case (child enums): one group per position, grammar as before."""
        prods = {'S*': ['<<alpha>> A', '<<alfa>> A'], 'A': ['<<x>>']}
        grammar, _ = processor.process_full_grammar(prods)
        # a single continuation -> a single group -> no _G split
        assert not [nt for nt, _ in grammar if '_G' in str(nt) and 'POS' in str(nt)]
        langs = reachable_terminal_sequences(grammar)
        assert toks(tokenizer, 'alpha', 'x') in langs
        assert toks(tokenizer, 'alfa', 'x') in langs
