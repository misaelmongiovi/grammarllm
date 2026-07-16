"""
test_decode_fidelity.py
=======================
Regression: generate_text's decoded `text` must spell exactly what the model
emitted, token for token — whatever the tokenizer's own settings say.

HuggingFace's `clean_up_tokenization_spaces` is a legacy heuristic from the
WordPiece era, when tokenizers put a space between every token and the decoder
had to glue punctuation back on:

    " ." -> "."     " ," -> ","     " 's" -> "'s"     " n't" -> "n't"   ...

For a byte-level BPE the spaces already live inside the tokens, so there is
nothing to glue — the cleanup only destroys. On a grammar with a whitespace
terminal it deletes the very separator the PDA consumed: a valid ' . . .' comes
back as '...'.

The failure is silent and self-contradictory: token_ids and pda_stack say the
grammar was satisfied, while `text` holds a string that grammar cannot generate.
The contract — the output conforms to the grammar — breaks in the last step,
after the masking did everything right.

Whether it bites depends on the tokenizer's config, which is why generate_text
must pass the flag explicitly rather than inherit the default:

    Qwen2.5     clean_up_tokenization_spaces = False   (harmless)
    Llama-3     clean_up_tokenization_spaces = True    (mangles)

The suite runs on Qwen, so these tests set the attribute to True to reproduce
the Llama configuration — otherwise they would pass for the wrong reason.
"""

import pytest
import torch

from grammarllm.generate_with_constraints import (
    get_parsing_table_and_map_tt, generate_grammar_parameters, generate_text)


@pytest.fixture(scope="module")
def model(tokenizer):
    from transformers import AutoModelForCausalLM
    from conftest import QWEN
    return AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float32).eval()


@pytest.fixture()
def cleanup_on(tokenizer):
    """Reproduce Llama-3's tokenizer config on the Qwen test tokenizer."""
    old = getattr(tokenizer, "clean_up_tokenization_spaces", None)
    tokenizer.clean_up_tokenization_spaces = True
    yield tokenizer
    tokenizer.clean_up_tokenization_spaces = old


# Each pairs a whitespace terminal with the punctuation that follows it — the
# shape a real grammar produces, and exactly what the cleanup rewrites.
MANGLED = [" . . .", "a , b", "it 's", "do n't"]


@pytest.mark.integration
@pytest.mark.parametrize("target", MANGLED)
def test_decoded_text_is_what_the_grammar_forced(cleanup_on, model, target):
    tokenizer = cleanup_on
    # Grammar: the target, or EOS (get_parsing_table_and_map_tt always appends an
    # EOS alternative to S*). min_new_tokens keeps the model from taking the EOS
    # exit, so the only path left spells out the target.
    n_tokens = len(tokenizer.encode(target, add_special_tokens=False))
    pars_tab, map_tt = get_parsing_table_and_map_tt(
        tokenizer, productions={'S*': [f'<<{target}>>']})
    # token_lookahead=False: get_parsing_table_and_map_tt always appends an EOS
    # alternative to S*, and under lookahead the model can SPELL that terminal
    # with ordinary tokens ('<|im_', 'end', '|>') instead of emitting the atomic
    # EOS id — which min_new_tokens would block. Boundary-strict keeps EOS
    # atomic, so min_new_tokens really does force the target branch.
    pdas, streamer = generate_grammar_parameters(tokenizer, pars_tab, map_tt,
                                                 token_lookahead=False)

    result = generate_text(model, tokenizer, "x", pdas, streamer,
                           max_new_tokens=n_tokens + 2, do_sample=False,
                           min_new_tokens=n_tokens)

    assert result['text'] == target, (
        f"decode mangled the grammar's output: {result['text']!r} != {target!r}"
    )
    assert result['pda_stack'] == [], "grammar not satisfied"


@pytest.mark.integration
def test_text_matches_its_own_token_ids(cleanup_on, model):
    """The result must be self-consistent: `text` is what `token_ids` spell."""
    tokenizer = cleanup_on
    target = " . . ."
    n_tokens = len(tokenizer.encode(target, add_special_tokens=False))
    pars_tab, map_tt = get_parsing_table_and_map_tt(
        tokenizer, productions={'S*': [f'<<{target}>>']})
    pdas, streamer = generate_grammar_parameters(tokenizer, pars_tab, map_tt,
                                                 token_lookahead=False)

    result = generate_text(model, tokenizer, "x", pdas, streamer,
                           max_new_tokens=n_tokens + 2, do_sample=False,
                           min_new_tokens=n_tokens)

    faithful = tokenizer.decode(result['token_ids'], skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
    assert result['text'] == faithful


def test_cleanup_is_what_destroys_the_separator(tokenizer):
    """Pins the mechanism: it is the cleanup, not the tokenizer, that loses the
    whitespace. If HF ever drops the heuristic this test says so."""
    ids = tokenizer.encode(" . . .", add_special_tokens=False)
    assert tokenizer.decode(ids, clean_up_tokenization_spaces=False) == " . . ."
    assert tokenizer.decode(ids, clean_up_tokenization_spaces=True) != " . . ."
