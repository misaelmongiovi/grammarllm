"""
test_chat_template.py
=====================
Regression: generate_text must render conversations with the tokenizer's OWN
chat template by default.

`grammarllm.chat_template` emits <|system|> / <|user|> / <|assistant|>, which are
NOT special tokens for modern instruct models — they shatter into
'<', '|', 'system', '|', '>' and the model sees a chat format it was never
trained on. It used to be the only documented path, and generate_text applied it
whenever it was passed, overwriting the tokenizer's native template.

Impact was large and silent: on WoS with Llama-3.2-1B-Instruct the model stopped
classifying and started echoing the system prompt back (L1 micro-F1 0.539 -> 0.145).
"""

import pytest

from grammarllm import chat_template, create_prompt


@pytest.fixture()
def convo():
    return create_prompt(prompt_input="hi", system_prompt="be brief", examples=None)


def render(tokenizer, convo, template):
    """What generate_text does: only override when a template is explicitly given."""
    if template is not None:
        tokenizer.chat_template = template
    return tokenizer.apply_chat_template(
        [convo], tokenize=False, add_generation_prompt=True)[0]


class TestNativeTemplateIsDefault:

    def test_native_markers_used_when_no_template_passed(self, tokenizer, convo):
        native = tokenizer.chat_template
        assert native is not None, "test needs an instruct tokenizer"
        out = render(tokenizer, convo, None)
        assert "<|system|>" not in out, "must not use the generic fallback template"
        # the tokenizer's own template survived untouched
        assert tokenizer.chat_template == native

    def test_generic_template_still_available_explicitly(self, tokenizer, convo):
        out = render(tokenizer, convo, chat_template)
        assert "<|system|>" in out

    def test_generic_markers_are_not_special_tokens(self, tokenizer):
        """Why the fallback is harmful: the role markers are not atomic."""
        pieces = tokenizer.tokenize("<|system|>")
        assert len(pieces) > 1, (
            "<|system|> tokenized atomically — this tokenizer would be unaffected; "
            "on Llama-3/Qwen it shatters into '<','|','system','|','>'")


class TestConversationDetectedByContent:
    """A conversation is recognised from its shape, not from being handed a template."""

    def test_single_conversation(self, convo):
        assert all(isinstance(m, dict) for m in convo)

    def test_batch_of_conversations(self, convo):
        batch = [convo, convo]
        assert all(isinstance(c, list) and all(isinstance(m, dict) for m in c)
                   for c in batch)

    def test_batch_of_raw_strings_is_not_a_conversation(self):
        batch = ["a", "b"]
        assert not all(isinstance(t, dict) for t in batch)
        assert all(isinstance(t, str) for t in batch)
