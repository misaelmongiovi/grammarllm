# Pydantic → Strict-JSON Grammar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework `grammarllm/utils/pydantic_to_grammar.py` so any constrained generation is valid JSON, round-trippable via `Model.model_validate_json()`.

**Architecture:** The translator emits "skeleton-chunk" grammars: because keys are always present, all JSON punctuation between values is constant and becomes one `<<tag>>` per gap. Open values (string chars, digits) use two regex terminals returned alongside the productions. The validator gains two rules: `int | float` unions rejected; `$ref` cycles allowed only when broken by an Optional/array edge.

**Tech Stack:** Python ≥3.10, pydantic v2 (`model_json_schema()`), existing GrammarLLM pipeline (`get_parsing_table_and_map_tt`, `PushdownAutomaton`), pytest, Hugging Face tokenizer `Qwen/Qwen2.5-0.5B-Instruct` (already in local HF cache).

**Spec:** `docs/superpowers/specs/2026-07-07-pydantic-json-grammar-design.md` — decisions D1–D7 referenced below.

## Global Constraints

- API is a **breaking change**: `pydantic_to_productions(Model)` returns `(productions, regex_dict)` (D2). No back-compat shim.
- Keys are always emitted; only values alternate with `null` (D3). No `_OPT` wrapper NTs anywhere.
- Quotes always belong to the string value NT, never the skeleton (D5).
- Canonical spacing baked into chunks: `": "` after keys, `", "` between pairs/array items, no padding inside `{}` / `[]`.
- Reserved NT names produced by the translator: `S*`, `JSON_STRING`, `JSON_CHARS`, `JSON_INT`, `JSON_SIGN`, `JSON_DIGITS`, `JSON_NUMBER`, `JSON_FRAC`, `JSON_BOOL`, plus `<PARENT>_<FIELD>`, `<NT>_BODY`, `<NT>_TAIL`, `<NT>_BRANCH<i>`, and uppercased `$defs` names.
- Regex terminals: `json_char` → `^[^"\\\x00-\x1f]+$`, `digit` → `^[0-9]+$` (token-level: each regex matches whole vocabulary token strings).
- No JSON escape sequences in v1 (`"`, `\`, control chars excluded from string content) — documented limitation, and enum values / field names containing them are rejected at conversion time.
- **Tests are NOT committed**: `grammarllm/tests/` is gitignored by user choice. Every commit step adds source/docs only; tests stay local. Run tests with `uv run --with pytest pytest …` (project venv has no pytest).
- All conversion failures raise `PydanticGrammarError` with messages naming the offending field and the fix, phrased in pydantic vocabulary.

## File Structure

| File | Responsibility |
|---|---|
| `grammarllm/utils/pydantic_to_grammar.py` (rewrite in place) | `PydanticGrammarError`, `_Validator` (Phase 1: reject non-LL(1) constructs, D6/D7), `_Translator` (Phase 2: schema → skeleton-chunk productions), `pydantic_to_productions()` public API |
| `grammarllm/tests/test_pydantic_json.py` (new, local-only) | All four test layers for the new behavior |
| `grammarllm/tests/test_pydantic_to_grammar.py` (delete in Task 9) | Old compact-format tests; still-valid rejection tests move to the new file |
| `docs/usage.md`, `README.md` (modify in Task 9) | Replace "experimental compact" pydantic sections with the strict-JSON API |

---

### Task 1: Tuple API + object skeleton + string enums

**Files:**
- Modify: `grammarllm/utils/pydantic_to_grammar.py` (module constants, `_Translator.__init__/translate/_emit_object_nt/_emit_enum/_value_symbol`, `pydantic_to_productions` return type)
- Test: `grammarllm/tests/test_pydantic_json.py` (new)

**Interfaces:**
- Consumes: existing `PydanticGrammarError`, `_Validator`, `_nt_name`, `_resolve_ref` (unchanged in this task).
- Produces: `pydantic_to_productions(model, type_terminal_map=None) -> tuple[dict[str, list[str]], dict[str, re.Pattern]]`; `_Translator._value_symbol(parent_nt: str, slot_name: str, schema: dict) -> str` (the dispatch hook every later task extends); `DEFAULT_TYPE_TERMINAL_MAP = {"string": "json_char", "integer": "digit", "number": "digit"}`.

- [ ] **Step 1: Write the failing tests**

Create `grammarllm/tests/test_pydantic_json.py`:

```python
"""
Tests for the strict-JSON pydantic converter.
Spec: docs/superpowers/specs/2026-07-07-pydantic-json-grammar-design.md

Layers:
  1. unit (this file, no tokenizer)
  2. integration (real tokenizer, marked 'integration')
  3. random-walk round-trip (marked 'integration')
  4. E2E smoke (marked 'e2e', slow)
"""
import json
import re
from typing import Literal, Optional

import pytest
from pydantic import BaseModel

from grammarllm.utils.pydantic_to_grammar import (
    pydantic_to_productions,
    PydanticGrammarError,
)


def reachable_nts(productions):
    seen, frontier = {"S*"}, ["S*"]
    while frontier:
        nt = frontier.pop()
        for prod in productions[nt]:
            for sym in prod.split():
                if sym in productions and sym not in seen:
                    seen.add(sym)
                    frontier.append(sym)
    return seen


class TestTupleApiAndSkeleton:
    def test_returns_productions_and_regex_dict(self):
        class Sentiment(BaseModel):
            label: Literal["positive", "negative"]

        prods, regex_dict = pydantic_to_productions(Sentiment)
        assert isinstance(prods, dict)
        assert regex_dict["regex_json_char"].match("hello")
        assert not regex_dict["regex_json_char"].match('he"llo')
        assert regex_dict["regex_digit"].match("42")
        assert not regex_dict["regex_digit"].match("4a")

    def test_single_field_skeleton(self):
        class Sentiment(BaseModel):
            label: Literal["positive", "negative"]

        prods, _ = pydantic_to_productions(Sentiment)
        assert prods["S*"] == ['<<{"label": >> S*_LABEL <<}>>']
        assert prods["S*_LABEL"] == ['<<"positive">>', '<<"negative">>']

    def test_two_field_skeleton_has_comma_chunk(self):
        class Pair(BaseModel):
            a: Literal["x"]
            b: Literal["y"]

        prods, _ = pydantic_to_productions(Pair)
        assert prods["S*"] == ['<<{"a": >> S*_A <<, "b": >> S*_B <<}>>']

    def test_all_nts_reachable(self):
        class Pair(BaseModel):
            a: Literal["x"]
            b: Literal["y"]

        prods, _ = pydantic_to_productions(Pair)
        assert reachable_nts(prods) == set(prods)

    def test_enum_value_with_quote_rejected(self):
        class Bad(BaseModel):
            label: Literal['say "hi"']

        with pytest.raises(PydanticGrammarError, match="escape"):
            pydantic_to_productions(Bad)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest pytest grammarllm/tests/test_pydantic_json.py -q`
Expected: FAIL — `test_returns_productions_and_regex_dict` errors with "cannot unpack non-sequence dict" (old API returns a dict), the others fail on assertions.

- [ ] **Step 3: Implement**

In `grammarllm/utils/pydantic_to_grammar.py`, replace the module constants:

```python
DEFAULT_TYPE_TERMINAL_MAP: dict[str, str] = {
    "string": "json_char",
    "integer": "digit",
    "number": "digit",
}

_JSON_CHAR_REGEX = r'^[^"\\\x00-\x1f]+$'
_DIGIT_REGEX = r"^[0-9]+$"

_UNSAFE_LITERAL = re.compile(r'["\\\x00-\x1f]')
```

Replace the entire `_Translator` class:

```python
class _Translator:
    """
    Translates a validated JSON Schema into strict-JSON skeleton-chunk
    productions (spec D3/D4/D5): keys are always present, every constant
    JSON fragment between value slots is a single <<tag>>, and quotes
    always belong to the string value NT.
    """

    def __init__(self, defs: dict[str, Any], type_terminal_map: dict[str, str]) -> None:
        self._defs = defs
        self._char_terminal = type_terminal_map["string"]
        self._digit_terminal = type_terminal_map["integer"]
        self._productions: dict[str, list[str]] = {}
        self._emitted: set[str] = set()

    def translate(self, root_schema: dict[str, Any]) -> dict[str, list[str]]:
        if "$ref" in root_schema:
            # Recursive root model: S* aliases the named def NT (Task 6).
            self._productions["S*"] = [self._ref_symbol(root_schema["$ref"])]
        else:
            self._emit_object_nt("S*", root_schema)
        return self._productions

    # ── object skeleton (D4) ───────────────────────────────────────────

    def _emit_object_nt(self, nt: str, schema: dict[str, Any]) -> None:
        if nt in self._emitted:
            return
        self._emitted.add(nt)
        if "allOf" in schema:
            schema = self._flatten_all_of(schema["allOf"])
        props = schema.get("properties", {})
        parts: list[str] = []
        chunk = "{"
        for i, (field_name, field_schema) in enumerate(props.items()):
            if _UNSAFE_LITERAL.search(field_name):
                raise PydanticGrammarError(
                    f"Field name {field_name!r} contains a quote, backslash or "
                    "control character. v1 emits no JSON escape sequences — "
                    "rename the field or use a safe alias."
                )
            if i:
                chunk += ", "
            chunk += f'"{field_name}": '
            parts.append(f"<<{chunk}>>")
            chunk = ""
            parts.append(self._value_symbol(nt, field_name, field_schema))
        parts.append("<<}>>")
        self._productions[nt] = [" ".join(parts)]

    # ── value slot dispatch ────────────────────────────────────────────

    def _value_symbol(self, parent_nt: str, slot_name: str, schema: dict[str, Any]) -> str:
        """Return the grammar symbol for a value slot, emitting sub-NTs as needed."""
        if "enum" in schema:
            nt = f"{parent_nt}_{slot_name.upper()}"
            self._emit_enum(nt, schema["enum"])
            return nt
        raise PydanticGrammarError(
            f"Cannot translate value schema for '{parent_nt}.{slot_name}': "
            f"unrecognised shape {schema}"
        )

    # ── enum (quotes inside the tag, D5) ───────────────────────────────

    def _emit_enum(self, nt: str, values: list[Any]) -> None:
        alts: list[str] = []
        for v in values:
            if not isinstance(v, str):
                raise PydanticGrammarError(
                    f"NT '{nt}': enum value {v!r} is not a string. "
                    "Use Literal['a', 'b'] with string values."
                )
            if _UNSAFE_LITERAL.search(v):
                raise PydanticGrammarError(
                    f"NT '{nt}': enum value {v!r} contains a quote, backslash or "
                    "control character; v1 emits no JSON escape sequences."
                )
            alts.append(f'<<"{v}">>')
        self._productions[nt] = alts

    # ── allOf flattening (unchanged semantics) ─────────────────────────

    def _flatten_all_of(self, branches: list[dict]) -> dict[str, Any]:
        merged_props: dict[str, Any] = {}
        merged_required: list[str] = []
        for branch in branches:
            resolved = self._inline(branch)
            merged_props.update(resolved.get("properties", {}))
            merged_required.extend(resolved.get("required", []))
        return {
            "type": "object",
            "properties": merged_props,
            "required": list(dict.fromkeys(merged_required)),
        }

    # ── helpers ────────────────────────────────────────────────────────

    def _inline(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Follow a $ref for inspection purposes only (no NT emission)."""
        if "$ref" in schema:
            return self._defs[schema["$ref"][len("#/$defs/"):]]
        return schema

    def _ref_symbol(self, ref: str) -> str:
        name = ref[len("#/$defs/"):]
        nt = _nt_name(name)
        if nt not in self._emitted:
            target = self._defs[name]
            if "enum" in target:
                self._emitted.add(nt)
                self._emit_enum(nt, target["enum"])
            else:
                self._emit_object_nt(nt, target)
        return nt
```

Replace the tail of `pydantic_to_productions` (after the validator call):

```python
    # ── Phase 2: translation ─────────────────────────────────────────────
    translator = _Translator(defs=defs, type_terminal_map=ttmap)
    productions = translator.translate(schema)

    # ── Regex terminals for open values (token-level regexes) ───────────
    regex_dict = {
        f"regex_{ttmap['string']}": re.compile(_JSON_CHAR_REGEX),
        f"regex_{ttmap['integer']}": re.compile(_DIGIT_REGEX),
    }

    return productions, regex_dict
```

Update the function's docstring example to the tuple form:

```python
    >>> productions, regex_dict = pydantic_to_productions(Sentiment)
    >>> # productions == {
    >>> #     'S*': ['<<{"label": >> S*_LABEL <<}>>'],
    >>> #     'S*_LABEL': ['<<"positive">>', '<<"negative">>', '<<"neutral">>'],
    >>> # }
```

Delete the now-dead `_is_nullable` / `_non_null_branch` module helpers only if nothing references them after Task 3 — leave them for now (Task 3 reuses the nullable concept inline).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest pytest grammarllm/tests/test_pydantic_json.py -q`
Expected: `5 passed`. (The OLD `test_pydantic_to_grammar.py` now fails — expected; it tests the removed compact format and is deleted in Task 9. Do not run it until then.)

- [ ] **Step 5: Commit (source only — tests are gitignored)**

```bash
git add grammarllm/utils/pydantic_to_grammar.py
git commit -m "feat(pydantic): tuple API + strict-JSON object skeleton + string enums"
```

---

### Task 2: Primitive values (str / int / float / bool)

**Files:**
- Modify: `grammarllm/utils/pydantic_to_grammar.py` (`_Translator._value_symbol` dispatch + four new `_json_*_nt` methods)
- Test: `grammarllm/tests/test_pydantic_json.py`

**Interfaces:**
- Consumes: `_value_symbol(parent_nt, slot_name, schema)` dispatch from Task 1; `self._char_terminal` / `self._digit_terminal`.
- Produces: `_json_string_nt() -> "JSON_STRING"`, `_json_int_nt() -> "JSON_INT"`, `_json_number_nt() -> "JSON_NUMBER"`, `_json_bool_nt() -> "JSON_BOOL"` — shared NTs emitted once, reused by Tasks 3–6.

- [ ] **Step 1: Write the failing tests**

Append to `test_pydantic_json.py`:

```python
class TestPrimitives:
    def test_string_field(self):
        class Doc(BaseModel):
            title: str

        prods, _ = pydantic_to_productions(Doc)
        assert prods["S*"] == ['<<{"title": >> JSON_STRING <<}>>']
        assert prods["JSON_STRING"] == ['<<">> JSON_CHARS <<">>']
        assert prods["JSON_CHARS"] == ["json_char JSON_CHARS", "ε"]

    def test_int_field(self):
        class Age(BaseModel):
            age: int

        prods, _ = pydantic_to_productions(Age)
        assert prods["S*"] == ['<<{"age": >> JSON_INT <<}>>']
        assert prods["JSON_INT"] == ["JSON_SIGN digit JSON_DIGITS"]
        assert prods["JSON_SIGN"] == ["<<->>", "ε"]
        assert prods["JSON_DIGITS"] == ["digit JSON_DIGITS", "ε"]

    def test_float_field(self):
        class Price(BaseModel):
            price: float

        prods, _ = pydantic_to_productions(Price)
        assert prods["S*"] == ['<<{"price": >> JSON_NUMBER <<}>>']
        assert prods["JSON_NUMBER"] == ["JSON_INT JSON_FRAC"]
        assert prods["JSON_FRAC"] == ["<<.>> digit JSON_DIGITS", "ε"]

    def test_bool_field(self):
        class Flag(BaseModel):
            active: bool

        prods, _ = pydantic_to_productions(Flag)
        assert prods["S*"] == ['<<{"active": >> JSON_BOOL <<}>>']
        assert prods["JSON_BOOL"] == ["<<true>>", "<<false>>"]

    def test_shared_nts_emitted_once(self):
        class Two(BaseModel):
            a: str
            b: str

        prods, _ = pydantic_to_productions(Two)
        # both fields reference the same shared NT — no duplicates
        assert prods["S*"] == ['<<{"a": >> JSON_STRING <<, "b": >> JSON_STRING <<}>>']
        assert reachable_nts(prods) == set(prods)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest pytest grammarllm/tests/test_pydantic_json.py -q -k Primitives`
Expected: FAIL with `PydanticGrammarError: Cannot translate value schema` (dispatch doesn't know `type: string` yet).

- [ ] **Step 3: Implement**

Extend `_value_symbol` (insert before the final `raise`):

```python
        t = schema.get("type")
        if t == "string":
            return self._json_string_nt()
        if t == "integer":
            return self._json_int_nt()
        if t == "number":
            return self._json_number_nt()
        if t == "boolean":
            return self._json_bool_nt()
```

Add the shared-NT methods to `_Translator`:

```python
    # ── shared primitive NTs (emitted once, reused everywhere) ─────────

    def _json_string_nt(self) -> str:
        if "JSON_STRING" not in self._productions:
            self._productions["JSON_STRING"] = ['<<">> JSON_CHARS <<">>']
            self._productions["JSON_CHARS"] = [f"{self._char_terminal} JSON_CHARS", "ε"]
        return "JSON_STRING"

    def _json_int_nt(self) -> str:
        if "JSON_INT" not in self._productions:
            self._productions["JSON_INT"] = [f"JSON_SIGN {self._digit_terminal} JSON_DIGITS"]
            self._productions["JSON_SIGN"] = ["<<->>", "ε"]
            self._productions["JSON_DIGITS"] = [f"{self._digit_terminal} JSON_DIGITS", "ε"]
        return "JSON_INT"

    def _json_number_nt(self) -> str:
        if "JSON_NUMBER" not in self._productions:
            self._json_int_nt()
            self._productions["JSON_NUMBER"] = ["JSON_INT JSON_FRAC"]
            self._productions["JSON_FRAC"] = [f"<<.>> {self._digit_terminal} JSON_DIGITS", "ε"]
        return "JSON_NUMBER"

    def _json_bool_nt(self) -> str:
        if "JSON_BOOL" not in self._productions:
            self._productions["JSON_BOOL"] = ["<<true>>", "<<false>>"]
        return "JSON_BOOL"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest pytest grammarllm/tests/test_pydantic_json.py -q`
Expected: `10 passed`.

- [ ] **Step 5: Commit**

```bash
git add grammarllm/utils/pydantic_to_grammar.py
git commit -m "feat(pydantic): shared JSON primitive NTs (string/int/float/bool)"
```

---

### Task 3: Optional / anyOf values + int|float rejection (D7)

**Files:**
- Modify: `grammarllm/utils/pydantic_to_grammar.py` (`_value_symbol` anyOf branch, `_Translator._emit_any_of`, `_Validator._validate_any_of`)
- Test: `grammarllm/tests/test_pydantic_json.py`

**Interfaces:**
- Consumes: `_value_symbol` dispatch, `_json_*_nt` helpers, `_inline`.
- Produces: `_emit_any_of(nt: str, schema: dict) -> None` — NT with one alternative per non-null branch plus `<<null>>` when nullable. Branch slot naming: `{nt}_BRANCH{i}` reserved for future non-shared branch NTs (objects/arrays inside unions, Tasks 4–5 route through `_value_symbol` so shared NTs stay shared).

- [ ] **Step 1: Write the failing tests**

Append to `test_pydantic_json.py`:

```python
class TestOptionalAndUnions:
    def test_optional_enum(self):
        class P(BaseModel):
            mood: Optional[Literal["happy", "sad"]] = None

        prods, _ = pydantic_to_productions(P)
        # key always present (D3); the VALUE alternates with null
        assert prods["S*"] == ['<<{"mood": >> S*_MOOD <<}>>']
        assert prods["S*_MOOD"] == ["S*_MOOD_BRANCH0", "<<null>>"]
        assert prods["S*_MOOD_BRANCH0"] == ['<<"happy">>', '<<"sad">>']
        # no double-epsilon _OPT wrapper anywhere
        assert not any(nt.endswith("_OPT") for nt in prods)

    def test_optional_string_uses_shared_nt(self):
        class P(BaseModel):
            note: Optional[str] = None

        prods, _ = pydantic_to_productions(P)
        assert prods["S*_NOTE"] == ["JSON_STRING", "<<null>>"]

    def test_int_str_union_allowed(self):
        class V(BaseModel):
            data: int | str

        prods, _ = pydantic_to_productions(V)
        assert prods["S*_DATA"] == ["JSON_INT", "JSON_STRING"]

    def test_int_float_union_rejected(self):
        class V(BaseModel):
            data: int | float

        with pytest.raises(PydanticGrammarError, match="digit"):
            pydantic_to_productions(V)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest pytest grammarllm/tests/test_pydantic_json.py -q -k Optional`
Expected: FAIL — `Cannot translate value schema` for anyOf; the D7 test fails because no error is raised.

- [ ] **Step 3: Implement**

Extend `_value_symbol` (insert BEFORE the `"enum"` check, since Optional enums arrive as `anyOf: [{enum...}, {type: null}]`):

```python
        if "anyOf" in schema or "oneOf" in schema:
            nt = f"{parent_nt}_{slot_name.upper()}"
            self._emit_any_of(nt, schema)
            return nt
```

Add to `_Translator`:

```python
    # ── anyOf / oneOf (Optional[X] → alternatives + <<null>>) ──────────

    def _emit_any_of(self, nt: str, schema: dict[str, Any]) -> None:
        branches = schema.get("anyOf", schema.get("oneOf", []))
        alts: list[str] = []
        branch_idx = 0
        for branch in branches:
            if branch.get("type") == "null":
                continue
            alts.append(self._value_symbol(nt, f"branch{branch_idx}", branch))
            branch_idx += 1
        if any(b.get("type") == "null" for b in branches):
            alts.append("<<null>>")
        self._productions[nt] = alts
```

In `_Validator._validate_any_of`, add the D7 check right after `branch_types` is built (before the same-type loop):

```python
        # D7: int and float branches both start with a digit — guaranteed
        # FIRST-set conflict at the token level.
        if "integer" in branch_types and "number" in branch_types:
            raise PydanticGrammarError(
                f"[{path}] anyOf/oneOf mixes 'int' and 'float' branches. "
                "Both start with a digit, so an LL(1) parser cannot tell them "
                "apart. Use a single float field (ints are valid floats)."
            )
```

Note: `_emit_enum` is reached for Optional enums via `_value_symbol(nt, "branch0", branch)` → NT name `S*_MOOD_BRANCH0`, matching the test.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest pytest grammarllm/tests/test_pydantic_json.py -q`
Expected: `14 passed`.

- [ ] **Step 5: Commit**

```bash
git add grammarllm/utils/pydantic_to_grammar.py
git commit -m "feat(pydantic): nullable values via anyOf + reject int|float unions (D7)"
```

---

### Task 4: Nested objects and non-recursive $ref

**Files:**
- Modify: `grammarllm/utils/pydantic_to_grammar.py` (`_value_symbol` object/$ref branches)
- Test: `grammarllm/tests/test_pydantic_json.py`

**Interfaces:**
- Consumes: `_emit_object_nt`, `_ref_symbol` (Task 1), `_emit_any_of` (Task 3).
- Produces: `_value_symbol` handles `{"type": "object"}` (inline nested model → `{parent}_{FIELD}` NT) and `{"$ref": ...}` (named shared NT, e.g. `ADDRESS`).

- [ ] **Step 1: Write the failing tests**

Append to `test_pydantic_json.py`:

```python
class TestNestedAndRef:
    def test_ref_becomes_named_shared_nt(self):
        class Address(BaseModel):
            city: Literal["rome", "milan"]

        class Person(BaseModel):
            name: Literal["mario"]
            home: Address
            office: Address

        prods, _ = pydantic_to_productions(Person)
        assert prods["S*"] == [
            '<<{"name": >> S*_NAME <<, "home": >> ADDRESS <<, "office": >> ADDRESS <<}>>'
        ]
        # one shared definition, quotes/braces in its own skeleton
        assert prods["ADDRESS"] == ['<<{"city": >> ADDRESS_CITY <<}>>']
        assert prods["ADDRESS_CITY"] == ['<<"rome">>', '<<"milan">>']
        assert reachable_nts(prods) == set(prods)

    def test_optional_ref(self):
        class Address(BaseModel):
            city: Literal["rome"]

        class Person(BaseModel):
            home: Optional[Address] = None

        prods, _ = pydantic_to_productions(Person)
        assert prods["S*_HOME"] == ["ADDRESS", "<<null>>"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest pytest grammarllm/tests/test_pydantic_json.py -q -k NestedAndRef`
Expected: FAIL with `Cannot translate value schema` (dispatch lacks `$ref`/object).

- [ ] **Step 3: Implement**

Extend `_value_symbol` — insert at the TOP of the method (a `$ref` has no `type`/`enum` keys of its own):

```python
        if "$ref" in schema:
            return self._ref_symbol(schema["$ref"])
```

and insert with the other `type` branches:

```python
        if t == "object" or "properties" in schema or "allOf" in schema:
            nt = f"{parent_nt}_{slot_name.upper()}"
            self._emit_object_nt(nt, schema)
            return nt
```

(`_ref_symbol` and `_emit_object_nt` already exist from Task 1 — no other changes.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest pytest grammarllm/tests/test_pydantic_json.py -q`
Expected: `16 passed`.

- [ ] **Step 5: Commit**

```bash
git add grammarllm/utils/pydantic_to_grammar.py
git commit -m "feat(pydantic): nested objects and shared named NTs for \$refs"
```

---

### Task 5: Arrays

**Files:**
- Modify: `grammarllm/utils/pydantic_to_grammar.py` (`_value_symbol` array branch + `_emit_array`)
- Test: `grammarllm/tests/test_pydantic_json.py`

**Interfaces:**
- Consumes: `_value_symbol` dispatch.
- Produces: `_emit_array(nt: str, schema: dict) -> None` emitting `nt`, `{nt}_BODY`, `{nt}_TAIL`. Empty arrays allowed (spec non-goal: no minItems).

- [ ] **Step 1: Write the failing tests**

Append to `test_pydantic_json.py`:

```python
class TestArrays:
    def test_array_of_enum(self):
        class Tags(BaseModel):
            tags: list[Literal["a", "b"]]

        prods, _ = pydantic_to_productions(Tags)
        assert prods["S*"] == ['<<{"tags": >> S*_TAGS <<}>>']
        assert prods["S*_TAGS"] == ["<<[>> S*_TAGS_BODY"]
        assert prods["S*_TAGS_BODY"] == ["S*_TAGS_ITEM S*_TAGS_TAIL", "<<]>>"]
        assert prods["S*_TAGS_TAIL"] == ["<<, >> S*_TAGS_ITEM S*_TAGS_TAIL", "<<]>>"]
        assert prods["S*_TAGS_ITEM"] == ['<<"a">>', '<<"b">>']

    def test_array_of_int_uses_shared_nt(self):
        class Nums(BaseModel):
            nums: list[int]

        prods, _ = pydantic_to_productions(Nums)
        assert prods["S*_NUMS_BODY"] == ["JSON_INT S*_NUMS_TAIL", "<<]>>"]
        assert reachable_nts(prods) == set(prods)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest pytest grammarllm/tests/test_pydantic_json.py -q -k Arrays`
Expected: FAIL with `Cannot translate value schema` for `type: array`.

- [ ] **Step 3: Implement**

Extend `_value_symbol` with:

```python
        if t == "array":
            nt = f"{parent_nt}_{slot_name.upper()}"
            self._emit_array(nt, schema)
            return nt
```

Add to `_Translator`:

```python
    # ── array: right-recursive, empty allowed, ',' vs ']' disjoint ─────

    def _emit_array(self, nt: str, schema: dict[str, Any]) -> None:
        body_nt = f"{nt}_BODY"
        tail_nt = f"{nt}_TAIL"
        item_symbol = self._value_symbol(nt, "item", schema["items"])
        self._productions[nt] = [f"<<[>> {body_nt}"]
        self._productions[body_nt] = [f"{item_symbol} {tail_nt}", "<<]>>"]
        self._productions[tail_nt] = [f"<<, >> {item_symbol} {tail_nt}", "<<]>>"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest pytest grammarllm/tests/test_pydantic_json.py -q`
Expected: `18 passed`.

- [ ] **Step 5: Commit**

```bash
git add grammarllm/utils/pydantic_to_grammar.py
git commit -m "feat(pydantic): array values as right-recursive [item, ...] grammars"
```

---

### Task 6: Recursion — D6 breakable-cycle validation + recursive translation

**Files:**
- Modify: `grammarllm/utils/pydantic_to_grammar.py` (`_Validator.__init__/_validate_ref/_validate_any_of/validate` array branch)
- Test: `grammarllm/tests/test_pydantic_json.py`

**Interfaces:**
- Consumes: `_ref_symbol` + `_emitted` guard (translation of recursion already works — `_emit_object_nt` registers the NT before descending into fields, so a self-reference returns the name without re-emitting).
- Produces: validator that allows `$ref` cycles crossed by ≥1 breakable edge (nullable-anyOf branch or array items) and rejects the rest. Internal state: `self._ref_stack: list[tuple[str, int]]`, `self._breakable_count: int`.

- [ ] **Step 1: Write the failing tests**

Append to `test_pydantic_json.py`:

```python
class TestRecursion:
    def test_optional_self_recursion_allowed(self):
        class Node(BaseModel):
            value: int
            next: Optional["Node"] = None

        prods, _ = pydantic_to_productions(Node)
        # root is the $ref'd def → S* aliases NODE
        assert prods["S*"] == ["NODE"]
        assert prods["NODE"] == ['<<{"value": >> JSON_INT <<, "next": >> NODE_NEXT <<}>>']
        assert prods["NODE_NEXT"] == ["NODE", "<<null>>"]
        assert reachable_nts(prods) == set(prods)

    def test_list_self_recursion_allowed(self):
        class Tree(BaseModel):
            name: Literal["n"]
            children: list["Tree"]

        prods, _ = pydantic_to_productions(Tree)
        assert "TREE" in prods
        assert prods["TREE_CHILDREN_BODY"] == ["TREE TREE_CHILDREN_TAIL", "<<]>>"]

    def test_unbroken_cycle_rejected(self):
        class A(BaseModel):
            b: "B"

        class B(BaseModel):
            a: A

        A.model_rebuild()
        with pytest.raises(PydanticGrammarError, match="Optional"):
            pydantic_to_productions(A)

    def test_required_self_recursion_rejected(self):
        class Loop(BaseModel):
            inner: "Loop"

        Loop.model_rebuild()
        with pytest.raises(PydanticGrammarError, match="Optional"):
            pydantic_to_productions(Loop)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest pytest grammarllm/tests/test_pydantic_json.py -q -k Recursion`
Expected: the two "allowed" tests FAIL with the old `Cyclic $ref detected` error; the two "rejected" tests may pass already (error message check `match="Optional"` should pass — the old message contains "Optional"). At least the first two must fail.

- [ ] **Step 3: Implement**

Rewrite `_Validator.__init__`:

```python
    def __init__(self, defs: dict[str, Any]) -> None:
        self._defs = defs
        # Stack of (def_name, breakable_count at push) — D6 cycle detection.
        self._ref_stack: list[tuple[str, int]] = []
        # Number of breakable edges (nullable-anyOf branch or array items)
        # on the current descent path.
        self._breakable_count = 0
```

Add a helper method:

```python
    def _validate_via_breakable_edge(self, schema: dict[str, Any], path: str) -> None:
        """Descend through an edge the model can terminate (null / ])."""
        self._breakable_count += 1
        try:
            self.validate(schema, path)
        finally:
            self._breakable_count -= 1
```

Rewrite `_validate_ref`:

```python
    def _validate_ref(self, ref: str, path: str) -> None:
        target_name = ref[len("#/$defs/"):] if ref.startswith("#/$defs/") else None
        if target_name is None:
            raise PydanticGrammarError(
                f"[{path}] Unsupported $ref '{ref}'. Only '#/$defs/<name>' is allowed."
            )

        # D6: a cycle is allowed iff at least one breakable edge (Optional
        # value or array item) lies on the path since the ref was entered —
        # the model can then terminate the recursion with null or ].
        for name, count_at_push in self._ref_stack:
            if name == target_name:
                if self._breakable_count > count_at_push:
                    return  # breakable cycle — this def is already being validated
                raise PydanticGrammarError(
                    f"[{path}] Cyclic $ref to '{target_name}' with no way to "
                    "terminate: every field on the cycle is required and "
                    "non-nullable, so generation could never end. Make one "
                    "field on the cycle Optional[...] (or a list[...])."
                )

        resolved = _resolve_ref(ref, self._defs)
        self._ref_stack.append((target_name, self._breakable_count))
        try:
            self.validate(resolved, path=f"{path}→{target_name}")
        finally:
            self._ref_stack.pop()
```

In `validate()`, change the array branch to use the breakable edge:

```python
        if schema_type == "array":
            items = schema.get("items")
            if items is None:
                raise PydanticGrammarError(
                    f"[{path}] 'array' without 'items' is an open schema. "
                    "Annotate your list field with a concrete element type, "
                    "e.g. list[str] instead of list."
                )
            self._validate_via_breakable_edge(items, path=f"{path}[items]")
            return
```

In `_validate_any_of`, change the final recursion loop so nullable unions descend via a breakable edge:

```python
        nullable = len(non_null) < len(branches)
        for idx, branch in enumerate(non_null):
            resolved = _resolve_ref(branch["$ref"], self._defs) if "$ref" in branch else branch
            branch_path = f"{path}/anyOf[{idx}]"
            if nullable:
                self._validate_via_breakable_edge(resolved, branch_path)
            else:
                self.validate(resolved, branch_path)
```

Note on `_validate_any_of` resolving refs inline: it validates the resolved target directly instead of going through `_validate_ref`, which would skip cycle tracking. Fix that too — replace the resolution above with a dispatch through `validate` so `$ref` branches hit `_validate_ref`:

```python
        nullable = len(non_null) < len(branches)
        for idx, branch in enumerate(non_null):
            branch_path = f"{path}/anyOf[{idx}]"
            if nullable:
                self._validate_via_breakable_edge(branch, branch_path)
            else:
                self.validate(branch, branch_path)
```

(`validate()` already routes `"$ref" in schema` to `_validate_ref`; the type-collection loop earlier in `_validate_any_of` may keep using `_resolve_ref` for inspection only.)

Also update the type-collection loop in `_validate_any_of`: a `$ref` branch has no `type` key; inspect the resolved target for typing but treat it as `"object"`:

```python
        branch_types: list[str] = []
        for idx, branch in enumerate(non_null):
            resolved = _resolve_ref(branch["$ref"], self._defs) if "$ref" in branch else branch
            t = resolved.get("type")
            if t is None and "enum" not in resolved and "properties" not in resolved:
                raise PydanticGrammarError(
                    f"[{path}] anyOf/oneOf branch {idx} has no 'type' and no 'enum'. "
                    "Each branch must be a concrete typed schema or an enum."
                )
            branch_types.append(t or ("enum" if "enum" in resolved else "object"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest pytest grammarllm/tests/test_pydantic_json.py -q`
Expected: `22 passed`.

- [ ] **Step 5: Commit**

```bash
git add grammarllm/utils/pydantic_to_grammar.py
git commit -m "feat(pydantic): allow \$ref cycles broken by Optional/array edges (D6)"
```

---

### Task 7: Integration layer — real tokenizer builds the parsing table

**Files:**
- Test: `grammarllm/tests/test_pydantic_json.py` (append)

**Interfaces:**
- Consumes: `pydantic_to_productions` (complete after Task 6); `grammarllm.get_parsing_table_and_map_tt`.
- Produces: `REPRESENTATIVE_MODELS` list + `built_grammar(model)` helper reused by Task 8.

This layer exists because the previous converter's bugs (double-ε, unreachable NTs) passed every unit test and only crashed inside the LL(1) table builder. No new source code — if the table builder raises for any representative model, fix the translator (that is the test's job).

- [ ] **Step 1: Write the tests**

Append to `test_pydantic_json.py`:

```python
# ── Layer 2: integration with the real pipeline ──────────────────────────────

class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]

class Profile(BaseModel):
    name: str
    age: int
    score: float
    active: bool
    mood: Optional[Literal["happy", "sad"]] = None

class Address(BaseModel):
    city: Literal["rome", "milan"]

class Person(BaseModel):
    name: Literal["mario", "luisa"]
    address: Address
    tags: list[Literal["a", "b"]]

class Node(BaseModel):
    value: int
    next: Optional["Node"] = None

REPRESENTATIVE_MODELS = [Sentiment, Profile, Person, Node]


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


def built_grammar(model, tokenizer):
    from grammarllm.generate_with_constraints import get_parsing_table_and_map_tt
    prods, regex_dict = pydantic_to_productions(model)
    return get_parsing_table_and_map_tt(tokenizer, prods, regex_dict)


@pytest.mark.integration
@pytest.mark.parametrize("model", REPRESENTATIVE_MODELS)
def test_parsing_table_builds(model, tokenizer):
    pars_table, map_tt = built_grammar(model, tokenizer)
    assert "S*" in pars_table
    # every regex terminal produced at least one vocabulary token
    assert map_tt["json_char"], "json_char matched no vocab tokens"
    assert map_tt["digit"], "digit matched no vocab tokens"
```

Register the markers — create `grammarllm/tests/pytest.ini` (local-only, gitignored with the rest of the dir):

```ini
[pytest]
markers =
    integration: needs the HF tokenizer from the local cache
    e2e: loads the full model; slow
```

- [ ] **Step 2: Run the tests**

Run: `uv run --with pytest pytest grammarllm/tests/test_pydantic_json.py -q -m integration`
Expected: `4 passed`. If any model raises `Conflict:` or `Token conflict:`, the translator has an LL(1) bug — fix it before proceeding (do not weaken the test).

- [ ] **Step 3: Run the full local suite**

Run: `uv run --with pytest pytest grammarllm/tests/test_pydantic_json.py grammarllm/tests/test_dev_review_fixes.py -q`
Expected: all pass.

- [ ] **Step 4: Commit**

Tests are gitignored — nothing to commit unless Step 2 forced a translator fix:

```bash
git diff --quiet grammarllm/utils/pydantic_to_grammar.py || {
  git add grammarllm/utils/pydantic_to_grammar.py
  git commit -m "fix(pydantic): LL(1) issues surfaced by table-builder integration tests"
}
```

---

### Task 8: Random-walk round-trip + E2E smoke

**Files:**
- Test: `grammarllm/tests/test_pydantic_json.py` (append)

**Interfaces:**
- Consumes: `built_grammar` / `REPRESENTATIVE_MODELS` (Task 7); `grammarllm.generate_grammar_parameters`, `grammarllm.generate_text`; `PushdownAutomaton.get_tokens/next_state/eos` (`grammarllm/modules/automaton.py`).
- Produces: proof that every PDA-reachable output is valid JSON for its model — without loading model weights.

- [ ] **Step 1: Write the random-walk test**

Append to `test_pydantic_json.py`:

```python
# ── Layer 3: random-walk round-trip (no model weights needed) ────────────────

import random


def random_walk_text(pars_table, map_tt, tokenizer, seed, max_steps=400):
    """Follow the PDA choosing random valid tokens until the grammar is
    satisfied. Returns the decoded text of a complete derivation."""
    from grammarllm.modules.automaton import PushdownAutomaton

    pda = PushdownAutomaton(grammar=pars_table, startSymbol="S*", map=map_tt)
    rng = random.Random(seed)
    eos_id = tokenizer.eos_token_id
    token_ids = []
    for _ in range(max_steps):
        if pda.eos():
            return tokenizer.decode(token_ids)
        valid = pda.get_tokens()
        assert valid, f"dead end at stack {pda.stack} after {token_ids}"
        # never end via the top-level EOS alternative: we want a full JSON object
        choices = [t for t in valid if t != eos_id] or valid
        token = rng.choice(choices)
        pda.next_state(token)
        token_ids.append(token)
    raise AssertionError(f"walk did not terminate within {max_steps} steps")


@pytest.mark.integration
@pytest.mark.parametrize("model", REPRESENTATIVE_MODELS)
@pytest.mark.parametrize("seed", range(10))
def test_random_walk_round_trip(model, tokenizer, seed):
    pars_table, map_tt = built_grammar(model, tokenizer)
    text = random_walk_text(pars_table, map_tt, tokenizer, seed)
    data = json.loads(text)            # valid JSON
    model.model_validate(data)         # valid instance of the model
```

- [ ] **Step 2: Run it**

Run: `uv run --with pytest pytest grammarllm/tests/test_pydantic_json.py -q -m integration -k random_walk`
Expected: `40 passed` (4 models × 10 seeds). Failure modes and their meaning:
- `json.loads` fails → a skeleton chunk or value emitter produces malformed JSON; inspect `text`.
- `model_validate` fails → grammar admits a value outside the model (e.g. wrong enum).
- "walk did not terminate" → for `Node`, random choice recursed too deep; if a specific seed loops, it is legitimate randomness — bias the walk by preferring `null`/`]` token choices at step > 200 rather than raising `max_steps` blindly:

```python
        if len(token_ids) > 200:
            closers = [t for t in choices
                       if tokenizer.decode([t]).lstrip().startswith(("n", "]"))]
            if closers:
                choices = closers
```

- [ ] **Step 3: Write the E2E smoke test**

Append to `test_pydantic_json.py`:

```python
# ── Layer 4: E2E smoke (loads model weights — slow) ──────────────────────────

@pytest.mark.e2e
def test_generation_round_trips():
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model_lm = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct"
    )
    from grammarllm.generate_with_constraints import (
        generate_grammar_parameters, generate_text,
    )

    pars_table, map_tt = built_grammar(Person, tok)
    pdas, streamer = generate_grammar_parameters(tok, pars_table, map_tt)
    result = generate_text(
        model_lm, tok,
        "Describe a person as JSON. Output:",
        pdas, streamer,
        max_new_tokens=60,
        num_beams=2,
    )
    assert result["pda_stack"] == []
    data = json.loads(result["text"])
    Person.model_validate(data)
```

- [ ] **Step 4: Run it**

Run: `uv run --with pytest pytest grammarllm/tests/test_pydantic_json.py -q -m e2e`
Expected: `1 passed` (takes ~1–2 min on CPU).

- [ ] **Step 5: Commit (only if translator fixes were needed)**

```bash
git diff --quiet grammarllm/utils/pydantic_to_grammar.py || {
  git add grammarllm/utils/pydantic_to_grammar.py
  git commit -m "fix(pydantic): issues surfaced by round-trip walks"
}
```

---

### Task 9: Retire old tests, update module docstring and docs

**Files:**
- Delete: `grammarllm/tests/test_pydantic_to_grammar.py` (local-only file; its still-valid rejection tests are re-homed below)
- Modify: `grammarllm/utils/pydantic_to_grammar.py` (module docstring), `docs/usage.md` (pydantic section), `README.md` (pydantic section + limitations line)
- Test: `grammarllm/tests/test_pydantic_json.py` (append re-homed rejection tests)

**Interfaces:**
- Consumes: everything above.
- Produces: single source of truth for pydantic tests; docs describing the shipped API.

- [ ] **Step 1: Re-home the still-valid rejection tests**

Append to `test_pydantic_json.py`:

```python
class TestRejectedConstructs:
    def test_additional_properties_true_blocked(self):
        from pydantic import ConfigDict

        class Open(BaseModel):
            model_config = ConfigDict(extra="allow")
            x: Literal["a"]

        with pytest.raises(PydanticGrammarError, match="additionalProperties"):
            pydantic_to_productions(Open)

    def test_untyped_list_blocked(self):
        class L(BaseModel):
            items: list

        with pytest.raises(PydanticGrammarError, match="items"):
            pydantic_to_productions(L)

    def test_non_string_enum_blocked(self):
        class E(BaseModel):
            v: Literal[1, 2]

        with pytest.raises(PydanticGrammarError):
            pydantic_to_productions(E)

    def test_same_type_union_blocked(self):
        class U(BaseModel):
            v: Literal["a"] | Literal["b"]   # two enum branches

        with pytest.raises(PydanticGrammarError):
            pydantic_to_productions(U)

    def test_instance_instead_of_class_raises_typeerror(self):
        class M(BaseModel):
            v: Literal["a"]

        with pytest.raises(TypeError):
            pydantic_to_productions(M(v="a"))
```

Then delete the old file:

```bash
rm grammarllm/tests/test_pydantic_to_grammar.py
```

- [ ] **Step 2: Run the full local suite**

Run: `uv run --with pytest pytest grammarllm/tests/ -q -m "not e2e"`
Expected: all pass (unit + integration layers + existing `test_dev_review_fixes.py`, `test_logit_processor_beam.py`, `test_multi_tag.py`).

- [ ] **Step 3: Update the module docstring**

Replace the header docstring of `grammarllm/utils/pydantic_to_grammar.py` so the pipeline diagram and the supported/rejected lists match the new behavior. Replace the two list sections with:

```
Supported JSON Schema constructs
─────────────────────────────────
✅  object with fixed fields          → {"field": value, ...} skeleton chunks
✅  enum / Literal (string values)    → <<"value">> alternatives
✅  Optional[X]                       → value alternates X | null (key always present)
✅  anyOf / oneOf of distinct types   → FIRST-disjoint alternatives
✅  array (homogeneous)               → [item, item, ...], empty allowed
✅  allOf (non-overlapping fields)    → flattened object
✅  $ref                              → named shared NT; cycles allowed when
                                        broken by an Optional or array edge
✅  str / int / float / bool          → shared JSON_* NTs over two regex
                                        terminals (json_char, digit)

Rejected constructs (PydanticGrammarError in Phase 1)
─────────────────────────────────────────────────────
❌  $ref cycles with no Optional/array edge (generation could never end)
❌  int | float unions (both start with a digit — FIRST conflict)
❌  anyOf / oneOf with two branches of the same JSON type
❌  if / then / else, patternProperties, not, contains
❌  additionalProperties: true
❌  non-string enum values; enum values or field names containing " \\ or
    control characters (v1 emits no escape sequences)
❌  array without items
```

- [ ] **Step 4: Update docs**

In `docs/usage.md`, replace the "Pydantic models (experimental)" section body with:

```markdown
## Pydantic models

`pydantic_to_productions(Model)` converts a Pydantic v2 model into a grammar
whose every output is **strict JSON**, round-trippable into the model:

    from typing import Literal, Optional
    from pydantic import BaseModel
    from grammarllm.utils.pydantic_to_grammar import pydantic_to_productions

    class Person(BaseModel):
        name: Literal["mario", "luisa"]
        mood: Optional[Literal["happy", "sad"]] = None

    productions, regex_dict = pydantic_to_productions(Person)
    pars_table, map_tt = get_parsing_table_and_map_tt(tokenizer, productions, regex_dict)
    # generated text: {"name": "mario", "mood": null}
    # json.loads(...) and Person.model_validate_json(...) always succeed

Rules and limits:

- Keys are always emitted; `Optional` fields produce `null` values, they are
  never omitted.
- Recursive models are supported when the cycle passes through an `Optional`
  or `list` field (`Node.next: Optional[Node]`); unbreakable cycles are
  rejected at conversion time.
- No JSON escape sequences: string content, enum values, and field names must
  not contain `"`, `\`, or control characters.
- `int | float` unions are rejected (indistinguishable with one token of
  lookahead) — use a single `float` field.

Unsupported constructs raise `PydanticGrammarError` at conversion time with a
message naming the field and the fix.
```

In `README.md`, replace the pydantic section body (keep the heading) with:

```markdown
```python
from typing import Literal, Optional
from pydantic import BaseModel
from grammarllm.utils.pydantic_to_grammar import pydantic_to_productions

class Person(BaseModel):
    name: Literal["mario", "luisa"]
    mood: Optional[Literal["happy", "sad"]] = None

productions, regex_dict = pydantic_to_productions(Person)
pars_table, map_tt = get_parsing_table_and_map_tt(tokenizer, productions, regex_dict)
# generated text is always valid JSON: {"name": "mario", "mood": null}
```

Output is strict JSON — `json.loads` and `Person.model_validate_json` always
succeed. Design: [docs/superpowers/specs/2026-07-07-pydantic-json-grammar-design.md](docs/superpowers/specs/2026-07-07-pydantic-json-grammar-design.md).
```

and update the README limitations bullet
`* Pydantic conversion is experimental and currently emits a compact non-JSON format (see spec above)` to
`* Pydantic conversion emits strict JSON; no escape sequences in string content (no ", \ or control chars)`.

Also remove the now-stale "(experimental)" wording in the README features list bullet: `* 🧬 **Pydantic → grammar conversion** — derive a strict-JSON grammar from a `BaseModel``.

- [ ] **Step 5: Final verification**

Run: `uv run --with pytest pytest grammarllm/tests/ -q -m "not e2e"` → all pass.
Run: `uv run --with pytest pytest grammarllm/tests/test_pydantic_json.py -q -m e2e` → 1 passed.

- [ ] **Step 6: Commit**

```bash
git add grammarllm/utils/pydantic_to_grammar.py docs/usage.md README.md
git commit -m "feat(pydantic): ship strict-JSON conversion — docs + docstring

Implements docs/superpowers/specs/2026-07-07-pydantic-json-grammar-design.md"
```

---

## Self-Review Notes

- **Spec coverage:** D1 (Task 1–5 emission), D2 (Task 1 API), D3 (Task 3, asserted by `test_optional_enum`), D4 (Task 1 skeleton chunks), D5 (Tasks 1–2, quotes in enum tags / `JSON_STRING`), D6 (Task 6), D7 (Task 3). Default regexes (Task 1). Testing layers 1–4 (Tasks 1–6 / 7 / 8 / 8). Docs (Task 9). Non-goals respected: no `_OPT` wrappers, no escape sequences (guarded by `_UNSAFE_LITERAL`), no minItems.
- **Type consistency:** `_value_symbol(parent_nt, slot_name, schema) -> str` used identically in Tasks 1–6; shared NT names (`JSON_STRING`, `JSON_INT`, …) match between Task 2 definitions and Tasks 3–5 assertions; `built_grammar`/`REPRESENTATIVE_MODELS` defined in Task 7, consumed in Task 8.
- **Known judgment call:** old `test_pydantic_to_grammar.py` breaks after Task 1 and is deleted in Task 9; interim runs must target `test_pydantic_json.py` only (called out in Task 1 Step 4).
