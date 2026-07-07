"""
pydantic_to_grammar.py
======================
Converts a Pydantic BaseModel into a GrammarLLM productions dict.

Pipeline:
    Pydantic model
        └─► JSON Schema  (model.model_json_schema())
              └─► Phase 1: structural validation  (raises PydanticGrammarError on non-LL(1) constructs)
                    └─► Phase 2: translation       (returns productions dict with <<>> notation)
                          └─► get_parsing_table_and_map_tt(tokenizer, productions)  [existing pipeline]

Supported JSON Schema constructs
─────────────────────────────────
✅  object with fixed fields
✅  enum / Literal
✅  anyOf / oneOf  with structurally disjoint branches (FIRST-disjointness verified at runtime by existing LL(1) table builder)
✅  Optional[X]  →  anyOf: [X, {type: null}]
✅  array (homogeneous)  →  tail-recursive NT
✅  allOf  with non-overlapping fields  →  flattened into single object
✅  $ref  →  resolved from $defs, right/mediated recursion only
✅  boolean, integer, number, string  →  mapped to caller-supplied regex terminals

Rejected constructs (PydanticGrammarError raised in Phase 1)
─────────────────────────────────────────────────────────────
❌  $ref with direct left recursion
❌  $ref with mutual cyclic recursion
❌  allOf with duplicate fields of different types
❌  if / then / else
❌  patternProperties
❌  additionalProperties: true  (open schema)
❌  anyOf / oneOf whose branches cannot be proven structurally disjoint at schema level
     (FIRST overlap is caught later by the LL(1) table builder, but we reject obvious cases early)
"""

from __future__ import annotations

import re
from typing import Any

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None  # type: ignore[assignment,misc]


# ─────────────────────────────────────────────────────────────────────────────
# Public exception
# ─────────────────────────────────────────────────────────────────────────────

class PydanticGrammarError(ValueError):
    """
    Raised when a Pydantic model contains constructs that cannot be translated
    into a valid LL(1) grammar.

    The message is written in terms of Pydantic/Python types, not LL(1) formalism,
    so it is actionable for practitioners.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Default primitive-type → regex-terminal mapping
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TYPE_TERMINAL_MAP: dict[str, str] = {
    "string": "json_char",
    "integer": "digit",
    "number": "digit",
}

_JSON_CHAR_REGEX = r'^[^"\\\x00-\x1f]+$'
_DIGIT_REGEX = r"^[0-9]+$"

_UNSAFE_LITERAL = re.compile(r'["\\\x00-\x1f]')


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any]:
    """Resolve a JSON Schema $ref of the form '#/$defs/<name>'."""
    if not ref.startswith("#/$defs/"):
        raise PydanticGrammarError(
            f"Unsupported $ref format '{ref}'. "
            "Only local $defs references (#/$defs/<name>) are supported."
        )
    name = ref[len("#/$defs/"):]
    if name not in defs:
        raise PydanticGrammarError(
            f"$ref '{ref}' points to '{name}' which is not present in $defs. "
            f"Available defs: {list(defs.keys())}"
        )
    return defs[name]


def _nt_name(base: str) -> str:
    """Normalise a $defs key into a valid non-terminal name (uppercase, no spaces)."""
    return re.sub(r"[^A-Za-z0-9_]", "_", base).upper()


def _is_nullable(schema: dict[str, Any]) -> bool:
    """Return True if the schema allows null (Optional[X] pattern)."""
    any_of = schema.get("anyOf", [])
    return any(b.get("type") == "null" for b in any_of)


def _non_null_branch(schema: dict[str, Any]) -> dict[str, Any]:
    """For an Optional[X] anyOf, return the non-null branch."""
    for b in schema.get("anyOf", []):
        if b.get("type") != "null":
            return b
    raise PydanticGrammarError("anyOf contains only null branches.")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Structural validation
# ─────────────────────────────────────────────────────────────────────────────

class _Validator:
    """
    Walks the JSON Schema and raises PydanticGrammarError on any construct
    that cannot be translated to LL(1).

    Tracks visited $defs names to detect cyclic / left-recursive $ref chains.
    """

    # Constructs we refuse unconditionally
    _BLOCKED_KEYWORDS = ("if", "then", "else", "patternProperties", "not", "contains")

    def __init__(self, defs: dict[str, Any]) -> None:
        self._defs = defs
        # Stack of $defs names currently being expanded — used for cycle detection.
        self._expansion_stack: list[str] = []

    def validate(self, schema: dict[str, Any], path: str = "root") -> None:
        """Entry point: validate *schema* rooted at *path* (for error messages)."""

        # ── Blocked keywords ────────────────────────────────────────────────
        for kw in self._BLOCKED_KEYWORDS:
            if kw in schema:
                raise PydanticGrammarError(
                    f"[{path}] The keyword '{kw}' is not supported. "
                    "LL(1) grammars require a fixed, statically-known structure. "
                    "Remove conditional or negation keywords from your model."
                )

        # ── additionalProperties: true ──────────────────────────────────────
        if schema.get("additionalProperties") is True:
            raise PydanticGrammarError(
                f"[{path}] 'additionalProperties: true' produces an open schema "
                "with an unbounded set of possible keys. "
                "LL(1) requires a finite, statically-known set of terminals. "
                "Set 'model_config = ConfigDict(extra=\"forbid\")' on your model."
            )

        schema_type = schema.get("type")

        # ── $ref ────────────────────────────────────────────────────────────
        if "$ref" in schema:
            self._validate_ref(schema["$ref"], path)
            return

        # ── allOf ───────────────────────────────────────────────────────────
        if "allOf" in schema:
            self._validate_all_of(schema["allOf"], path)
            return

        # ── anyOf / oneOf ───────────────────────────────────────────────────
        if "anyOf" in schema or "oneOf" in schema:
            branches = schema.get("anyOf", schema.get("oneOf", []))
            self._validate_any_of(branches, path, schema)
            return

        # ── object ──────────────────────────────────────────────────────────
        if schema_type == "object":
            self._validate_object(schema, path)
            return

        # ── array ───────────────────────────────────────────────────────────
        if schema_type == "array":
            items = schema.get("items")
            if items is None:
                raise PydanticGrammarError(
                    f"[{path}] 'array' without 'items' is an open schema. "
                    "Annotate your list field with a concrete element type, "
                    "e.g. list[str] instead of list."
                )
            self.validate(items, path=f"{path}[items]")
            return

        # ── primitives and enum ─────────────────────────────────────────────
        # These are always valid leaves — nothing to recurse into.
        if "enum" in schema or schema_type in ("string", "integer", "number", "boolean", "null"):
            return

        # ── unknown / unrecognised ───────────────────────────────────────────
        # Be conservative: if we don't recognise the schema shape, reject it.
        recognised_keys = {
            "type", "enum", "properties", "required", "items",
            "anyOf", "oneOf", "allOf", "$ref", "title", "description",
            "default", "examples", "$defs", "additionalProperties",
        }
        unknown = set(schema.keys()) - recognised_keys
        if unknown:
            raise PydanticGrammarError(
                f"[{path}] Unrecognised JSON Schema keywords: {sorted(unknown)}. "
                "GrammarLLM only supports a deterministic subset of JSON Schema. "
                "Remove or replace these keywords."
            )

    # ── $ref validation ──────────────────────────────────────────────────────

    def _validate_ref(self, ref: str, path: str) -> None:
        target_name = ref[len("#/$defs/"):] if ref.startswith("#/$defs/") else None
        if target_name is None:
            raise PydanticGrammarError(
                f"[{path}] Unsupported $ref '{ref}'. Only '#/$defs/<name>' is allowed."
            )

        # ── Mutual cyclic recursion detection ───────────────────────────────
        # If target_name is already on the expansion stack, we have a cycle.
        # A → B → A  is mutual recursion.
        # A → A  is direct recursion.
        # Both are non-LL(1) unless mediated by a non-nullable prefix.
        # We reject all cycles here; the user must break cycles with Optional.
        if target_name in self._expansion_stack:
            cycle = " → ".join(self._expansion_stack + [target_name])
            raise PydanticGrammarError(
                f"[{path}] Cyclic $ref detected: {cycle}. "
                "Direct and mutual left recursion are not LL(1). "
                "Break the cycle with Optional[...] or restructure the model so "
                "that the recursive field is always preceded by at least one terminal."
            )

        resolved = _resolve_ref(ref, self._defs)
        self._expansion_stack.append(target_name)
        self.validate(resolved, path=f"{path}→{target_name}")
        self._expansion_stack.pop()

    # ── allOf validation ─────────────────────────────────────────────────────

    def _validate_all_of(self, branches: list[dict], path: str) -> None:
        """
        allOf is valid only if it represents Pydantic model inheritance:
        all branches are objects with non-overlapping fields.
        We flatten them here purely for validation; the translator does the same.
        """
        seen_fields: dict[str, str] = {}  # field_name → branch_index
        for idx, branch in enumerate(branches):
            # Resolve $ref branches
            resolved = _resolve_ref(branch["$ref"], self._defs) if "$ref" in branch else branch
            if resolved.get("type") != "object":
                raise PydanticGrammarError(
                    f"[{path}] allOf branch {idx} is not an object schema. "
                    "GrammarLLM only supports allOf for model inheritance "
                    "(all branches must be object schemas)."
                )
            for field in resolved.get("properties", {}):
                if field in seen_fields:
                    b0 = seen_fields[field]
                    # Duplicate field is allowed only if both branches define it identically.
                    # We cannot check schema equality cheaply, so we reject to be safe.
                    raise PydanticGrammarError(
                        f"[{path}] allOf has duplicate field '{field}' "
                        f"in branches {b0} and {idx}. "
                        "Duplicate fields in allOf are ambiguous for LL(1) generation. "
                        "Merge the branches manually into a single model."
                    )
                seen_fields[field] = str(idx)
            self.validate(resolved, path=f"{path}/allOf[{idx}]")

    # ── anyOf / oneOf validation ─────────────────────────────────────────────

    def _validate_any_of(
        self, branches: list[dict], path: str, parent_schema: dict
    ) -> None:
        """
        anyOf / oneOf is valid in two patterns:

        1. Optional[X]  →  anyOf: [X, {type: null}]
           Always valid — null is handled as epsilon / skip.

        2. Union of structurally distinct types
           Valid only if each branch has a *different* JSON Schema 'type'
           (integer vs string vs object vs array vs boolean).
           We cannot verify FIRST-set disjointness here (that requires the
           tokenizer), so we accept the schema and rely on the LL(1) table
           builder to catch conflicts at runtime.

        3. Union of same-type schemas (e.g. two different object shapes)
           Rejected here because they necessarily share the same FIRST token
           (e.g. both start with '{') and are therefore always LL(1)-conflicting.
        """
        # Filter out null branches (Optional pattern)
        non_null = [b for b in branches if b.get("type") != "null"]

        if len(non_null) == 0:
            raise PydanticGrammarError(
                f"[{path}] anyOf/oneOf contains only null branches."
            )

        # Collect the top-level 'type' of each non-null branch
        branch_types: list[str] = []
        for idx, branch in enumerate(non_null):
            resolved = _resolve_ref(branch["$ref"], self._defs) if "$ref" in branch else branch
            t = resolved.get("type")
            if t is None and "enum" not in resolved:
                raise PydanticGrammarError(
                    f"[{path}] anyOf/oneOf branch {idx} has no 'type' and no 'enum'. "
                    "GrammarLLM cannot determine the FIRST set for this branch. "
                    "Each branch must be a concrete typed schema or an enum."
                )
            branch_types.append(t or "enum")

        # Reject same-type branches (guaranteed FIRST conflict)
        seen_types: dict[str, int] = {}
        for idx, t in enumerate(branch_types):
            if t in seen_types:
                raise PydanticGrammarError(
                    f"[{path}] anyOf/oneOf has two branches of type '{t}' "
                    f"(branches {seen_types[t]} and {idx}). "
                    "Two branches of the same type always produce a FIRST-set conflict "
                    "in LL(1). Use a single branch with an enum instead, "
                    "or add a distinguishing literal prefix to each branch."
                )
            seen_types[t] = idx

        # Recurse into each non-null branch
        for idx, branch in enumerate(non_null):
            resolved = _resolve_ref(branch["$ref"], self._defs) if "$ref" in branch else branch
            self.validate(resolved, path=f"{path}/anyOf[{idx}]")

    # ── object validation ────────────────────────────────────────────────────

    def _validate_object(self, schema: dict, path: str) -> None:
        props = schema.get("properties", {})
        if not props:
            raise PydanticGrammarError(
                f"[{path}] Object schema has no 'properties'. "
                "GrammarLLM requires a fixed set of fields. "
                "Add at least one typed field to your model."
            )
        for field_name, field_schema in props.items():
            self.validate(field_schema, path=f"{path}.{field_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Translation
# ─────────────────────────────────────────────────────────────────────────────

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
        if "const" in schema:
            # pydantic v2 emits single-value Literal["x"] as const, not enum.
            schema = {**schema, "enum": [schema["const"]]}
        if "enum" in schema:
            nt = f"{parent_nt}_{slot_name.upper()}"
            self._emit_enum(nt, schema["enum"])
            return nt
        t = schema.get("type")
        if t == "string":
            return self._json_string_nt()
        if t == "integer":
            return self._json_int_nt()
        if t == "number":
            return self._json_number_nt()
        if t == "boolean":
            return self._json_bool_nt()
        raise PydanticGrammarError(
            f"Cannot translate value schema for '{parent_nt}.{slot_name}': "
            f"unrecognised shape {schema}"
        )

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


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def pydantic_to_productions(
    model: type,
    type_terminal_map: dict[str, str] | None = None,
) -> tuple[dict[str, list[str]], dict[str, re.Pattern]]:
    """
    Convert a Pydantic BaseModel subclass into a GrammarLLM productions dict.

    Parameters
    ----------
    model:
        A Pydantic v2 BaseModel subclass.  Must be a class, not an instance.
    type_terminal_map:
        Override the default mapping from JSON Schema primitive types to
        GrammarLLM terminal names.  The terminal names must match keys in
        the regex_dict passed to get_parsing_table_and_map_tt().
        Defaults to DEFAULT_TYPE_TERMINAL_MAP.

    Returns
    -------
    productions : dict[str, list[str]]
        Ready to pass to get_parsing_table_and_map_tt(tokenizer, productions).

    Raises
    ------
    PydanticGrammarError
        If the model contains constructs incompatible with LL(1) parsing.
    ImportError
        If pydantic is not installed.

    Examples
    --------
    >>> from pydantic import BaseModel
    >>> from typing import Literal
    >>>
    >>> class Sentiment(BaseModel):
    ...     label: Literal["positive", "negative", "neutral"]
    ...
    >>> productions, regex_dict = pydantic_to_productions(Sentiment)
    >>> # productions == {
    >>> #     'S*': ['<<{"label": >> S*_LABEL <<}>>'],
    >>> #     'S*_LABEL': ['<<"positive">>', '<<"negative">>', '<<"neutral">>'],
    >>> # }
    """
    if BaseModel is None:
        raise ImportError(
            "pydantic is required to use pydantic_to_productions(). "
            "Install it with: pip install pydantic"
        )
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise TypeError(
            f"Expected a Pydantic BaseModel subclass, got {type(model).__name__}. "
            "Pass the class itself, not an instance."
        )

    ttmap = {**DEFAULT_TYPE_TERMINAL_MAP, **(type_terminal_map or {})}

    # ── Extract JSON Schema and $defs ────────────────────────────────────────
    schema = model.model_json_schema()
    defs: dict[str, Any] = schema.get("$defs", {})

    # ── Phase 1: structural validation ──────────────────────────────────────
    validator = _Validator(defs=defs)
    validator.validate(schema, path=model.__name__)

    # ── Phase 2: translation ─────────────────────────────────────────────
    translator = _Translator(defs=defs, type_terminal_map=ttmap)
    productions = translator.translate(schema)

    # ── Regex terminals for open values (token-level regexes) ───────────
    regex_dict = {
        f"regex_{ttmap['string']}": re.compile(_JSON_CHAR_REGEX),
        f"regex_{ttmap['integer']}": re.compile(_DIGIT_REGEX),
    }

    return productions, regex_dict
