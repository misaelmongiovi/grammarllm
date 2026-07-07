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
#
# These are the terminal *names* that will appear in the productions dict.
# The caller must supply matching entries in regex_dict when calling
# get_parsing_table_and_map_tt, e.g.:
#
#   regex_dict = {
#       'regex_integer_token': re.compile(r'\d+'),
#       'regex_number_token':  re.compile(r'\d+([.,]\d+)?'),
#       'regex_string_token':  re.compile(r'[A-Za-z0-9_]+'),
#   }
#
# You can override the mapping via the `type_terminal_map` argument of
# pydantic_to_productions().
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TYPE_TERMINAL_MAP: dict[str, str] = {
    "integer": "integer_token",
    "number":  "number_token",
    "string":  "string_token",
    "boolean": "boolean_token",   # 'true' | 'false' — caller supplies regex
}


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
    Translates a validated JSON Schema into a GrammarLLM productions dict.

    NT naming convention:
        - Top-level model     →  'S*'  (reserved start symbol, always)
        - $defs model         →  _nt_name(def_name)   e.g. 'ADDRESS'
        - array tail NT       →  '<NT>_LIST'           e.g. 'TAG_LIST'
        - allOf flattened     →  same as object
        - anyOf               →  inline alternatives on the parent NT
    """

    def __init__(
        self,
        defs: dict[str, Any],
        type_terminal_map: dict[str, str],
    ) -> None:
        self._defs = defs
        self._type_map = type_terminal_map
        # Accumulated productions dict  {NT_name: [production_string, ...]}
        self._productions: dict[str, list[str]] = {}
        # Track which $defs NTs have already been emitted (avoid duplicates)
        self._emitted: set[str] = set()

    # ── Public entry point ───────────────────────────────────────────────────

    def translate(self, root_schema: dict[str, Any]) -> dict[str, list[str]]:
        """
        Translate *root_schema* (the top-level model schema) and return the
        productions dict ready for ProductionRuleProcessor.
        """
        self._emit_nt("S*", root_schema)
        return self._productions

    # ── NT emission ──────────────────────────────────────────────────────────

    def _emit_nt(self, nt: str, schema: dict[str, Any]) -> None:
        """
        Generate all production rules for *nt* from *schema* and add them to
        self._productions.  Recursively emits referenced NTs.
        """
        if nt in self._emitted:
            return
        self._emitted.add(nt)

        schema = self._resolve(schema)

        # ── object ───────────────────────────────────────────────────────────
        if schema.get("type") == "object" or "properties" in schema:
            self._emit_object(nt, schema)
            return

        # ── allOf → flatten into object ──────────────────────────────────────
        if "allOf" in schema:
            flat = self._flatten_all_of(schema["allOf"])
            self._emit_object(nt, flat)
            return

        # ── anyOf / oneOf ────────────────────────────────────────────────────
        if "anyOf" in schema or "oneOf" in schema:
            self._emit_any_of(nt, schema)
            return

        # ── enum ─────────────────────────────────────────────────────────────
        if "enum" in schema:
            self._emit_enum(nt, schema["enum"])
            return

        # ── array ─────────────────────────────────────────────────────────────
        if schema.get("type") == "array":
            self._emit_array(nt, schema)
            return

        # ── primitives ────────────────────────────────────────────────────────
        primitive = schema.get("type")
        if primitive in self._type_map:
            terminal = self._type_map[primitive]
            self._productions[nt] = [terminal]
            return

        raise PydanticGrammarError(
            f"Cannot translate schema to NT '{nt}': unrecognised shape {schema}"
        )

    # ── Object translation ───────────────────────────────────────────────────

    def _emit_object(self, nt: str, schema: dict[str, Any]) -> None:
        """
        object { "a": A, "b": B }
        →  NT: ["<<a>> A_NT <<b>> B_NT"]

        Fields are emitted in schema-definition order (deterministic).
        Each field value becomes its own NT so the grammar stays LL(1).

        BUG FIX (double epsilon): a non-required Optional[X] field has BOTH
        signals — it is absent from 'required' AND its schema is an
        anyOf [X, null].  _emit_any_of already appends an ε alternative for
        the null branch; wrapping the field in a second `_OPT → NT | ε`
        layer made ε derivable twice under the same FOLLOW set, which the
        LL(1) table builder rejects ("Conflict: ... $ []").  The _OPT
        wrapper is now added only when the field schema itself is NOT
        already nullable.
        """
        props = schema.get("properties", {})
        required = set(schema.get("required", props.keys()))
        parts: list[str] = []

        for field_name, field_schema in props.items():
            field_nt = f"{nt}_{field_name.upper()}"
            # Literal key as exact terminal
            parts.append(f"<<{field_name}>>")
            resolved = self._resolve(field_schema)
            if field_name in required or _is_nullable(resolved):
                # Required field, or Optional[X] whose anyOf-null branch
                # already gives the NT its own ε alternative.
                parts.append(field_nt)
                self._emit_nt(field_nt, resolved)
            else:
                # Non-required, non-nullable (e.g. field with a default):
                # wrap in an epsilon alternative.
                opt_nt = f"{field_nt}_OPT"
                parts.append(opt_nt)
                self._emit_nt(field_nt, resolved)
                self._productions[opt_nt] = [field_nt, "ε"]

        self._productions[nt] = [" ".join(parts)]

    # ── allOf flattening ─────────────────────────────────────────────────────

    def _flatten_all_of(self, branches: list[dict]) -> dict[str, Any]:
        """Merge all allOf branches into a single synthetic object schema."""
        merged_props: dict[str, Any] = {}
        merged_required: list[str] = []
        for branch in branches:
            resolved = self._resolve(branch)
            merged_props.update(resolved.get("properties", {}))
            merged_required.extend(resolved.get("required", []))
        return {
            "type": "object",
            "properties": merged_props,
            "required": list(dict.fromkeys(merged_required)),  # dedup, preserve order
        }

    # ── anyOf / oneOf translation ────────────────────────────────────────────

    def _emit_any_of(self, nt: str, schema: dict[str, Any]) -> None:
        """
        anyOf / oneOf  →  one production alternative per non-null branch.
        Optional[X]    →  [X_NT, "ε"]
        """
        branches = schema.get("anyOf", schema.get("oneOf", []))
        non_null = [b for b in branches if b.get("type") != "null"]
        nullable = len(non_null) < len(branches)

        alternatives: list[str] = []
        for idx, branch in enumerate(non_null):
            branch_nt = f"{nt}_BRANCH{idx}"
            self._emit_nt(branch_nt, branch)
            alternatives.append(branch_nt)

        if nullable:
            alternatives.append("ε")

        self._productions[nt] = alternatives

    # ── enum translation ──────────────────────────────────────────────────────

    def _emit_enum(self, nt: str, values: list[Any]) -> None:
        """enum ["a", "b", "c"]  →  NT: ["<<a>>", "<<b>>", "<<c>>"]"""
        alts: list[str] = []
        for v in values:
            if not isinstance(v, str):
                raise PydanticGrammarError(
                    f"NT '{nt}': enum value {v!r} is not a string. "
                    "GrammarLLM can only generate string tokens. "
                    "Use Literal['a', 'b', 'c'] with string values."
                )
            alts.append(f"<<{v}>>")
        self._productions[nt] = alts

    # ── array translation ─────────────────────────────────────────────────────

    def _emit_array(self, nt: str, schema: dict[str, Any]) -> None:
        """
        array of ITEM  →  NT:      ["ITEM_NT NT_LIST"]
                          NT_LIST: ["ITEM_NT NT_LIST", "ε"]

        This is a right-recursive (tail-recursive) rule, which is LL(1).
        """
        item_nt = f"{nt}_ITEM"
        list_nt = f"{nt}_LIST"
        self._emit_nt(item_nt, schema["items"])
        self._productions[nt] = [f"{item_nt} {list_nt}"]
        self._productions[list_nt] = [f"{item_nt} {list_nt}", "ε"]

    # ── helpers ───────────────────────────────────────────────────────────────

    def _resolve(self, schema: dict[str, Any]) -> dict[str, Any]:
        """
        Follow a $ref if present, otherwise return the schema as-is.

        BUG FIX: this used to ALSO emit a named NT for the def (e.g. ADDRESS)
        and return a synthetic schema with a dead '_nt_ref' key that no caller
        read.  The result was two parallel NT trees per $ref — the named one
        (unreachable from S*) and the inlined per-field one — polluting the
        grammar and the parsing table.  Refs are now purely inlined; cycle
        safety is guaranteed because the validator already rejects cyclic
        $ref chains in Phase 1.
        """
        if "$ref" in schema:
            name = schema["$ref"][len("#/$defs/"):]
            return self._defs[name]
        return schema


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def pydantic_to_productions(
    model: type,
    type_terminal_map: dict[str, str] | None = None,
) -> dict[str, list[str]]:
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
    >>> productions = pydantic_to_productions(Sentiment)
    >>> # productions == {
    >>> #     'S*': ['<<label>> S*_LABEL'],
    >>> #     'S*_LABEL': ['<<positive>>', '<<negative>>', '<<neutral>>'],
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

    # ── Phase 2: translation ─────────────────────────────────────────────────
    translator = _Translator(defs=defs, type_terminal_map=ttmap)
    productions = translator.translate(schema)

    return productions
