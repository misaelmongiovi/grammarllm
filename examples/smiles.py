"""
OpenSMILES grammar as an LL(prefix) grammar for grammarllm.

Source: OpenSMILES specification, section 2.2 "Grammar"
(http://opensmiles.org/opensmiles.html#_grammar).

Coverage: the whole of section 2.2 — the 114 IUPAC element symbols, the 8
aromatic symbols, the 10 organic-subset atoms, bracket-atom optional fields
(isotope, chirality, hcount, charge, atom class), branches, dot
disconnections, all seven bond orders including the quadruple bond `$`, and
ring closures in all four spec forms (bare `1`, two-digit `%12`, and both of
those preceded by an explicit bond order, `=1` / `=%12`).


Why this file is not a rule-for-rule transcription
---------------------------------------------------
The spec's EBNF is not LL(1), and grammarllm needs LL(1) after tag
expansion. The sticking point is the ring closure:

    ringbond      ::= bond? DIGIT | bond? '%' DIGIT DIGIT
    branched_atom ::= atom ringbond* branch*
    chain         ::= ... | chain bond branched_atom | ...

Right after an atom, a bond symbol is a valid lookahead for two different
things at once: the start of a ring closure (`C=1...`) and the bond joining
the next chain atom (`C=C`). Only the token AFTER the bond tells them apart
(digit vs atom), so one token of lookahead is not enough:

    Conflict: RINGBONDS -> - ['RINGBOND', 'RINGBONDS']!
    Regola attuale: - ['ε']!

This is a FIRST/FOLLOW conflict — a nullable non-terminal whose FOLLOW set
meets its own FIRST set — and the two competing alternatives live in
different non-terminals, so no amount of left-factoring reaches them.

Writing the spec's rule verbatim now works anyway: parsing_table() detects
the conflict and rewrites the grammar automatically (see the "Riscrittura
automatica in forma LL(1)" section of generate_LL1_parsing_table.py). This
file nonetheless keeps the resolution written out by hand, for two reasons:
the generated non-terminal names (`RINGBONDS_NT__C__C2`) are far harder to
read in `temp/table_parsing.json` when debugging a grammar, and the
automatic pass is a bounded semi-algorithm — no algorithm can LL(1)-ify
every grammar, since not every deterministic language has an LL(1) grammar
at all — so an explicit grammar is one less thing that can fail to
converge.

The resolution is to defer the decision by one grammar level rather than
resolve it with lookahead: TAIL_NT consumes the bond and hands over to
AFTER_BOND_NT, which dispatches on the *next* token —

    TAIL_NT       -> ... | BOND_NT AFTER_BOND_NT | ...
    AFTER_BOND_NT -> digit TAIL_NT                 # it was a ring closure
                   | '%' digit digit TAIL_NT       # ... two-digit form
                   | ATOM_NT TAIL_NT               # it was a chain bond

The essential detail is that the chain-bond branch continues at the SAME
stack level (`ATOM_NT TAIL_NT`). An earlier attempt handed off to a nested
`BRANCHED_ATOM` instead, which left the outer atom's branch section
stranded underneath on the stack — two live branch frames both accepting
'(' — a real ambiguity that surfaced as
`Conflict: BRANCHES -> ( ['BRANCH','BRANCHES']! Regola attuale: ( ['ε']!`.

Because the top-level chain ends at EOS while a branch-internal chain ends
at ')', the two need separate (but finite — exactly two) copies of this
machinery: TAIL_NT/AFTER_BOND_NT/... for the former, BTAIL_NT/
BAFTER_BOND_NT/... for the latter.

Per the spec's `atom ringbond* branch*`, ring closures may not follow a
branch; the grammar enforces that (`C(F)1C` is rejected).

One genuine simplification of the spec: chirality keeps only the enumerated
forms (`@TB1`..`@TB20`, `@OH1`..`@OH30`) and drops the generalized
`'@TB' DIGIT DIGIT` spelling the spec also lists, which would let e.g.
"@TB10" derive two different ways. The enumeration spans the spec's full
ranges, so no value is lost.


Two mechanisms worth knowing before editing this file
------------------------------------------------------
*Every non-terminal ends in _NT, and that is load-bearing, not cosmetic.*
SMILES element terminals are single and double capitals ('C', 'N', 'S',
'In', 'No', ...), so a bare non-terminal named 'C' silently aliases the
carbon terminal, quietly corrupting the FIRST sets (it shows up as a
baffling conflict such as `S* -> .` claiming a production that cannot
start with a dot).

*The top-level chain recurses through S* itself.* Termination works only
because get_parsing_table_and_map_tt() appends `S* -> eos_token` and the
PDA's stack must literally empty out for PushdownAutomaton.eos() to become
True. Epsilon reductions never happen spontaneously — the stack is only
popped by consuming a terminal — so a trailing nullable tail is flushed
precisely by the eos_token propagating into its FOLLOW set, which requires
S* to sit structurally at the end of the chain. Route the chain through a
helper non-terminal instead and generation can never legally stop; it runs
to max_new_tokens with a half-finished molecule. (Same pattern as the
README's RDF example, `'S*': ["SUBJ PRED OBJ . S*"]`.)

Not a defect: ring-closure digit *pairing* — the same digit appearing
exactly twice — is context-sensitive, so no CFG can enforce it, the
official EBNF included. "C1CC" (ring opened, never closed) parses here
exactly as it would under the spec's own rules.

Run:  uv run python examples/smiles.py
      GRAMMARLLM_MODEL=/path/to/local/model uv run python examples/smiles.py
"""
import os
import re

from transformers import AutoTokenizer, AutoModelForCausalLM

from grammarllm import (
    get_parsing_table_and_map_tt,
    generate_grammar_parameters,
    generate_text,
    setup_logging,
    create_prompt,
)

MODEL = os.environ.get("GRAMMARLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

BOND_SYMBOLS = ['-', '=', '#', '$', ':', '/', '\\']

ALIPHATIC_ORGANIC_SYMBOLS = ['B', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
AROMATIC_ORGANIC_SYMBOLS = ['b', 'c', 'n', 'o', 's', 'p']
AROMATIC_SYMBOLS = ['b', 'c', 'n', 'o', 'p', 's', 'se', 'as']

ELEMENT_SYMBOLS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Fl', 'Lv',
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
]

CHIRAL_TAGS = (
    ['@', '@@', '@TH1', '@TH2', '@AL1', '@AL2', '@SP1', '@SP2', '@SP3']
    + [f'@TB{i}' for i in range(1, 21)]
    + [f'@OH{i}' for i in range(1, 31)]
)


def _tags(symbols):
    return [f"<<{s}>>" for s in symbols]


def build_smiles_grammar():
    """Return (productions, regex_dict) for the OpenSMILES grammar above."""

    productions = {
        # ============ top-level chain — terminates via S* -> eos_token ======
        'S*': ["ATOM_NT TAIL_NT"],

        # "an atom was just consumed; still in the ring-closure section"
        'TAIL_NT': [
            "digit TAIL_NT",                       # ring closure, e.g. C1
            "<<%>> digit digit TAIL_NT",           # two-digit ring closure, C%12
            "BOND_NT AFTER_BOND_NT",               # ambiguous — defer one level
            "<<(>> BRANCH_BODY_NT <<)>> BRANCH_TAIL_NT",
            "CONT_NT",
        ],
        # "a bond was just consumed right after an atom" — the next token
        # decides what that bond meant, with no lookahead required.
        'AFTER_BOND_NT': [
            "digit TAIL_NT",                       # ...it was a ring closure: C=1
            "<<%>> digit digit TAIL_NT",           # ...two-digit form:         C=%12
            "ATOM_NT TAIL_NT",                     # ...it was a chain bond:    C=C
        ],
        # Past the first branch no ring closure may open (spec: ringbond*
        # branch*), so a bond here is unambiguously a chain bond.
        'BRANCH_TAIL_NT': [
            "<<(>> BRANCH_BODY_NT <<)>> BRANCH_TAIL_NT",
            "CONT_AFTER_BRANCH_NT",
        ],
        'CONT_NT': ["S*", "DOT_NT ATOM_NT TAIL_NT"],
        'CONT_AFTER_BRANCH_NT': [
            "BOND_NT ATOM_NT TAIL_NT",
            "S*",
            "DOT_NT ATOM_NT TAIL_NT",
        ],

        # ============ branch-internal chain — terminates on ')' =============
        # Same machinery, second copy: here the closing ')' plays the role
        # EOS plays at the top level.
        'BRANCH_BODY_NT': [
            "BOND_NT BCHAIN_NT",                   # '(' bond chain ')'
            "DOT_NT BCHAIN_NT",                    # '(' dot chain ')'
            "BCHAIN_NT",                           # '(' chain ')'
        ],
        'BCHAIN_NT': ["ATOM_NT BTAIL_NT"],
        'BTAIL_NT': [
            "digit BTAIL_NT",
            "<<%>> digit digit BTAIL_NT",
            "BOND_NT BAFTER_BOND_NT",
            "<<(>> BRANCH_BODY_NT <<)>> BBRANCH_TAIL_NT",
            "BCONT_NT",
        ],
        'BAFTER_BOND_NT': [
            "digit BTAIL_NT",
            "<<%>> digit digit BTAIL_NT",
            "ATOM_NT BTAIL_NT",
        ],
        'BBRANCH_TAIL_NT': [
            "<<(>> BRANCH_BODY_NT <<)>> BBRANCH_TAIL_NT",
            "BCONT_AFTER_BRANCH_NT",
        ],
        'BCONT_NT': ["BCHAIN_NT", "DOT_NT BCHAIN_NT", "ε"],
        'BCONT_AFTER_BRANCH_NT': [
            "BOND_NT ATOM_NT BTAIL_NT",
            "BCHAIN_NT",
            "DOT_NT BCHAIN_NT",
            "ε",
        ],

        # ============================== atoms ==============================
        'ATOM_NT': ["BRACKET_ATOM_NT", "ALIPHATIC_ORGANIC_NT", "AROMATIC_ORGANIC_NT", "<<*>>"],
        'BRACKET_ATOM_NT': [
            "<<[>> ISOTOPE_OPT_NT SYMBOL_NT CHIRAL_OPT_NT HCOUNT_OPT_NT "
            "CHARGE_OPT_NT CLASS_OPT_NT <<]>>"
        ],
        'ISOTOPE_OPT_NT': ["NUMBER_NT", "ε"],
        'SYMBOL_NT': ["ELEMENT_SYMBOL_NT", "AROMATIC_SYMBOL_NT", "<<*>>"],
        'CHIRAL_OPT_NT': ["CHIRAL_NT", "ε"],
        'HCOUNT_OPT_NT': ["HCOUNT_NT", "ε"],
        'HCOUNT_NT': ["<<H>> HCOUNT_TAIL_NT"],
        'HCOUNT_TAIL_NT': ["digit", "ε"],
        'CHARGE_OPT_NT': ["CHARGE_NT", "ε"],
        'CHARGE_NT': ["<<->> CHARGE_MAG_NT", "<<+>> CHARGE_MAG_NT"],
        'CHARGE_MAG_NT': ["digit CHARGE_MAG2_NT", "ε"],
        'CHARGE_MAG2_NT': ["digit", "ε"],
        'CLASS_OPT_NT': ["CLASS_NT", "ε"],
        'CLASS_NT': ["<<:>> NUMBER_NT"],

        # Isotope mass / atom class: a regex terminal matches one whole
        # vocabulary token, so a multi-digit run needs recursion to span
        # several real subword tokens (same idiom as STRING/alfanum in the
        # README's RDF example).
        'NUMBER_NT': ["digit NUMBER_TAIL_NT"],
        'NUMBER_TAIL_NT': ["digit NUMBER_TAIL_NT", "ε"],

        'BOND_NT': _tags(BOND_SYMBOLS),
        'DOT_NT': ["<<.>>"],

        'ALIPHATIC_ORGANIC_NT': _tags(ALIPHATIC_ORGANIC_SYMBOLS),
        'AROMATIC_ORGANIC_NT': _tags(AROMATIC_ORGANIC_SYMBOLS),
        'ELEMENT_SYMBOL_NT': _tags(ELEMENT_SYMBOLS),
        'AROMATIC_SYMBOL_NT': _tags(AROMATIC_SYMBOLS),
        'CHIRAL_NT': _tags(CHIRAL_TAGS),
    }

    regex_dict = {
        'regex_digit': re.compile(r'^[0-9]$'),
    }

    return productions, regex_dict


def main():
    setup_logging()

    productions, regex_dict = build_smiles_grammar()

    system_prompt = (
        "You are a chemistry assistant. Given a compound name, respond with "
        "exactly one valid SMILES string for it and nothing else."
    )
    examples = [
        {"role": "user", "content": "acetic acid"},
        {"role": "assistant", "content": "CC(=O)O"},
        {"role": "user", "content": "benzene"},
        {"role": "assistant", "content": "c1ccccc1"},
    ]

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)

    pars_table, map_tt = get_parsing_table_and_map_tt(tokenizer, productions, regex_dict=regex_dict)
    pdas, streamer = generate_grammar_parameters(tokenizer, pars_table, map_tt)

    prompt = create_prompt(
        prompt_input="ibuprofen",
        system_prompt=system_prompt,
        examples=examples,
    )

    result = generate_text(
        model, tokenizer, prompt, pdas, streamer,
        max_new_tokens=64,
    )

    print(f"text        : {result['text']!r}")
    print(f"final stack : {result['pda_stack']}   ([] = grammar satisfied)")


if __name__ == "__main__":
    main()
