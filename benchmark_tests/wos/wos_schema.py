"""
Pydantic model for the JSON-notation WoS grammar (grammar.mode: json in config.yaml).

Mirrors the same parent/child taxonomy as the <<>>-tag grammar in config.yaml
(grammar.productions), but expressed as a Pydantic model and converted via
pydantic_to_productions(). Note: `child` is a flat enum over ALL subcategories,
not scoped to the chosen `parent` — pydantic_to_grammar rejects anyOf branches
of the same JSON type (object vs object), so a discriminated union of one model
per parent is not accepted by the CONVERTER.

This is a limitation of pydantic_to_grammar, NOT of LL(1): a per-parent child
schema is perfectly expressible. Writing it directly as a <<>>-tag grammar

    'S*':  ['<<{"parent": "cs", "child": ">> C_J', ...]
    'C_J': ['<<computer graphics">>', ...]

builds and enforces parent->child correctly: the shared token prefix
'{" parent ": Ġ"' is left-factored and the parent literal discriminates.
(Before the step-1 grouping fix in grammar_generation.py this shape was broken —
it either raised a spurious LL(1) Conflict or, when the child FIRST sets were
disjoint, silently accepted cross products like parent=cs + child=electricity.)

Use grammar.mode: tags in config.yaml when strict parent->child hierarchy
enforcement is required through the Pydantic path.
"""
from typing import Literal

from pydantic import BaseModel

PARENTS = ["biochemistry", "civil", "cs", "ece", "mae", "medical", "psychology"]

CHILDREN = list(dict.fromkeys([
    # biochemistry
    "cell biology", "dna/rna sequencing", "enzymology", "genetics", "human metabolism",
    "immunology", "molecular biology", "northern blotting", "polymerase chain reaction",
    "southern blotting",
    # civil
    "ambient intelligence", "bamboo as building material", "construction management",
    "geotextile", "green building", "highway network system", "nano concrete",
    "rainwater harvesting", "remote sensing", "smart material", "solar energy",
    "stealth technology", "suspension bridge", "transparent concrete",
    "underwater windmill", "water pollution",
    # cs
    "algorithm design", "bioinformatics", "computer graphics", "computer programming",
    "computer vision", "cryptography", "data structures", "distributed computing",
    "image processing", "machine learning", "network security", "operating systems",
    "parallel computing", "relational databases", "software engineering",
    "structured storage", "symbolic computation",
    # ece
    "analog signal processing", "control engineering", "digital control",
    "electric motor", "electrical circuits", "electrical generator",
    "electrical network", "electricity", "lorentz force law", "microcontroller",
    "operational amplifier", "pid controller", "satellite radio",
    "signal-flow graph", "single-phase electric power", "state space representation",
    "system identification", "voltage law",
    # mae
    "computer-aided design", "fluid mechanics", "hydraulics",
    "internal combustion engine", "machine design", "manufacturing engineering",
    "materials engineering", "strength of materials", "thermodynamics",
    # medical
    "addiction", "allergies", "alzheimer's disease", "ankylosing spondylitis",
    "anxiety", "asthma", "atopic dermatitis", "atrial fibrillation", "autism",
    "bipolar disorder", "birth control", "cancer", "children's health",
    "crohn's disease", "dementia", "depression", "diabetes", "digestive health",
    "emergency contraception", "fungal infection", "headache", "healthy sleep",
    "heart disease", "hepatitis c", "hereditary angioedema", "hiv/aids",
    "hypothyroidism", "idiopathic pulmonary fibrosis", "irritable bowel syndrome",
    "kidney health", "low testosterone", "lymphoma", "medicare", "menopause",
    "mental health", "migraine", "multiple sclerosis", "myelofibrosis",
    "osteoarthritis", "osteoporosis", "outdoor health", "overactive bladder",
    "parenting", "parkinson's disease", "polycythemia vera", "psoriasis",
    "psoriatic arthritis", "rheumatoid arthritis", "schizophrenia", "senior health",
    "skin care", "smoking cessation", "sports injuries", "sprains and strains",
    "stress management", "weight loss",
    # psychology
    "antisocial personality disorder", "attention", "borderline personality disorder",
    "child abuse", "eating disorders", "false memories", "gender roles",
    "leadership", "media violence", "nonverbal communication", "person perception",
    "prejudice", "prenatal development", "problem-solving", "prosocial behavior",
    "seasonal affective disorder", "social cognition",
]))


class WosClassification(BaseModel):
    parent: Literal[tuple(PARENTS)]
    child: Literal[tuple(CHILDREN)]
