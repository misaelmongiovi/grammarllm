"""
generation_analysis.py
=======================
Analizza l'impatto del masking grammaticale sulla distribuzione del modello
ad ogni step di generazione.

Concetti chiave
---------------
Ad ogni step t, il modello produce una distribuzione P_free su tutto il
vocabolario. Il processor applica una maschera binaria: i token non ammessi
dalla grammatica diventano -inf. La distribuzione risultante P_constrained
è una ri-normalizzazione su soli i token validi.

Le metriche calcolate per ogni step sono:

- preserved_mass (float in [0,1]):
    Σ P_free(token) per tutti i token validi.
    È la frazione di massa probabilistica originale che ricade sui token
    ammessi dalla grammatica. Un valore vicino a 1 significa che il modello
    "stava già per generare" qualcosa di valido — il vincolo non distorce.
    Un valore basso (es. 0.05) significa che il modello era quasi certamente
    diretto verso qualcosa di non valido, e il vincolo lo ha forzato.

- token_id (int): token effettivamente generato al passo t.
- token_str (str): decodifica del token.
- true_prob (float): P_free(token_generato) — probabilità che il modello
    libero avrebbe assegnato a quel token.
- constrained_prob (float): P_constrained(token_generato) — probabilità
    dopo il masking e la ri-normalizzazione.
- prob_ratio (float): constrained_prob / true_prob = 1 / preserved_mass
    (approssimativamente). Indica l'amplificazione dovuta al vincolo.
- n_valid_tokens (int): numero di token validi dopo il masking.
- step (int): indice del passo (0-based).

Caso d'uso tipico
-----------------
    result = generate_text(model, tokenizer, prompt, pdas, streamer,
                           output_scores=True)

    analysis = compute_generation_analysis(
        result,
        stateless_processor,
        tokenizer,
        seq_idx=0,
    )

    fig = plot_generation_analysis(analysis, title="Prompt generico")
    fig.savefig("analysis.png")

    # Confronto tra due prompt
    figs = compare_analyses(
        [analysis_generic, analysis_specific],
        labels=["Prompt generico", "Prompt specifico"],
    )

Integrazione con generate_text()
---------------------------------
generate_text() deve essere chiamato con output_scores=True per popolare
result["original_scores"]. Questo attiva il salvataggio della history
dei logit in StatelessLogitsProcessor, necessaria per questa analisi.

    result = generate_text(..., output_scores=True)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import math


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepAnalysis:
    """
    Metriche di analisi per un singolo step di generazione.

    Attributi
    ---------
    step : int
        Indice 0-based del passo di generazione.
    token_id : int
        Token ID generato dal modello vincolato.
    token_str : str
        Decodifica leggibile del token (es. " happy", "positive").
    true_prob : float
        P_free(token_id) — probabilità del token nella distribuzione
        originale del modello (prima del masking).
        Misura quanto il modello libero avrebbe scelto spontaneamente
        questo token.
    constrained_prob : float
        P_constrained(token_id) — probabilità del token dopo il masking
        e la ri-normalizzazione sulla distribuzione vincolata.
    preserved_mass : float
        Σ P_free(t) per t in token_validi.
        Valore in [0,1]. Vicino a 1 → il modello era già allineato con la
        grammatica. Basso → il vincolo ha spostato molta probabilità.
    prob_ratio : float
        constrained_prob / true_prob. Indica di quanto il vincolo ha
        amplificato la probabilità del token generato. Equivale a
        1 / preserved_mass (a meno di errori numerici).
    n_valid_tokens : int
        Numero di token validi dopo il masking.
        Indica quanto è restrittivo il vincolo a questo step.
    entropy_free : float
        Entropia di Shannon (in bit) della distribuzione originale.
        H = -Σ P_free(t) * log2(P_free(t))
    entropy_constrained : float
        Entropia della distribuzione vincolata.
        Sempre <= entropy_free per costruzione (il vincolo riduce l'incertezza).
    """
    step: int
    token_id: int
    token_str: str
    true_prob: float
    constrained_prob: float
    preserved_mass: float
    prob_ratio: float
    n_valid_tokens: int
    entropy_free: float
    entropy_constrained: float


@dataclass
class SequenceAnalysis:
    """
    Analisi completa di una sequenza generata.

    Contiene i dati step-by-step e le metriche aggregate sull'intera
    sequenza, pronte per il plotting.

    Attributi
    ---------
    generated_text : str
        Testo generato decodificato (skip_special_tokens=True).
    total_probability : float
        Probabilità congiunta della sequenza (prodotto delle prob vincolate).
    steps : list[StepAnalysis]
        Un entry per ogni token generato.
    mean_preserved_mass : float
        Media di preserved_mass su tutti gli step.
        Metrica riassuntiva dell'allineamento modello-grammatica.
    min_preserved_mass : float
        Minimo di preserved_mass — identifica il passo più "critico"
        (dove il vincolo ha forzato maggiormente il modello).
    min_preserved_mass_step : int
        Indice del passo con preserved_mass minima.
    mean_true_prob : float
        Media di true_prob — quanto il modello libero concordava con
        le scelte fatte sotto vincolo.
    label : str
        Etichetta opzionale per il plotting (es. nome del prompt).
    """
    generated_text: str
    total_probability: float
    steps: List[StepAnalysis]
    mean_preserved_mass: float
    min_preserved_mass: float
    min_preserved_mass_step: int
    mean_true_prob: float
    label: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_generation_analysis(
    result: dict,
    tokenizer,
    seq_idx: int = 0,
    label: str = "",
) -> SequenceAnalysis:
    """
    Calcola le metriche di analisi per una sequenza generata da generate_text().

    Prerequisito
    ------------
    generate_text() deve essere stato chiamato con output_scores=True:
        result = generate_text(..., output_scores=True)
    Questo garantisce che result["original_scores"] sia popolato.

    Allineamento step ↔ token
    -------------------------
    original_scores[t] ha shape (batch_size * num_beams, vocab_size) e
    contiene i logit del modello PRIMA del masking al passo t.
    Il token generato al passo t è new_tokens[t].
    L'allineamento è diretto: step t → original_scores[t] → new_tokens[t].

    Calcolo preserved_mass
    ----------------------
    Il processor ha salvato:
    - original_scores[t]: logit PRE-masking → softmax → P_free
    - scores[t]:          logit POST-masking → il token con -inf ha P≈0

    Per ricavare quali token sono validi al passo t, usiamo il fatto che
    scores[t][token] == -inf ↔ token non valido. Quindi:

        valid_mask[t] = scores[t] > -1e30
        preserved_mass[t] = Σ P_free[t][token] per token in valid_mask[t]

    Parameters
    ----------
    result : dict
        Output di generate_text() con output_scores=True.
        Deve contenere: "text", "original_scores", "scores",
        "transition_scores" (opzionale), "probability".
    tokenizer : transformers.PreTrainedTokenizer
        Usato per decodificare i token_id.
    seq_idx : int
        Indice della sequenza nel batch/beam (default 0 = sequenza migliore).
    label : str
        Etichetta opzionale per il plotting.

    Returns
    -------
    SequenceAnalysis

    Raises
    ------
    KeyError
        Se result non contiene "original_scores" o "scores".
        Assicurarsi di aver chiamato generate_text() con output_scores=True.
    ValueError
        Se seq_idx è fuori range o i tensori hanno shape inattesa.
    """
    import torch
    import torch.nn.functional as F

    if "original_scores" not in result:
        raise KeyError(
            "'original_scores' non trovato nel result. "
            "Assicurarsi di chiamare generate_text() con output_scores=True."
        )
    if "scores" not in result:
        raise KeyError(
            "'scores' non trovato nel result. "
            "Assicurarsi di chiamare generate_text() con output_scores=True."
        )

    original_scores_list = result["original_scores"]   # list[list[float]] o list[Tensor]
    filtered_scores_list = result["scores"]            # list[list[float]] o list[Tensor]
    generated_text       = result.get("text", "")
    total_prob           = result.get("probability", float("nan"))

    # Normalizza a tensor se sono liste (generate_text può restituire .tolist())
    def to_tensor(x):
        if isinstance(x, list):
            return torch.tensor(x)
        return x  # già tensor

    # BUG FIX: was `len(original_scores_list) - 1`, silently dropping the
    # last generation step (typically the EOS decision) from every analysis.
    n_steps = len(original_scores_list)
    if n_steps == 0:
        raise ValueError("original_scores è vuoto: nessun token generato.")

    # Ricostruiamo la sequenza di token generati dagli scores filtrati:
    # il token generato al passo t è argmax dei filtered scores (o, per
    # sampling, il token che ha received prob > 0 dopo il masking).
    # Ma il modo più affidabile è usare transition_scores se disponibili,
    # altrimenti derivarli dal testo decodificato.
    # La strategia più robusta: per ogni step t, il token generato è
    # quello con prob non-zero nei filtered scores per la sequenza seq_idx.
    # In pratica per greedy/beam: è l'argmax. Per sampling: è il token
    # effettivamente campionato, che possiamo recuperare da transition_scores.

    # Approccio diretto: usiamo i token ID effettivamente generati salvati nel result.
    # Questo è 100% accurato e risolve problemi di re-tokenizzazione (specialmente con beam search).
    # Fallback su encode() solo se per qualche motivo token_ids non è presente.
    tokens_generated = result.get("token_ids")
    if tokens_generated is None:
        tokens_generated = tokenizer.encode(generated_text, add_special_tokens=False)

    # Se la lunghezza non combacia (può capitare con subword tokenization e
    # skip_special_tokens), usiamo la lunghezza minore come limite sicuro.
    effective_steps = min(n_steps, len(tokens_generated))

    steps: list[StepAnalysis] = []

    for t in range(effective_steps):
        orig_logits  = to_tensor(original_scores_list[t])  # (vocab_size,) o (batch*beams, vocab)
        filt_logits  = to_tensor(filtered_scores_list[t])

        # Se i tensori hanno dim batch, estraiamo la sequenza seq_idx
        if orig_logits.dim() == 2:
            orig_logits = orig_logits[seq_idx]
            filt_logits = filt_logits[seq_idx]

        token_id  = tokens_generated[t]
        token_str = tokenizer.decode([token_id])

        # Distribuzioni di probabilità
        p_free        = F.softmax(orig_logits.float(), dim=-1)
        valid_mask    = filt_logits > -1e30          # True = token valido
        n_valid       = int(valid_mask.sum().item())

        # preserved_mass: quanto della distribuzione libera cade sui token validi
        preserved_mass = float(p_free[valid_mask].sum().item())
        preserved_mass = max(0.0, min(1.0, preserved_mass))  # clamp numerics

        # Probabilità del token generato nelle due distribuzioni
        true_prob = float(p_free[token_id].item())

        # P_constrained: softmax solo sui token validi
        # = P_free(token) / preserved_mass  (teoricamente)
        # Calcoliamo direttamente dal filtered per precisione numerica
        p_constrained_raw = F.softmax(
            torch.where(valid_mask, orig_logits.float(), torch.tensor(-1e9)), dim=-1
        )
        constrained_prob = float(p_constrained_raw[token_id].item())

        # Ratio di amplificazione: quanto il vincolo ha boost-ato questo token
        prob_ratio = (constrained_prob / true_prob) if true_prob > 1e-10 else float("inf")

        # Entropie (in bit = log base 2)
        def _entropy(probs):
            p = probs.clamp(min=1e-12)
            return float(-(p * torch.log2(p)).sum().item())

        entropy_free         = _entropy(p_free)
        p_constrained_full   = F.softmax(
            torch.where(valid_mask, orig_logits.float(), torch.full_like(orig_logits.float(), -1e9)),
            dim=-1
        )
        entropy_constrained  = _entropy(p_constrained_full)

        steps.append(StepAnalysis(
            step=t,
            token_id=token_id,
            token_str=token_str,
            true_prob=true_prob,
            constrained_prob=constrained_prob,
            preserved_mass=preserved_mass,
            prob_ratio=prob_ratio,
            n_valid_tokens=n_valid,
            entropy_free=entropy_free,
            entropy_constrained=entropy_constrained,
        ))

        # BUG FIX: sequences shorter than the batch max are padded after EOS;
        # analyzing pad steps polluted the metrics. Include the EOS step
        # (it is a real, grammar-driven decision), then stop.
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None and token_id == eos_id:
            break

    if not steps:
        raise ValueError("Nessuno step analizzabile: sequenza generata vuota.")

    # Metriche aggregate
    masses          = [s.preserved_mass for s in steps]
    mean_mass       = sum(masses) / len(masses)
    min_mass        = min(masses)
    min_mass_step   = masses.index(min_mass)
    mean_true_prob  = sum(s.true_prob for s in steps) / len(steps)

    return SequenceAnalysis(
        generated_text=generated_text,
        total_probability=total_prob,
        steps=steps,
        mean_preserved_mass=mean_mass,
        min_preserved_mass=min_mass,
        min_preserved_mass_step=min_mass_step,
        mean_true_prob=mean_true_prob,
        label=label,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_generation_analysis(
    analysis: SequenceAnalysis,
    title: str = "",
    figsize: tuple = (14, 10),
    show_entropy: bool = True,
):
    """
    Produce un pannello di 4 grafici (o 3 se show_entropy=False) per una
    singola sequenza analizzata.

    Pannelli
    --------
    1. Preserved mass per step (barre + linea media)
       Il grafico principale. Mostra dove il vincolo ha forzato di più.
       Soglie colorate: verde >= 0.5, arancio >= 0.2, rosso < 0.2.

    2. True probability vs Constrained probability per step
       Linee sovrapposte. Mostra come il masking ha amplificato la
       probabilità del token generato.

    3. Numero di token validi per step
       Indica quanto è restrittiva la grammatica a ogni passo.

    4. Entropia libera vs vincolata per step (se show_entropy=True)
       Mostra la riduzione di incertezza imposta dal vincolo.

    Parameters
    ----------
    analysis : SequenceAnalysis
    title : str
        Titolo del pannello (es. nome del prompt).
    figsize : tuple
    show_entropy : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        raise ImportError(
            "matplotlib è necessario per il plotting. "
            "Installarlo con: pip install matplotlib"
        )

    n_panels = 4 if show_entropy else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize)

    steps_x     = [s.step for s in analysis.steps]
    token_labels = [f"[{s.step}]\n{s.token_str!r}" for s in analysis.steps]
    masses       = [s.preserved_mass   for s in analysis.steps]
    true_probs   = [s.true_prob        for s in analysis.steps]
    cons_probs   = [s.constrained_prob for s in analysis.steps]
    n_valid      = [s.n_valid_tokens   for s in analysis.steps]
    ent_free     = [s.entropy_free     for s in analysis.steps]
    ent_cons     = [s.entropy_constrained for s in analysis.steps]

    # ── Pannello 1: Preserved mass ────────────────────────────────────────────
    ax = axes[0]
    colors = [
        "green" if m >= 0.5 else ("orange" if m >= 0.2 else "red")
        for m in masses
    ]
    bars = ax.bar(steps_x, masses, color=colors, alpha=0.75, edgecolor="white")
    ax.axhline(analysis.mean_preserved_mass, color="steelblue", linestyle="--",
               linewidth=1.5, label=f"media = {analysis.mean_preserved_mass:.3f}")
    ax.axhline(0.5, color="green",  linestyle=":", linewidth=1, alpha=0.5)
    ax.axhline(0.2, color="orange", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Preserved mass")
    ax.set_xticks(steps_x)
    ax.set_xticklabels(token_labels, fontsize=8)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        f"Massa probabilistica preservata per step\n"
        f"Testo: {analysis.generated_text!r}  |  "
        f"P(seq)={analysis.total_probability:.4f}"
    )
    # Annotazioni numeriche sulle barre
    for bar, m in zip(bars, masses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{m:.2f}", ha="center", va="bottom", fontsize=7)

    # Legenda colori
    patch_g = mpatches.Patch(color="green",  alpha=0.75, label="≥ 0.5 (basso impatto)")
    patch_o = mpatches.Patch(color="orange", alpha=0.75, label="0.2–0.5 (impatto medio)")
    patch_r = mpatches.Patch(color="red",    alpha=0.75, label="< 0.2 (alto impatto)")
    ax.legend(handles=[patch_g, patch_o, patch_r], loc="lower right", fontsize=8)

    # ── Pannello 2: True prob vs Constrained prob ─────────────────────────────
    ax = axes[1]
    ax.plot(steps_x, true_probs,  "o-", color="steelblue",  label="P_free (senza vincolo)", linewidth=1.5)
    ax.plot(steps_x, cons_probs, "s-", color="firebrick", label="P_constrained (con vincolo)", linewidth=1.5)
    ax.fill_between(steps_x, true_probs, cons_probs,
                    where=[c > t for c, t in zip(cons_probs, true_probs)],
                    alpha=0.15, color="firebrick", label="amplificazione vincolo")
    ax.set_xticks(steps_x)
    ax.set_xticklabels(token_labels, fontsize=8)
    ax.set_ylabel("Probabilità")
    ax.set_ylim(bottom=0)
    ax.set_title("Probabilità del token generato: libera vs vincolata")
    ax.legend(fontsize=9)

    # Annotazioni prob_ratio dove > 2
    for s in analysis.steps:
        if s.prob_ratio > 2.0:
            ax.annotate(f"×{s.prob_ratio:.1f}",
                        xy=(s.step, s.constrained_prob),
                        xytext=(s.step, s.constrained_prob + 0.04),
                        ha="center", fontsize=7, color="firebrick")

    # ── Pannello 3: Numero token validi ───────────────────────────────────────
    ax = axes[2]
    ax.bar(steps_x, n_valid, color="slategray", alpha=0.7, edgecolor="white")
    ax.set_xticks(steps_x)
    ax.set_xticklabels(token_labels, fontsize=8)
    ax.set_ylabel("# token validi")
    ax.set_title("Token validi dopo il masking per step")
    for i, (x, n) in enumerate(zip(steps_x, n_valid)):
        ax.text(x, n + 0.5, str(n), ha="center", va="bottom", fontsize=7)

    # ── Pannello 4: Entropie (opzionale) ─────────────────────────────────────
    if show_entropy:
        ax = axes[3]
        ax.plot(steps_x, ent_free, "o-", color="steelblue", label="H_free (bit)", linewidth=1.5)
        ax.plot(steps_x, ent_cons, "s-", color="firebrick", label="H_constrained (bit)", linewidth=1.5)
        ax.fill_between(steps_x, ent_cons, ent_free, alpha=0.15, color="steelblue",
                        label="riduzione incertezza")
        ax.set_xticks(steps_x)
        ax.set_xticklabels(token_labels, fontsize=8)
        ax.set_ylabel("Entropia (bit)")
        ax.set_title("Entropia della distribuzione: libera vs vincolata")
        ax.legend(fontsize=9)

    fig.suptitle(
        title or f"Analisi generazione: {analysis.label or analysis.generated_text!r}",
        fontsize=13, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    return fig


def compare_analyses(
    analyses: list[SequenceAnalysis],
    labels: list[str] | None = None,
    metric: str = "preserved_mass",
    figsize: tuple = (14, 6),
) -> "matplotlib.figure.Figure":
    """
    Confronta la preserved_mass (o altra metrica) di più analisi sullo stesso grafico.

    Uso tipico
    ----------
    Confrontare l'impatto di prompt diversi, grammatiche diverse, o livelli
    di few-shot sulla distribuzione del modello vincolato.

        fig = compare_analyses(
            [analysis_generic, analysis_fewshot, analysis_fewshot_good],
            labels=["Prompt generico", "Few-shot scarso", "Few-shot buono"],
            metric="preserved_mass",
        )

    Parameters
    ----------
    analyses : list[SequenceAnalysis]
    labels : list[str], optional
        Etichette per la legenda. Se None, usa analysis.label o l'indice.
    metric : str
        La metrica da plottare per ogni step. Opzioni:
        "preserved_mass", "true_prob", "constrained_prob",
        "n_valid_tokens", "entropy_free", "entropy_constrained", "prob_ratio".
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
    except ImportError:
        raise ImportError("matplotlib e numpy sono necessari per il plotting.")

    if labels is None:
        labels = [a.label or f"Sequenza {i}" for i, a in enumerate(analyses)]

    metric_map = {
        "preserved_mass":    lambda s: s.preserved_mass,
        "true_prob":         lambda s: s.true_prob,
        "constrained_prob":  lambda s: s.constrained_prob,
        "n_valid_tokens":    lambda s: s.n_valid_tokens,
        "entropy_free":      lambda s: s.entropy_free,
        "entropy_constrained": lambda s: s.entropy_constrained,
        "prob_ratio":        lambda s: min(s.prob_ratio, 50.0),  # cap per leggibilità
    }
    if metric not in metric_map:
        raise ValueError(f"Metrica '{metric}' non riconosciuta. Opzioni: {list(metric_map)}")

    extractor = metric_map[metric]
    colors = cm.tab10.colors

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ── Sinistra: curva per step ──────────────────────────────────────────────
    ax = axes[0]
    for i, (analysis, label) in enumerate(zip(analyses, labels)):
        values = [extractor(s) for s in analysis.steps]
        steps  = [s.step for s in analysis.steps]
        color  = colors[i % len(colors)]
        ax.plot(steps, values, "o-", color=color, label=label,
                linewidth=1.8, markersize=5, alpha=0.85)

    if metric == "preserved_mass":
        ax.axhline(0.5, color="green",  linestyle=":", alpha=0.4, linewidth=1)
        ax.axhline(0.2, color="orange", linestyle=":", alpha=0.4, linewidth=1)
        ax.set_ylim(0, 1.05)

    ax.set_xlabel("Step di generazione")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} per step")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ── Destra: boxplot comparativo ───────────────────────────────────────────
    ax = axes[1]
    data    = [[extractor(s) for s in a.steps] for a in analyses]
    bp      = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], colors[:len(analyses)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay dei singoli punti (jitter)
    import random
    random.seed(42)
    for i, (vals, color) in enumerate(zip(data, colors[:len(analyses)])):
        jitter = [i + 1 + random.uniform(-0.12, 0.12) for _ in vals]
        ax.scatter(jitter, vals, color=color, alpha=0.5, s=20, zorder=3)

    # Annotazione media
    for i, vals in enumerate(data):
        mean = sum(vals) / len(vals)
        ax.text(i + 1, mean, f"μ={mean:.2f}",
                ha="center", va="bottom", fontsize=8, color="black")

    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Distribuzione {metric.replace('_', ' ')} (boxplot)")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Confronto: {metric.replace('_', ' ').title()}",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    return fig


def print_analysis_summary(analysis: SequenceAnalysis) -> None:
    """
    Stampa un riepilogo testuale dell'analisi, utile per ambienti senza GUI.

    Output esempio
    --------------
    ═══════════════════════════════════════════
    Analisi generazione: "positive"
    P(sequenza) = 0.8923
    ─────────────────────────────────────────────────────────────────────────
    Step  Token           P_free   P_cstr  Preserved  Ratio  N_valid  H_free
    ─────────────────────────────────────────────────────────────────────────
       0  ' positive'    0.0234   0.8912   0.0263    38.1x       3    13.2
    ─────────────────────────────────────────────────────────────────────────
    Media preserved mass : 0.026
    Min  preserved mass  : 0.026 @ step 0
    Media true prob      : 0.023
    ═══════════════════════════════════════════

    Parameters
    ----------
    analysis : SequenceAnalysis
    """
    sep = "─" * 75
    print("═" * 75)
    label = analysis.label or analysis.generated_text
    print(f"Analisi generazione: {label!r}")
    print(f"P(sequenza) = {analysis.total_probability:.6f}")
    print(sep)
    header = f"{'Step':>4}  {'Token':<18} {'P_free':>7} {'P_cstr':>7} {'Pres.mass':>9} {'Ratio':>7} {'N_val':>6} {'H_free':>7}"
    print(header)
    print(sep)
    for s in analysis.steps:
        ratio_str = f"{s.prob_ratio:6.1f}x" if s.prob_ratio < 999 else "  ∞    "
        print(
            f"{s.step:>4}  {repr(s.token_str):<18} "
            f"{s.true_prob:>7.4f} {s.constrained_prob:>7.4f} "
            f"{s.preserved_mass:>9.4f} {ratio_str:>7} "
            f"{s.n_valid_tokens:>6}  {s.entropy_free:>6.2f}"
        )
    print(sep)
    print(f"Media preserved mass : {analysis.mean_preserved_mass:.4f}")
    print(f"Min  preserved mass  : {analysis.min_preserved_mass:.4f} @ step {analysis.min_preserved_mass_step}")
    print(f"Media true prob      : {analysis.mean_true_prob:.4f}")
    print("═" * 75)