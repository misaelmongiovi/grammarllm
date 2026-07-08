"""
logits_processor.py
===================
Implementa StatelessLogitsProcessor, il componente che vincola la generazione
del modello alla grammatica LL(1) mascherando i logit ad ogni step.

Posizione nella pipeline
------------------------
È il componente runtime centrale, attivo durante model.generate():

    setup:  PushdownAutomaton (base_pdas)
                    ↓
    runtime: StatelessLogitsProcessor.__call__(input_ids, scores)
                    ↓
            get_tokens() → maschera logit → modello genera token valido
                    ↓
            next step: nuovo input_ids con il token generato

Architettura "stateless"
-------------------------
A differenza di un approccio stateful (un PDA per beam aggiornato in-place),
il processor ri-simula o recupera dalla cache lo stato PDA ad ogni step
partendo dalla history dei token generati (input_ids).

Questo è necessario per il beam search: HuggingFace può scartare e rimpiazzare
beam tra uno step e l'altro, rendendo impossibile mantenere uno stato PDA
"live" sincronizzato con il beam corrente.

La verità è sempre input_ids: la history dei token generati determina univocamente
lo stato PDA corrispondente.

Cache LRU
---------
Per evitare la ri-simulazione completa O(L) ad ogni step, il processor mantiene
una cache { (prompt_idx, history_tuple): PDA }.  Ad ogni step:
  - Cache hit: clona il PDA cached (O(1)).
  - Case A: prefix in cache → clona e avanza di 1 token (O(1)).
  - Case B: nessun prefix → ri-simula da base_pda (O(L)).

La cache usa una politica LRU tramite dict insertion-order: ogni accesso
sposta la chiave in fondo (pop + reinsert), e l'eviction rimuove dal fronte.
"""

import logging
from transformers import LogitsProcessor
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from io import StringIO
import os

try:
    from .lookahead import lookahead_tokens, get_vocab_trie
except ImportError:
    from lookahead import lookahead_tokens, get_vocab_trie

# FIX: cap the PDA cache to avoid unbounded memory growth during long generations.
# Each entry stores a PDA object with its stack; with num_beams=5 and
# max_new_tokens=512 this would otherwise accumulate ~2560 live PDA objects.
_MAX_CACHE_SIZE = 2048

class StatelessLogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor HuggingFace che vincola la generazione a una grammatica LL(1)
    re-simulando il PDA ad ogni step di generazione.

    Architettura "stateless"
    ------------------------
    A differenza di un approccio stateful (dove un PDA viene aggiornato
    incrementalmente), questo processore deriva lo stato del PDA direttamente
    dalla history dei token generati (input_ids) ad ogni step. Questo è
    necessario per il beam search, dove le hypothesis vengono rimescolate e
    scambiate tra beam: un PDA stateful riceverebbe token di beam diversi
    e si corromperebbe.

    Ottimizzazione con cache
    ------------------------
    La re-simulation naïve è O(L) per step (L = lunghezza history). La cache
    pda_cache = { (prompt_idx, history_tuple): pda_state } riduce il costo
    a O(1) nel caso comune: si recupera lo stato del passo precedente e si
    avanza di un solo token (Case A). La cache usa una politica LRU implementata
    tramite pop+reinsert su dict Python 3.7+ (che preserva l'ordine di inserzione).

    Integrazione
    ------------
    Istanziato da generate_with_constraints.generate_text():
        stateless_processor = StatelessLogitsProcessor(
            tokenizer, base_pdas, sequences_per_prompt, prompt_len, temperature
        )
        outputs = model.generate(..., logits_processor=[stateless_processor])
    Dopo la generazione, generate_text() chiama get_pda_for_sequence()
    per ricostruire pda_history nel risultato.
    """

    def __init__(self, tokenizer, base_pdas, sequences_per_prompt=1, prompt_len=0, temperature=1.0, track_score_history=False):
        """
        Inizializza il processor con i PDA template e i parametri di generazione.

        Parametri
        ---------
        tokenizer : HuggingFace tokenizer
            Il tokenizer del modello.  Usato per:
            - Identificare i token speciali (BOS, PAD, UNK) da saltare nella
              re-simulation della history.
            - Forzare il token EOS quando la grammatica è soddisfatta (eos_token_id).

        base_pdas : list[PushdownAutomaton]
            Lista di PDA template — uno per ogni prompt nel batch.
            Vengono clonati (mai modificati) per ogni beam/sequenza.
            Prodotti da generate_grammar_parameters() in generate_with_constraints.py.

        sequences_per_prompt : int
            Numero di sequenze generate per ogni prompt.
            = max(num_beams, num_return_sequences).
            Usato per mappare l'indice flat del batch (0..batch_size-1) al
            prompt di origine: prompt_idx = i // sequences_per_prompt.

        prompt_len : int
            Lunghezza del prompt in token (inclusi token speciali).
            = input_ids.shape[1] al momento della creazione del processor.
            Usato per estrarre la history dei token generati da input_ids:
            history = input_ids[i][prompt_len:]

        temperature : float
            DEPRECATO — non più applicata dal processor.
            BUG FIX: il processor divideva scores / temperature, ma la
            temperature passata in kwargs a model.generate() viene applicata
            anche dal TemperatureLogitsWarper di HuggingFace (dopo i
            logits_processor) → doppia applicazione con do_sample=True.
            La temperatura è ora gestita esclusivamente da HF generate().
            Il parametro resta per retro-compatibilità di firma.

        track_score_history : bool
            Se True, accumula original_scores_history / filtered_scores_history
            (un tensor (batch, vocab) per step — costoso in memoria).
            Default False: nessun clone e nessun accumulo.

        Stato interno
        -------------
        - pda_cache : dict { (prompt_idx, tuple(history)): PDA }
            Cache LRU degli stati PDA.  Inizialmente vuota, si popola
            durante la generazione.  Resettata da reset() prima di ogni
            nuova generazione.
        - original_scores_history, filtered_scores_history : list[Tensor]
            Storico dei logit per analisi post-hoc.  NOTA: crescono senza
            limite se reset() non viene chiamato tra generazioni.
        """
        self.tokenizer = tokenizer
        self.base_pdas = base_pdas
        self.sequences_per_prompt = sequences_per_prompt
        self.prompt_len = prompt_len
        self.temperature = temperature
        self.track_score_history = track_score_history

        # History tracking (if requested)
        self.original_scores_history = []
        self.filtered_scores_history = []
        
        # Cache for PDA states: { tuple(token_ids): pda_state }
        # Key: tuple of tokens (history)
        # Value: PDA object (cloned and advanced)
        self.pda_cache = {}

        # Digest-keyed lookahead mask cache: {(tuple(stack), residue): {tid: path}}
        self.mask_cache = {}
        self.vocab_trie = None
        if any(getattr(p, "lookahead", False) for p in base_pdas):
            self.vocab_trie = get_vocab_trie(tokenizer)

        # Logging limiter
        self.log_counter = 0
        
        # Detail Logger
        self.detail_logger = logging.getLogger("grammarllm.detail")

    def reset(self):
        """
        Svuota la cache PDA, le history dei logit e il contatore di log.

        Deve essere chiamato prima di ogni nuova generazione per evitare
        la crescita illimitata delle history (OOM su generazioni lunghe)
        e per garantire che la cache non contenga stati obsoleti.

        Integrazione
        ------------
        Chiamato esplicitamente da generate_with_constraints.generate_text()
        immediatamente dopo aver istanziato il processor:
            stateless_processor.reset()
        Anche se il processor viene ricreato ad ogni chiamata a generate_text(),
        il reset esplicito documenta l'invariante e protegge contro refactoring
        futuri che potrebbero riutilizzare l'istanza.
        """
        self.original_scores_history = []
        self.filtered_scores_history = []
        self.pda_cache = {}
        self.mask_cache = {}
        self.log_counter = 0

    def log_comparison(self, orig_probs, filt_probs, beam_idx, step):
        """
        Logga le top-10 distribuzioni originale e filtrata come Rich Table.

        Utile per debug: mostra quali token avevano alta probabilità prima
        del masking (distribuzione del modello) e quali rimangono dopo
        (distribuzione vincolata dalla grammatica).

        Performance
        -----------
        Questa funzione è costosa: costruisce e serializza una Rich Table
        per ogni beam ad ogni step. Con num_beams=5 e max_new_tokens=200,
        produce ~1000 tabelle. Per questo è protetta da:
            if self.detail_logger.isEnabledFor(logging.DEBUG):
        In produzione il logger è a livello INFO → nessun overhead.

        Integrazione
        ------------
        Chiamata da __call__() solo se il detail_logger è in modalità DEBUG.
        L'output va nel file grammarllm/temp/GRAM-DETAIL.log configurato
        da generate_with_constraints.setup_logging().
        """
        # Get Top 10 for Original
        top_orig_val, top_orig_ind = torch.topk(orig_probs, 10)
        orig_tokens = self.tokenizer.convert_ids_to_tokens(top_orig_ind.tolist())
        orig_vals = top_orig_val.tolist()

        # Get Top 10 for Filtered
        top_filt_val, top_filt_ind = torch.topk(filt_probs, 10)
        filt_tokens = self.tokenizer.convert_ids_to_tokens(top_filt_ind.tolist())
        filt_vals = top_filt_val.tolist()
        
        table = Table(title=f"Sequence {beam_idx} - Step {step} (Comparison)", show_lines=True)
        table.add_column("Original Token", style="cyan")
        table.add_column("Orig Prob", justify="right", style="green")
        table.add_column("Filtered Token", style="magenta")
        table.add_column("Filt Prob", justify="right", style="yellow")
        
        for i in range(10):
            o_tok = str(orig_tokens[i]) if i < len(orig_tokens) else ""
            o_prob = f"{orig_vals[i]:.6f}" if i < len(orig_vals) else ""
            
            f_tok = str(filt_tokens[i]) if i < len(filt_tokens) else ""
            f_prob = f"{filt_vals[i]:.6f}" if i < len(filt_vals) else ""
            
            table.add_row(o_tok, o_prob, f_tok, f_prob)
            
        # Capture rich output to string
        buf = StringIO()
        console = Console(file=buf, force_terminal=False, width=120)
        console.print(table)
        
        self.detail_logger.info(buf.getvalue())

    def __call__(self, input_ids, scores):
        """
        Applica il masking grammaticale ai logit ad ogni step di generazione.

        Questo metodo è chiamato da HuggingFace model.generate() prima del
        campionamento, per ogni step e per ogni sequenza nel batch.

        Flusso per ogni sequenza i
        --------------------------
        1. Identifica il prompt di appartenenza: prompt_idx = i // sequences_per_prompt
        2. Estrae la history generata: input_ids[i][prompt_len:]
        3. Recupera/simula il PDA (con cache LRU):
           - Cache hit: clone del PDA cached, LRU bookkeeping (pop+reinsert)
           - Case A: clone dell'antenato (history[:-1]), avanza di 1 token
           - Case B: clone del base PDA, replay dell'intera history
        4. Maschera i logit: token non in pda.get_tokens() → -inf

        Gestione EOS
        ------------
        Se pda.eos() (grammatica consumata): tutti i token → -inf tranne EOS.
        Se pda.get_tokens() vuoto (dead-end): idem (fallback di sicurezza).

        Invariante sull'errore
        -----------------------
        Se next_state() lancia ValueError durante la re-simulation, l'errore
        viene propagato senza essere catturato. Questo significa che un token
        invalido è entrato nella history — indica un bug nel masking upstream
        o un token speciale non filtrato. Non viene mai usato un workaround.

        Parameters
        ----------
        input_ids : torch.Tensor
            Shape (batch_size * num_beams, seq_len). Contiene prompt + token
            generati. I token generati iniziano all'indice prompt_len.
        scores : torch.Tensor
            Shape (batch_size * num_beams, vocab_size). Logit del modello.
            Viene modificato in-place (mascheratura a -inf).

        Returns
        -------
        torch.Tensor
            scores modificato con i token invalidi mascherati a -inf.
        """
        batch_size = scores.shape[0]
        
        # Debug only first call or sparse logging
        # We assume prompt is skipped, so shape[1] is absolute length including prompt
        current_len = input_ids.shape[1]
        

        # BUG FIX: temperature scaling removed from the processor.
        # It was applied here AND by HuggingFace's TemperatureLogitsWarper
        # (which runs after logits_processor when do_sample=True and
        # `temperature` is in the generate kwargs) — double scaling.
        # Temperature is now handled exclusively by HF generate().

        # Check if we should log based on logger level
        do_log = logging.getLogger().getEffectiveLevel() <= logging.INFO

        # Capture the original (pre-masking) logits only when needed:
        # cloning a (batch, vocab) tensor every step is expensive and the
        # history accumulation was the main OOM source on long generations.
        do_detail_log = self.detail_logger.isEnabledFor(logging.DEBUG)
        needs_original = self.track_score_history or do_detail_log
        raw_scores = scores.clone() if needs_original else None

        for i in range(batch_size):
            # 1. Identify Parent Prompt
            # If batch_size=6 and num_beams=3 -> Prompts: 0,0,0, 1,1,1
            # Index i maps to prompt: i // num_beams
            # Note: We must clamp index just in case of mismatch, though standard HF behavior guarantees this.
            prompt_idx = i // self.sequences_per_prompt
            
            if prompt_idx >= len(self.base_pdas):
                logging.warning(f"Batch index {i} maps to prompt {prompt_idx} but only {len(self.base_pdas)} PDAs available. Wrapping mod.")
                prompt_idx = prompt_idx % len(self.base_pdas)

            base_pda = self.base_pdas[prompt_idx]

            # 2. Extract Generation History (Skip Prompt)
            # input_ids[i] includes prompt + new tokens
            current_seq = input_ids[i]
            history_tokens = current_seq[self.prompt_len:].tolist()
            history_tuple = tuple(history_tokens)
            # Use (prompt_idx, history_tuple) as key to avoid collisions between different prompts in a batch
            cache_key = (prompt_idx, history_tuple)

            # 3. Retrieve or Re-Simulate PDA
            if cache_key in self.pda_cache:
                # Cache Hit.
                # BUG FIX: return a clone, not the cached object itself.
                # get_tokens() modifies current_terminals in-place; if two beams
                # share the same history key (common in beam search) and we hand
                # out the same object, step-4 calls on different batch indices
                # corrupt each other's current_terminals.
                # Mark the key as recently-used by reinserting it at the end
                # (LRU bookkeeping — dict preserves insertion order in Python 3.7+).
                _cached = self.pda_cache.pop(cache_key)
                self.pda_cache[cache_key] = _cached
                pda = _cached.clone()
            else:
                # Cache Miss - Needs Re-simulation
                # Optimization: Can we find a prefix in cache?
                # Ideally, history = prefix + [new_token].
                # We check cache[prefix].
                
                # Try finding closest ancestor in cache
                found_ancestor = False
                prefix_tuple = history_tuple[:-1]
                
                if len(history_tokens) > 0 and (prompt_idx, prefix_tuple) in self.pda_cache:
                    # Case A: Linear Advance from Cache.
                    # The prefix state is already validated and cached.
                    # We only need to advance by the single new token.
                    # If that token is invalid for the grammar, this sequence
                    # is outside the language — raise immediately.
                    # The old code caught the exception and fell back to full
                    # re-simulation, which then silently resumed from a partial
                    # state, effectively bypassing the grammar constraint.
                    #
                    # LRU bookkeeping: the ancestor is being actively used as
                    # a parent for a new beam state.  Move it to the end of the
                    # dict so it is not evicted before its children are built.
                    ancestor_key = (prompt_idx, prefix_tuple)
                    _ancestor = self.pda_cache.pop(ancestor_key)
                    self.pda_cache[ancestor_key] = _ancestor
                    ancestor_pda = _ancestor
                    pda = ancestor_pda.clone()
                    # Let ValueError propagate for truly invalid tokens — a
                    # token the grammar does not allow must never have been
                    # generated.  _advance_token handles the legitimate
                    # exceptions (special tokens, forced EOS).
                    self._advance_token(pda, history_tokens[-1])
                    found_ancestor = True

                if not found_ancestor:
                    # Case B: Full Re-simulation from Base.
                    # This path is taken only when no cached prefix exists
                    # (first token, or cache was evicted).
                    # Every token in history_tokens was already masked by
                    # this processor at the step it was generated, so every
                    # token MUST be valid for the grammar.
                    # If next_state raises, it means either:
                    #   (a) the masking logic has a bug, or
                    #   (b) a special token slipped through — skip it.
                    # We no longer swallow grammar errors silently.
                    pda = base_pda.clone()
                    for token in history_tokens:
                        # Propagate ValueError for truly invalid tokens.
                        # A grammar violation here indicates a masking failure
                        # at an earlier step — surface it so it can be fixed.
                        if not self._advance_token(pda, token):
                            break
                
                # Store in cache — evict least-recently-used entries when full.
                # BUG FIX: the original FIFO policy evicted the *oldest* entries,
                # which are the *shortest-prefix* states — exactly the most reusable
                # ancestors for beam search.  Evicting them forces full re-simulation
                # for all descendant beams that shared them.
                # LRU is correct: evict the entries that have not been accessed
                # recently (they are unlikely to be needed again).
                # Python 3.7+ dicts preserve insertion order; we implement LRU by
                # popping and re-inserting on every access (done in the cache-hit
                # branch above) and evicting from the front here.
                if len(self.pda_cache) >= _MAX_CACHE_SIZE:
                    evict_count = _MAX_CACHE_SIZE // 4
                    for lru_key in list(self.pda_cache.keys())[:evict_count]:
                        del self.pda_cache[lru_key]
                self.pda_cache[cache_key] = pda

            # 4. Get Valid Tokens & Mask
            if pda.eos():
                # Stack empty -> Allow only EOS or Pad
                # Assuming eos_token_id is available in tokenizer
                # Mask EVERYTHING except EOS
                scores[i, :] = -float("inf")
                scores[i, self.tokenizer.eos_token_id] = 0
            else:
                valid_tokens, _paths = self._valid_ids(pda)
                
                if not valid_tokens:
                     # No valid tokens but stack not empty? Dead end logic
                     if do_log: logging.warning(f"PDA {i} Dead End. Stack: {pda.stack}")
                     # Force EOS to exit gracefully
                     scores[i, :] = -float("inf")
                     scores[i, self.tokenizer.eos_token_id] = 0
                else:
                    # Convert terminals to token IDs
                    # Note: pda.get_tokens() returns TERMINALS or TOKENS?
                    # Looking at PushdownAutomaton.py: get_tokens() returns TOKENS (ids).
                    # "tokens.update(self.map_terminals_tokens[terminal])"
                    
                    valid_ids = list(valid_tokens)
                    mask = torch.ones_like(scores[i], dtype=torch.bool)
                    mask[valid_ids] = False # False = Do not mask (Keep)
                    

                    scores[i] = scores[i].masked_fill(mask, -float('inf'))

        # Log Comparison (Original vs Filtered) — only when detail logger is at DEBUG level.
        # FIX: log_comparison builds and serializes a Rich Table for every beam at every step.
        # With num_beams=5 and max_new_tokens=200 this causes ~1000 Rich Table renders
        # and is the main production performance bottleneck. Guard it explicitly.
        if do_detail_log:
            original_probs = F.softmax(raw_scores, dim=-1)
            filtered_probs = F.softmax(scores, dim=-1)
            for i in range(batch_size):
                self.log_comparison(original_probs[i], filtered_probs[i], beam_idx=i, step=current_len)

        # Save score history for optional post-hoc analysis.
        # BUG FIX: previously appended unconditionally — one (batch, vocab)
        # tensor per step (~1.3 GB with vocab=128k, 5 beams, 512 steps).
        # Now gated behind track_score_history, set by generate_text() only
        # when the caller asked for output_scores.
        if self.track_score_history:
            self.original_scores_history.append(raw_scores)
            self.filtered_scores_history.append(scores.clone())

        return scores

    def _advance_token(self, pda, token):
        """
        Avanza il PDA di un token durante la re-simulation della history.

        Ritorna False quando il replay deve fermarsi:
          - la grammatica è già completamente consumata (stack vuoto), oppure
          - il token è l'EOS del modello ma NON è un terminale valido nello
            stato corrente.  Questo accade quando __call__ ha forzato EOS via
            il fallback dead-end (stack non vuoto ma nessun token valido):
            quell'EOS non appartiene al linguaggio e chiamare next_state()
            farebbe crashare la re-simulation di uno stato che il processor
            stesso ha prodotto.

        I token speciali (BOS/PAD/UNK) vengono saltati (ritorna True senza
        avanzare).  Per ogni altro token invalido next_state() propaga
        ValueError: indica un bug nel masking upstream e non va mai nascosto.
        """
        if pda.eos():
            return False
        if token in (self.tokenizer.bos_token_id, self.tokenizer.pad_token_id,
                     getattr(self.tokenizer, 'unk_token_id', None)):
            return True
        if token == self.tokenizer.eos_token_id:
            valid, paths = self._valid_ids(pda)
            if token not in valid:
                return False        # EOS forced by the dead-end fallback
            self._advance(pda, token, paths)
            return True
        self._advance(pda, token)
        return True

    def _valid_ids(self, pda):
        """
        Valid next token ids for *pda*, routed by engine.

        Legacy: (pda.get_tokens(), None).
        Lookahead: (ids, paths) from the digest-memoized g_t_r mask.
        """
        if not getattr(pda, "lookahead", False):
            return pda.get_tokens(), None
        digest = (tuple(pda.stack), pda.residue)
        entry = self.mask_cache.get(digest)
        if entry is None:
            entry = lookahead_tokens(pda, self.vocab_trie)
            if len(self.mask_cache) >= _MAX_CACHE_SIZE:
                for old_key in list(self.mask_cache.keys())[:_MAX_CACHE_SIZE // 4]:
                    del self.mask_cache[old_key]
            self.mask_cache[digest] = entry
        return list(entry.keys()), entry

    def _advance(self, pda, token, paths=None):
        """Consume *token* on *pda*, routed by engine. ValueError on invalid."""
        if not getattr(pda, "lookahead", False):
            pda.next_state(token)
            return
        if paths is None:
            _, paths = self._valid_ids(pda)
        path = paths.get(token)
        if path is None:
            raise ValueError(
                f"Token {token} is not derivable from state "
                f"(stack={pda.stack}, residue={pda.residue!r}) under lookahead"
            )
        pda.apply_lookahead_path(*path)

    def get_pda_for_sequence(self, token_ids, prompt_idx=0):
        """
        Restituisce un PDA avanzato alla posizione corrispondente a token_ids.

        API pubblica per l'ispezione post-hoc dello stato del PDA.
        Usata da generate_with_constraints.generate_text() dopo la generazione
        per ricostruire la pda_history nel risultato:
            for t in range(1, len(new_tokens) + 1):
                pda_at_t = stateless_processor.get_pda_for_sequence(new_tokens[:t])
                stack_history.append(list(pda_at_t.stack))

        Ottimizzazione
        --------------
        Se la history è già in pda_cache (il che è quasi sempre vero,
        perché __call__() ha già simulato tutti i prefissi durante la
        generazione), restituisce un clone del PDA cachato senza ricalcolare.
        Nel caso raro di cache miss, esegue una full re-simulation.

        Invariante sull'errore
        -----------------------
        Come in __call__(), next_state() può propagare ValueError per token
        invalidi. Non viene mai usato il vecchio pattern try/except break
        che restituiva stati parziali (FIX Bug-2).

        Parameters
        ----------
        token_ids : list[int] | tuple[int]
            Sequenza di token ID (senza il prompt).
        prompt_idx : int
            Indice del prompt (0-based) per selezionare il base PDA corretto.

        Returns
        -------
        PushdownAutomaton
            Istanza clonata al passo corrispondente a token_ids.
        """
        history_tuple = tuple(token_ids)
        cache_key = (prompt_idx, history_tuple)
        
        # Check cache
        if cache_key in self.pda_cache:
            return self.pda_cache[cache_key].clone()
            
        # Fallback: Re-simulate from base PDA.
        # BUG FIX: the old code caught ValueError silently and broke out of the
        # loop, returning a partial PDA state — the same silent-bypass that was
        # fixed in __call__.  This method is called to build pda_history in the
        # final result; a partial state would misrepresent which tokens were
        # actually consumed.  Let ValueError propagate so callers see the error.
        pda = self.base_pdas[prompt_idx].clone()
        for token in token_ids:
            # ValueError propagates for invalid tokens — no silent bypass.
            # _advance_token stops cleanly on grammar completion or forced EOS.
            if not self._advance_token(pda, token):
                break
        return pda