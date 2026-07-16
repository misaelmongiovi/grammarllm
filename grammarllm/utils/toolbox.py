"""
toolbox.py
==========
Helper per costruire i prompt.

`create_prompt` restituisce una conversazione (list[dict]) da passare a
generate_text, che di default la renderizza con il chat template NATIVO del
tokenizer.  Nella maggior parte dei casi non serve toccare `chat_template`.
"""

# ATTENZIONE — fallback generico, NON usarlo con i modelli instruct moderni.
#
# Questo template rende <|system|> / <|user|> / <|assistant|>, che NON sono
# token speciali per Llama-3, Qwen, Mistral & co.: il tokenizer li frantuma in
# '<', '|', 'system', '|', '>' e il modello riceve un formato di chat su cui non
# è mai stato addestrato.  L'impatto è grosso e silenzioso — su WoS con
# Llama-3.2-1B-Instruct: L1 micro-F1 0.539 -> 0.145, con il modello che smette di
# classificare e inizia a ripetere il system prompt.
#
# generate_text() ora usa `tokenizer.chat_template` (quello nativo del modello)
# per default e non lo sovrascrive più.  Passa questo template SOLO per modelli
# base/legacy che non ne hanno uno proprio (`tokenizer.chat_template is None`).
chat_template = """
{%- for message in messages %}
    {{- '<|' + message['role'] + '|>\n' }}
    {{- message['content'].strip() + eos_token + '\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|assistant|>\n' }}
{%- endif %}
"""

def create_prompt(prompt_input, system_prompt, examples):
    """Create a tokenized prompt for the model."""

    messages = [
        {
            "role": "system",
            "content": f"{system_prompt}\n\n",
        }
    ]
    
    if examples:
        for item in examples:
            messages.append(item)

    messages.append({"role": "user", "content": prompt_input})
    return messages