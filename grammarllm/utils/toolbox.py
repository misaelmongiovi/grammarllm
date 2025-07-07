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