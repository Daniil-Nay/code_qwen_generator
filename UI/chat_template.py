def get_chat_template(tokenizer, chat_template="qwen-2.5", **kwargs):
    
    template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{'<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n'}}"
        "{% elif message['role'] == 'assistant' %}"
        "{{'<|im_start|>assistant\\n' + message['content'] + '<|im_end|>\\n' }}"
        "{% else %}"
        "{{ '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\\n' }}"
        "{% endif %}"
    )
    
    tokenizer.chat_template = template
    return tokenizer 