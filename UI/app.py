import streamlit as st
import torch
import os
import sys
from pathlib import Path

root_dir = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, root_dir)

from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from chat_template import get_chat_template
import asyncio

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        for stop_ids in self.stop_token_ids:
            if input_ids[0][-len(stop_ids):].tolist() == stop_ids:
                return True
        return False

st.set_page_config(
    page_title="Генератор python кода",
    page_icon="🐍",
    layout="wide"
)

st.title("Генератор python кода")
st.markdown("Генерация python кода с помощью tuned Qwen модели")

@st.cache_resource(show_spinner=True)
def load_model():
    try:
        base_model_id = os.getenv("BASE_MODEL_ID", "unsloth/Qwen3-0.6B-Base")
        adapter_path = os.getenv("MODEL_ID", "dxnay/Qwen3-0.6B-Base-3_epochs_tuned")

        st.info(f"Загрузка базовой модели: {base_model_id}")
        st.info(f"Загрузка адаптера: {adapter_path}")

        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            trust_remote_code=True,
            use_fast=True
        )

        tokenizer = get_chat_template(tokenizer)

        model = PeftModel.from_pretrained(
            model, 
            adapter_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {str(e)}")
        st.exception(e)
        raise e

def generate_code(prompt, model, tokenizer, max_tokens=512, temperature=0.7):
    try:
        prompt_text = f"<|im_start|>system\nYou are a helpful AI programming assistant. Generate only Python code without any explanations.\n<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
        
        stop_token_ids = [
            tokenizer.encode("<|im_end|>", add_special_tokens=False),
            tokenizer.encode("\n<|im", add_special_tokens=False),
        ]
        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                top_k=40,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria
            )

        response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        response = response.split("<|im_end|>")[0].strip()
        return response
    except Exception as e:
        st.error(f"Ошибка генерации: {str(e)}")
        st.exception(e)
        return None

st.sidebar.title("Настройки")
st.sidebar.markdown("### Параметры генерации")
temperature = st.sidebar.slider("Температура", 0.1, 1.0, 0.7, 0.1, 
                              help="Чем выше значение, тем более творческие, но менее предсказуемые результаты")
max_tokens = st.sidebar.slider("Максимальная длина", 128, 2048, 512, 128,
                             help="Максимальное количество токенов в генерируемом коде")

task_description = st.text_area(
    "Опишите, какой код нужно сгенерировать",
    "Напиши функцию для нахождения n-го числа Фибоначчи."
)

if st.button("Сгенерировать код"):
    with st.spinner("Загрузка модели и генерация кода..."):
        try:
            model, tokenizer = load_model()
            if model and tokenizer:
                generated_code = generate_code(
                    f"Write Python code for the following task: {task_description}",
                    model, 
                    tokenizer,
                    max_tokens,
                    temperature
                )
                if generated_code:
                    st.code(generated_code, language="python")
        except Exception as e:
            st.error("Не удалось сгенерировать код. Попробуйте еще раз.")
            st.error(f"Детали ошибки: {str(e)}")
            st.exception(e)

st.markdown("---")
st.markdown("""
### О модели

Это fine-tuned версия модели Qwen для Python задач и 
модель была обучена на 20к данных
""")
