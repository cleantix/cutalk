"""Загрузка модели и генерация ответов. Всё синхронно и на CPU."""
import logging
import threading

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config

log = logging.getLogger(__name__)

# Генерация на CPU — тяжёлая и не потокобезопасная в смысле нагрузки:
# держим один общий лок, чтобы не запускать две генерации одновременно.
_gen_lock = threading.Lock()


class Generator:
    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        if config.TORCH_THREADS > 0:
            torch.set_num_threads(config.TORCH_THREADS)
            log.info("torch threads: %d", config.TORCH_THREADS)

        log.info("Загружаю токенизатор из %s", config.ADAPTER_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(config.ADAPTER_PATH)

        log.info("Загружаю базовую модель %s (CPU, bfloat16)", config.BASE_MODEL)
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                config.BASE_MODEL, dtype=torch.bfloat16
            )
        except TypeError:
            # transformers < 4.56: параметр назывался torch_dtype
            base_model = AutoModelForCausalLM.from_pretrained(
                config.BASE_MODEL, torch_dtype=torch.bfloat16
            )

        log.info("Прикручиваю LoRA-адаптер из %s", config.ADAPTER_PATH)
        model = PeftModel.from_pretrained(base_model, config.ADAPTER_PATH)

        log.info("merge_and_unload()...")
        model = model.merge_and_unload()
        model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = model
        log.info("Модель готова")

    def build_prompt(self, history_lines: list[str]) -> str:
        """История вида ["Имя: текст", ...] -> строка промпта через chat template."""
        messages = [
            {"role": "system", "content": config.DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(history_lines)},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, history_lines: list[str]) -> str:
        """Блокирующая генерация. Вызывать только через run_in_executor."""
        if self.model is None:
            raise RuntimeError("Модель не загружена")

        prompt = self.build_prompt(history_lines)
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with _gen_lock, torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=True,
                temperature=config.TEMPERATURE,
                top_k=config.TOP_K,
                top_p=config.TOP_P,
                repetition_penalty=config.REPETITION_PENALTY,
                pad_token_id=self.tokenizer.pad_token_id,
                stop_strings=["\n"],
                tokenizer=self.tokenizer,
            )

        generated = output[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return clean_reply(text)


def clean_reply(text: str) -> str:
    """Берём первую непустую строку и снимаем возможный префикс с именем."""
    text = text.strip().split("\n")[0].strip()
    prefix = f"{config.BOT_NAME}:"
    while text.lower().startswith(prefix.lower()):
        text = text[len(prefix):].strip()
    return text
