"""Загрузка модели и генерация ответов. Всё синхронно и на CPU."""
import logging
import os
import re
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
        self._check_adapter()

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

    @staticmethod
    def _check_adapter() -> None:
        """Понятная ошибка вместо HFValidationError: transformers принимает
        несуществующий локальный путь за repo id на Hugging Face Hub."""
        path = config.ADAPTER_PATH
        if not os.path.isdir(path):
            raise RuntimeError(
                f"Папка с адаптером не найдена: {path!r} "
                f"(рабочая директория: {os.getcwd()}). "
                "Распакуйте adapter.tar.gz рядом с main.py. Учтите, что архив "
                "может развернуться в скрытую папку '.adapter' — переименуйте её "
                "в 'adapter' или укажите путь через ADAPTER_PATH."
            )
        missing = [
            f
            for f in ("adapter_config.json", "adapter_model.safetensors")
            if not os.path.isfile(os.path.join(path, f))
        ]
        if missing:
            raise RuntimeError(
                f"В папке {path!r} не хватает файлов адаптера: {', '.join(missing)}"
            )

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

        # Модель иногда выдаёт одни команды — тогда пробуем ещё раз,
        # иначе бот слишком часто молчал бы
        for attempt in range(1, config.GEN_MAX_ATTEMPTS + 1):
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
            raw = self.tokenizer.decode(generated, skip_special_tokens=True)
            text = clean_reply(raw)
            if text:
                return text
            log.info(
                "Попытка %d/%d дала пустой ответ после чистки (было: %r)",
                attempt, config.GEN_MAX_ATTEMPTS, raw.strip()[:80],
            )

        return ""


# Команда в Telegram — это латиница/цифры/подчёркивание после слеша,
# опционально с @имя_бота. Слеш должен стоять в начале или после пробела,
# чтобы не резать "и/или" и "10/10".
_COMMAND_RE = re.compile(r"(?:^|(?<=\s))/[A-Za-z][A-Za-z0-9_]*(?:@[A-Za-z0-9_]+)?\b")


def clean_reply(text: str) -> str:
    """Первая непустая строка без префикса с именем и без команд.

    Пустая строка на выходе означает «отвечать нечем» — вызывающий код
    либо перегенерирует, либо промолчит.
    """
    text = text.strip().split("\n")[0].strip()

    prefix = f"{config.BOT_NAME}:"
    while text.lower().startswith(prefix.lower()):
        text = text[len(prefix):].strip()

    text = _COMMAND_RE.sub(" ", text)
    return " ".join(text.split())
