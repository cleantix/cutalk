"""Скользящая история по чатам + список известных чатов на диске."""
import json
import logging
import os
import tempfile
from collections import defaultdict, deque

from . import config

log = logging.getLogger(__name__)

# chat_id -> deque(["Имя: текст", ...])
_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=config.HISTORY_MAXLEN))


def add_line(chat_id: int, name: str, text: str) -> None:
    _history[chat_id].append(f"{name}: {text}")


def get_history(chat_id: int) -> list[str]:
    return list(_history[chat_id])


def has_history(chat_id: int) -> bool:
    return len(_history[chat_id]) > 0


def load_chats() -> set[int]:
    try:
        with open(config.CHATS_FILE, encoding="utf-8") as f:
            return {int(x) for x in json.load(f)}
    except FileNotFoundError:
        return set()
    except Exception:
        log.exception("Не смог прочитать %s, начинаю с пустого списка", config.CHATS_FILE)
        return set()


def save_chats(chat_ids: set[int]) -> None:
    """Атомарная запись, чтобы не получить битый файл при падении."""
    try:
        directory = os.path.dirname(os.path.abspath(config.CHATS_FILE))
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(sorted(chat_ids), f)
        os.replace(tmp, config.CHATS_FILE)
    except Exception:
        log.exception("Не смог сохранить список чатов")
