"""Конфигурация бота. Всё читается из окружения; .env подхватывается здесь."""
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv необязателен, на проде переменные даёт systemd
    pass

BOT_TOKEN = os.getenv("BOT_TOKEN", "")

BASE_MODEL = os.getenv("BASE_MODEL", "Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "./adapter")

BOT_NAME = os.getenv("BOT_NAME", "ЦУтолк")

DEFAULT_SYSTEM_PROMPT = (
    "Тебе показывают фрагмент переписки в чате — сообщения разных участников "
    "с их именами. Напиши только текст следующего сообщения, без имени, "
    "который естественно продолжил бы разговор — с той же интонацией, юмором "
    "и манерой, что и в остальном чате."
)

# Сколько последних сообщений держим в контексте на чат
HISTORY_MAXLEN = int(os.getenv("HISTORY_MAXLEN", "8"))

# Вероятность ответа на обычное сообщение (не реплай и не упоминание)
REPLY_PROBABILITY = float(os.getenv("REPLY_PROBABILITY", "0.15"))

# Период самостоятельных сообщений в чат, в часах
PROACTIVE_INTERVAL_HOURS = float(os.getenv("PROACTIVE_INTERVAL_HOURS", "12"))
# Не писать самому, если в чате совсем нет истории
PROACTIVE_ENABLED = os.getenv("PROACTIVE_ENABLED", "1") not in ("0", "false", "False")

# Файл, где храним id чатов, в которых бот состоит (для проактивных сообщений)
CHATS_FILE = os.getenv("CHATS_FILE", "./chats.json")

# Параметры генерации
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_K = int(os.getenv("TOP_K", "20"))
TOP_P = float(os.getenv("TOP_P", "0.8"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.1"))

# Количество потоков torch на CPU (0 = не трогать)
TORCH_THREADS = int(os.getenv("TORCH_THREADS", "0"))
