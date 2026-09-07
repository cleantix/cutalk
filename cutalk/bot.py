"""ЦУтолк — бот-разговорщик для групповых чатов."""
import asyncio
import contextlib
import logging
import random
import sys
from concurrent.futures import ThreadPoolExecutor

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.filters import Command
from aiogram.types import Message

from . import config, storage
from .generator import Generator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("cutalk")

GROUP_TYPES = {"group", "supergroup"}

dp = Dispatcher()
generator = Generator()
# Одна генерация за раз: модель на CPU, параллелить нечем
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gen")
known_chats: set[int] = set()


def display_name(message: Message) -> str:
    user = message.from_user
    if user is None:
        return "Аноним"
    if user.full_name:
        return user.full_name
    return user.username or str(user.id)


async def generate_reply(history: list[str]) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, generator.generate, history)


def remember_chat(chat_id: int) -> None:
    if chat_id not in known_chats:
        known_chats.add(chat_id)
        storage.save_chats(known_chats)
        log.info("Новый чат в списке: %s", chat_id)


async def decide_and_reply(message: Message, reason: str) -> None:
    chat_id = message.chat.id
    history = storage.get_history(chat_id)
    if not history:
        return

    log.info("chat=%s: отвечаю (%s), контекст из %d строк", chat_id, reason, len(history))
    try:
        with contextlib.suppress(Exception):
            await message.bot.send_chat_action(chat_id, ChatAction.TYPING)
        text = await generate_reply(history)
    except Exception:
        log.exception("chat=%s: ошибка генерации", chat_id)
        return

    if not text:
        log.info("chat=%s: генерация вернула пустую строку, молчу", chat_id)
        return

    log.info("chat=%s: сгенерирован ответ: %r", chat_id, text)
    try:
        await message.answer(text)
    except Exception:
        log.exception("chat=%s: не смог отправить сообщение", chat_id)
        return

    storage.add_line(chat_id, config.BOT_NAME, text)


@dp.message(Command("ping"))
async def cmd_ping(message: Message) -> None:
    await message.answer("тут я")


@dp.message(F.chat.type.in_(GROUP_TYPES), F.text)
async def on_group_message(message: Message) -> None:
    chat_id = message.chat.id
    text = (message.text or "").strip()
    if not text:
        return

    remember_chat(chat_id)
    name = display_name(message)
    storage.add_line(chat_id, name, text)
    log.info("chat=%s: сообщение от %s: %r", chat_id, name, text)

    me = await message.bot.me()
    is_reply_to_bot = (
        message.reply_to_message is not None
        and message.reply_to_message.from_user is not None
        and message.reply_to_message.from_user.id == me.id
    )
    lowered = text.lower()
    is_mention = (
        config.BOT_NAME.lower() in lowered
        or (me.username is not None and f"@{me.username.lower()}" in lowered)
    )

    if is_reply_to_bot:
        reason = "реплай на бота"
    elif is_mention:
        reason = "упоминание имени"
    elif random.random() < config.REPLY_PROBABILITY:
        reason = f"случайно, p={config.REPLY_PROBABILITY}"
    else:
        log.info("chat=%s: пропускаю", chat_id)
        return

    await decide_and_reply(message, reason)


async def proactive_loop(bot: Bot) -> None:
    """Раз в PROACTIVE_INTERVAL_HOURS пишем сам в каждый известный чат."""
    interval = config.PROACTIVE_INTERVAL_HOURS * 3600
    log.info("Проактивный режим включён: раз в %s ч", config.PROACTIVE_INTERVAL_HOURS)
    while True:
        await asyncio.sleep(interval)
        for chat_id in list(known_chats):
            history = storage.get_history(chat_id)
            if not history:
                log.info("chat=%s: пустая история, самому писать нечего", chat_id)
                continue
            try:
                text = await generate_reply(history)
            except Exception:
                log.exception("chat=%s: ошибка проактивной генерации", chat_id)
                continue
            if not text:
                continue
            try:
                await bot.send_message(chat_id, text)
            except Exception:
                log.exception("chat=%s: не смог отправить проактивное сообщение", chat_id)
                continue
            storage.add_line(chat_id, config.BOT_NAME, text)
            log.info("chat=%s: проактивное сообщение: %r", chat_id, text)


async def main() -> None:
    if not config.BOT_TOKEN:
        log.error("BOT_TOKEN не задан (переменная окружения или .env)")
        sys.exit(1)

    known_chats.update(storage.load_chats())
    log.info("Известных чатов: %d", len(known_chats))

    # Грузим модель в отдельном потоке, чтобы не блокировать старт event loop
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(executor, generator.load)

    bot = Bot(token=config.BOT_TOKEN)
    task = None
    if config.PROACTIVE_ENABLED:
        task = asyncio.create_task(proactive_loop(bot))

    try:
        await dp.start_polling(bot)
    finally:
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await bot.session.close()


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
