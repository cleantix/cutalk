"""ЦУтолк — бот-разговорщик для групповых чатов."""
import asyncio
import contextlib
import hashlib
import logging
import random
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.filters import Command
from aiogram.types import (
    InlineQuery,
    InlineQueryResultArticle,
    InlineQueryResultsButton,
    InputTextMessageContent,
    Message,
)

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

# Последний inline-запрос каждого пользователя — чтобы не генерировать на
# промежуточные версии недопечатанной фразы
_inline_latest: dict[int, str] = {}
_inline_cache: "OrderedDict[str, str]" = OrderedDict()
# Генерации, которые прямо сейчас выполняются: кэш наполняется только по
# завершении, а на CPU это секунды — без этого один и тот же текст успевает
# уйти в очередь дважды
_inline_inflight: dict[str, asyncio.Task] = {}


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
        # Именно реплай на исходное сообщение: в живом чате за те секунды,
        # что идёт генерация, успевают написать ещё, и ответ без привязки
        # повисает непонятно к чему.
        # allow_sending_without_reply — на случай, если сообщение успели удалить.
        await message.reply(text, allow_sending_without_reply=True)
    except Exception:
        log.exception("chat=%s: не смог отправить сообщение", chat_id)
        return

    storage.add_line(chat_id, config.BOT_NAME, text)


@dp.message(Command("ping"))
async def cmd_ping(message: Message) -> None:
    await message.answer("тут я")


@dp.message(F.chat.type == "private", F.text)
async def on_private_message(message: Message) -> None:
    """В личке каждое сообщение адресовано боту, кубик не кидаем."""
    text = (message.text or "").strip()
    if not text:
        return
    storage.add_line(message.chat.id, display_name(message), text)
    log.info("chat=%s (лс): сообщение %r", message.chat.id, text)
    if config.PRIVATE_ALWAYS_REPLY:
        await decide_and_reply(message, "личка")


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


def _cache_get(key: str) -> str | None:
    value = _inline_cache.get(key)
    if value is not None:
        _inline_cache.move_to_end(key)
    return value


def _cache_put(key: str, value: str) -> None:
    _inline_cache[key] = value
    _inline_cache.move_to_end(key)
    while len(_inline_cache) > config.INLINE_CACHE_SIZE:
        _inline_cache.popitem(last=False)


@dp.inline_query()
async def on_inline_query(query: InlineQuery) -> None:
    """Вызов вида "@botname текст" из любого чата, даже там, где бота нет."""
    if not config.INLINE_ENABLED:
        return

    text = (query.query or "").strip()
    user_id = query.from_user.id

    if not text:
        await query.answer(
            [],
            cache_time=1,
            is_personal=True,
            button=InlineQueryResultsButton(
                text=f"Напишите текст — {config.BOT_NAME} его продолжит",
                start_parameter="inline_help",
            ),
        )
        return

    reply = _cache_get(text)
    if reply is None:
        # Telegram шлёт запрос на каждое нажатие клавиши, а генерация на CPU
        # занимает секунды — ждём паузы в наборе и бросаем устаревшие запросы
        _inline_latest[user_id] = text
        await asyncio.sleep(config.INLINE_DEBOUNCE)
        if _inline_latest.get(user_id) != text:
            log.info("inline: запрос %r устарел, пропускаю", text)
            return

        name = query.from_user.full_name or query.from_user.username or "Человек"
        task = _inline_inflight.get(text)
        mine = task is None
        if mine:
            log.info("inline: генерирую для %s: %r", name, text)
            task = asyncio.create_task(generate_reply([f"{name}: {text}"]))
            _inline_inflight[text] = task
        else:
            log.info("inline: %r уже генерируется, жду тот же результат", text)
        try:
            reply = await task
        except Exception:
            log.exception("inline: ошибка генерации для %r", text)
            return
        finally:
            if mine:
                _inline_inflight.pop(text, None)
        if not reply:
            log.info("inline: пустой результат для %r", text)
            return
        _cache_put(text, reply)
        if mine:
            log.info("inline: результат %r", reply)

    result = InlineQueryResultArticle(
        id=hashlib.md5(f"{text}|{reply}".encode()).hexdigest(),
        title="Generated message",
        description=reply,
        input_message_content=InputTextMessageContent(message_text=reply),
    )
    with contextlib.suppress(Exception):
        # запрос живёт недолго; если протух — молча забываем
        await query.answer([result], cache_time=0, is_personal=True)


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
