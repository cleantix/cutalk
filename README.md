# ЦУтолк

Telegram-бот-разговорщик для групповых чатов. Продолжает переписку в стиле участников
через Vikhr-Qwen-2.5-1.5B-Instruct + LoRA-адаптер. Инференс — на CPU, без CUDA.

## Когда бот пишет

1. **Реплай на его сообщение** — отвечает всегда.
2. **Упоминание** имени `ЦУтолк` или `@username` бота в тексте — отвечает всегда.
3. **Случайно** — с вероятностью `REPLY_PROBABILITY` (по умолчанию 0.15) на любое
   другое сообщение в чате.
4. **Сам по себе** — раз в `PROACTIVE_INTERVAL_HOURS` часов (по умолчанию 12)
   пишет в каждый чат, где уже накопилась история.

Контекст — скользящее окно из последних `HISTORY_MAXLEN` (8) сообщений на чат,
в формате `Имя: текст`. Своя реплика добавляется в историю как `ЦУтолк: текст`,
но в сам Telegram уходит без префикса — имя и так видно по аккаунту бота.

## ⚠️ Обязательно: выключить Privacy Mode

Без этого бот получает только команды и прямые упоминания, а не все сообщения чата —
пункты 3 и 4 работать не будут.

@BotFather → `/mybots` → выбрать бота → **Bot Settings** → **Group Privacy** →
**Turn off**. Затем **перезайти ботом в группу** (удалить и добавить заново) —
иначе настройка не применится к уже существующему членству.

## Локальный запуск

```bash
git clone <repo> cutalk && cd cutalk
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Положить адаптер рядом с `main.py`:

```bash
tar -xzf adapter.tar.gz     # получится ./adapter с adapter_config.json,
                            # adapter_model.safetensors и файлами токенизатора
```

Настроить окружение:

```bash
cp .env.example .env
$EDITOR .env                # вписать BOT_TOKEN
python main.py
```

Первый запуск скачает базовую модель с HuggingFace (~3 ГБ) — это займёт время.
Проверка живости: `/ping` в чате.

## Деплой на VPS (systemd)

Требования: 2+ ГБ RAM (модель в bf16 ≈ 3 ГБ, лучше 4 ГБ или swap), Python 3.10+.

```bash
sudo useradd -r -m -d /opt/cutalk -s /usr/sbin/nologin cutalk
sudo -u cutalk git clone <repo> /opt/cutalk
cd /opt/cutalk
sudo -u cutalk python3 -m venv .venv
sudo -u cutalk .venv/bin/pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# адаптер и конфиг
sudo -u cutalk tar -xzf adapter.tar.gz -C /opt/cutalk
sudo -u cutalk cp .env.example .env
sudo -u cutalk $EDITOR .env      # BOT_TOKEN, TORCH_THREADS=<число ядер>
sudo chmod 600 /opt/cutalk/.env

sudo cp cutalk-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cutalk-bot
```

Логи и управление:

```bash
journalctl -u cutalk-bot -f
sudo systemctl restart cutalk-bot
```

`Restart=always` поднимает процесс после падения через 10 секунд.

## Структура

| Файл | Что делает |
|---|---|
| `main.py` | точка входа |
| `cutalk/config.py` | все настройки из окружения / `.env` |
| `cutalk/generator.py` | загрузка модели, merge LoRA, генерация |
| `cutalk/storage.py` | история по чатам, список чатов на диске |
| `cutalk/bot.py` | хендлеры aiogram, логика решений, проактивный цикл |
| `cutalk-bot.service` | unit для systemd |

## Заметки по производительности

Генерация 128 токенов на CPU идёт секунды-десятки секунд, поэтому:

- она вынесена в `run_in_executor` и не блокирует event loop;
- одновременно выполняется только одна генерация (общий лок) — на слабой VPS
  параллельные запросы всё равно упрутся в CPU;
- перед ответом бот шлёт `typing`, чтобы пауза не выглядела зависанием;
- `TORCH_THREADS` стоит выставить равным числу ядер VPS;
- если bf16 на вашем CPU медленный (старые процессоры без AVX512-BF16),
  имеет смысл перейти на float32 при достатке RAM или на квантованную сборку.

Список чатов для проактивных сообщений хранится в `chats.json` и пополняется,
как только в чате появляется первое сообщение после старта бота.
