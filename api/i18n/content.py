"""Localized command, feature, and prompt content.

Keep prose here instead of beside routing or rendering logic. Structural data such
as aliases, visibility, and categories remains in the modules that use it.
"""

from __future__ import annotations

from api.i18n import Locale, current_locale

COMMAND_DESCRIPTIONS: dict[str, dict[Locale, str]] = {
    "ask_ai": {"es": "te contesto cualquier gilada", "en": "ask me anything"},
    "config_command": {
        "es": "tocás la config del gordo y de los links",
        "en": "open all bot settings",
    },
    "language_command": {
        "es": "cambiás el idioma del bot [es|en]",
        "en": "change the bot language [es|en]",
    },
    "convert_base": {"es": "te paso números entre bases", "en": "convert numbers between bases"},
    "select_random": {
        "es": "elijo por vos entre opciones o números",
        "en": "pick an option or number for you",
    },
    "get_prices": {
        "es": "precios crypto por símbolo [moneda] [1h/24h/7d/30d]",
        "en": "crypto prices by symbol [currency] [1h/24h/7d/30d]",
    },
    "get_weather": {
        "es": "clima actual [ciudad o ubicación]",
        "en": "current weather [city or location]",
    },
    "get_dollar_rates": {
        "es": "cotizaciones del dolar [1h/6h/12h/24h/48h]",
        "en": "dollar exchange rates [1h/6h/12h/24h/48h]",
    },
    "get_oil_price": {
        "es": "te paso el precio del Brent y del WTI",
        "en": "Brent and WTI oil prices",
    },
    "get_stock_prices": {
        "es": "precios por símbolo o empresa [aapl tsla]",
        "en": "stock prices by symbol or company [aapl tsla]",
    },
    "get_polymarket_global_elections": {
        "es": "top 10 de elecciones globales en Polymarket por liquidez",
        "en": "top global Polymarket elections by liquidity",
    },
    "get_rulo": {
        "es": "te armo los rulos desde el oficial",
        "en": "calculate arbitrage from the official exchange rate",
    },
    "get_devo": {
        "es": "te calculo el arbitraje entre tarjeta y crypto",
        "en": "calculate card and crypto arbitrage",
    },
    "powerlaw": {
        "es": "te tiro el precio justo de btc según power law",
        "en": "Bitcoin power-law fair price",
    },
    "rainbow": {
        "es": "te tiro el precio justo de btc según rainbow chart",
        "en": "Bitcoin rainbow-chart fair price",
    },
    "satoshi": {"es": "te digo cuánto vale un satoshi", "en": "current value of one satoshi"},
    "get_timestamp": {"es": "timestamp unix actual", "en": "current Unix timestamp"},
    "convert_to_command": {
        "es": "te lo convierto en comando de telegram",
        "en": "convert text into a Telegram command",
    },
    "get_instance_name": {
        "es": "nombre de esta instancia del bot",
        "en": "name of this bot instance",
    },
    "get_help": {"es": "te muestro todos los comandos", "en": "show all commands"},
    "handle_transcribe": {
        "es": "te transcribo audio o describo imagen",
        "en": "transcribe audio or describe an image",
    },
    "handle_bcra_variables": {
        "es": "te tiro las variables económicas del bcra",
        "en": "show BCRA economic variables",
    },
    "topup_command": {
        "es": "cargás créditos IA con Telegram Stars por privado",
        "en": "add AI credits with Telegram Stars in private",
    },
    "balance_command": {"es": "te muestro tu saldo IA", "en": "show your AI balance"},
    "charges_command": {
        "es": "te muestro cuánto pagaste por cada uso de IA [cantidad]",
        "en": "show what each AI use cost [count]",
    },
    "transfer_command": {
        "es": "le pasás créditos tuyos al grupo",
        "en": "move your credits to the group",
    },
    "get_good_morning": {"es": "gif de buenos días", "en": "random good-morning GIF"},
    "get_good_night": {"es": "gif de buenas noches", "en": "random good-night GIF"},
    "task_command": {
        "es": "creá una tarea con texto o listá las existentes",
        "en": "create a task from text or list existing tasks",
    },
    "summary_command": {
        "es": "resumí la conversación [enfoque opcional]",
        "en": "summarize the conversation [optional focus]",
    },
}

FEATURE_TEXT: dict[str, dict[Locale, tuple[str, str]]] = {
    "ai_chat": {
        "es": (
            "chat ia",
            "te contesto mensajes normales; en grupos respondo si me mencionan, me responden, usan trigger random o mandan comando ia",
        ),
        "en": (
            "AI chat",
            "I answer normal messages; in groups I respond to mentions, replies, random triggers, and AI commands",
        ),
    },
    "web_search": {
        "es": (
            "búsqueda web nativa",
            "en mensajes normales puedo buscar en internet cuando hace falta",
        ),
        "en": ("web search", "I can search the web when a current answer needs it"),
    },
    "crypto": {
        "es": ("crypto prices", "precios crypto por ranking, símbolo, moneda base y variación"),
        "en": ("crypto prices", "crypto prices by ranking, symbol, base currency, and time window"),
    },
    "weather": {
        "es": ("clima", "clima actual para cualquier ciudad o ubicación"),
        "en": ("weather", "current weather for any city or location"),
    },
    "token_cards": {
        "es": (
            "token cards",
            "si el mensaje completo es un address Solana/EVM o un $ticker, mando card con chart/imagen, stats, socials, links y botones",
        ),
        "en": ("token cards", "send a Solana or EVM address, or a $ticker, for a market card"),
    },
    "dollar": {
        "es": ("dólar", "cotizaciones del dólar y variaciones por ventana"),
        "en": ("dollar", "dollar exchange rates and changes by time window"),
    },
    "stocks": {
        "es": ("acciones", "precios de acciones por símbolo o empresa desde Yahoo Finance"),
        "en": ("stocks", "stock prices by symbol or company from Yahoo Finance"),
    },
    "oil": {"es": ("petróleo", "precio Brent y WTI"), "en": ("oil", "Brent and WTI oil prices")},
    "bcra": {
        "es": ("bcra", "variables económicas del BCRA"),
        "en": ("BCRA", "economic variables from Argentina's central bank"),
    },
    "elections": {
        "es": ("elección", "top 10 de elecciones globales en Polymarket por liquidez"),
        "en": ("elections", "top global election markets on Polymarket by liquidity"),
    },
    "arbitrage": {
        "es": (
            "arbitrajes",
            "rulo desde oficial, arbitraje tarjeta/crypto, power law, rainbow chart y sats",
        ),
        "en": (
            "arbitrage",
            "official-rate, card/crypto, power-law, rainbow-chart, and satoshi tools",
        ),
    },
    "media": {
        "es": (
            "media",
            "transcribo voice/audio/video/video_note y describo fotos o stickers respondiendo al mensaje; también puedo procesar media cuando me hablan",
        ),
        "en": ("media", "transcribe audio and video or describe images and stickers"),
    },
    "links": {
        "es": (
            "links",
            "arreglo links de X/Twitter, Bluesky, Instagram y Reddit según config; leo metadata, tweets y transcripts de YouTube como contexto",
        ),
        "en": ("links", "fix supported social links and read linked content as context"),
    },
    "tasks": {
        "es": (
            "tareas",
            "agendo recordatorios y tareas recurrentes por lenguaje natural; cualquiera de los comandos lista sin texto y crea con texto",
        ),
        "en": ("tasks", "create reminders and recurring tasks with natural language"),
    },
    "memory": {
        "es": (
            "resúmenes y memoria",
            "resumo el chat, guardo resumen acumulado y recupero mensajes relevantes para responder con contexto",
        ),
        "en": ("summaries and memory", "summarize chats and retrieve relevant prior messages"),
    },
    "utilities": {
        "es": (
            "utilidades",
            "random, conversión de bases, comandos Telegram, timestamp e instancia",
        ),
        "en": (
            "utilities",
            "random selection, base conversion, Telegram commands, timestamps, and instance info",
        ),
    },
    "gifs": {
        "es": ("gifs", "gif random de buenos días o buenas noches"),
        "en": ("GIFs", "random good-morning and good-night GIFs"),
    },
    "config": {
        "es": (
            "config",
            "config por chat: idioma, links, followups, timezone, random replies y límite gratis por usuario/hora",
        ),
        "en": (
            "settings",
            "all chat settings, including language, links, timezone, replies, and group limits",
        ),
    },
    "language": {
        "es": ("idioma", "cambiá entre español e inglés"),
        "en": ("language", "switch between Spanish and English"),
    },
    "credits": {
        "es": (
            "créditos ia",
            "saldo, historial de gastos, topup con Telegram Stars y transferencia de créditos personales al grupo",
        ),
        "en": (
            "AI credits",
            "balance, expense history, Telegram Stars top-ups, and group transfers",
        ),
    },
    "credit_admin": {
        "es": ("admin créditos", "mint y log de créditos, solo admin"),
        "en": ("credit admin", "mint and inspect credits; admin only"),
    },
    "help": {
        "es": ("help", "muestro comandos y features"),
        "en": ("help", "show commands and features"),
    },
}

FEATURE_EXAMPLES: dict[str, dict[Locale, tuple[str, ...]]] = {
    "ai_chat": {"es": ("/gordo explicame esto",), "en": ("/gordo explain this",)},
    "web_search": {"es": ("buscá qué pasó con...",), "en": ("search what happened with...",)},
    "crypto": {
        "es": (
            "/prices btc eth xmr",
            "/prices 20",
            "/prices 100 in eur",
            "/prices btc 7d",
            "/prices stables",
        ),
        "en": (
            "/prices btc eth xmr",
            "/prices 20",
            "/prices 100 in eur",
            "/prices btc 7d",
            "/prices stables",
        ),
    },
    "weather": {"es": ("/clima Córdoba, Argentina",), "en": ("/weather London",)},
    "token_cards": {"es": ("J8PS...pump", "$GLORP"), "en": ("J8PS...pump", "$GLORP")},
    "dollar": {"es": ("/usd 1h",), "en": ("/usd 1h",)},
    "stocks": {
        "es": ("/acciones aapl tsla", "/acciones Mercado Libre"),
        "en": ("/stocks aapl tsla", "/stocks Mercado Libre"),
    },
    "arbitrage": {"es": ("/devo 0.5, 100",), "en": ("/devo 0.5, 100",)},
    "tasks": {
        "es": ("/tarea mañana recordame pagar el alquiler", "/tasks"),
        "en": ("/task tomorrow remind me to pay rent", "/tasks"),
    },
    "memory": {"es": ("/resumen focus en crypto",), "en": ("/summary focus on crypto",)},
    "utilities": {
        "es": ("/random pizza, carne, sushi", "/convertbase 101, 2, 10"),
        "en": ("/random pizza, steak, sushi", "/convertbase 101, 2, 10"),
    },
    "credits": {"es": ("/charges 10", "/transfer 1.5"), "en": ("/charges 10", "/transfer 1.5")},
}

WEATHER_DESCRIPTIONS: dict[int, dict[Locale, str]] = {
    0: {"es": "despejado", "en": "clear"},
    1: {"es": "mayormente despejado", "en": "mostly clear"},
    2: {"es": "parcialmente nublado", "en": "partly cloudy"},
    3: {"es": "nublado", "en": "cloudy"},
    45: {"es": "neblina", "en": "foggy"},
    48: {"es": "niebla", "en": "freezing fog"},
    51: {"es": "llovizna leve", "en": "light drizzle"},
    53: {"es": "llovizna moderada", "en": "moderate drizzle"},
    55: {"es": "llovizna intensa", "en": "heavy drizzle"},
    56: {"es": "llovizna helada leve", "en": "light freezing drizzle"},
    57: {"es": "llovizna helada intensa", "en": "heavy freezing drizzle"},
    61: {"es": "lluvia leve", "en": "light rain"},
    63: {"es": "lluvia moderada", "en": "moderate rain"},
    65: {"es": "lluvia intensa", "en": "heavy rain"},
    66: {"es": "lluvia helada leve", "en": "light freezing rain"},
    67: {"es": "lluvia helada intensa", "en": "heavy freezing rain"},
    71: {"es": "nevada leve", "en": "light snow"},
    73: {"es": "nevada moderada", "en": "moderate snow"},
    75: {"es": "nevada intensa", "en": "heavy snow"},
    77: {"es": "granizo", "en": "snow grains"},
    80: {"es": "lluvia leve intermitente", "en": "light rain showers"},
    81: {"es": "lluvia moderada intermitente", "en": "moderate rain showers"},
    82: {"es": "lluvia fuerte intermitente", "en": "heavy rain showers"},
    85: {"es": "nevada leve intermitente", "en": "light snow showers"},
    86: {"es": "nevada intensa intermitente", "en": "heavy snow showers"},
    95: {"es": "tormenta", "en": "thunderstorm"},
    96: {"es": "tormenta con granizo leve", "en": "thunderstorm with light hail"},
    99: {"es": "tormenta con granizo intenso", "en": "thunderstorm with heavy hail"},
}

CATEGORY_NAMES: dict[str, dict[Locale, str]] = {
    "ai": {"es": "ia", "en": "AI"},
    "markets": {"es": "mercado", "en": "markets"},
    "general": {"es": "general", "en": "general"},
    "media": {"es": "media", "en": "media"},
    "links": {"es": "links", "en": "links"},
    "productivity": {"es": "productividad", "en": "productivity"},
    "memory": {"es": "memoria", "en": "memory"},
    "utilities": {"es": "utilidades", "en": "utilities"},
    "settings": {"es": "config", "en": "settings"},
    "credits": {"es": "créditos", "en": "credits"},
    "admin": {"es": "admin", "en": "admin"},
}

HELP_TEXT: dict[Locale, dict[str, str | tuple[str, ...]]] = {
    "es": {
        "title": "esto es lo que sé hacer, boludo:",
        "example": "ejemplo",
        "admin": "solo admin",
        "capabilities": (
            "CAPACIDADES DEL BOT:",
            "- si el usuario pregunta qué podés hacer, respondé desde esta lista",
            "- no inventes comandos; /buscar y /search no existen",
            "- si existe comando exacto para algo, sugerilo con el comando exacto",
        ),
    },
    "en": {
        "title": "what I can do:",
        "example": "example",
        "admin": "admin only",
        "capabilities": (
            "BOT CAPABILITIES:",
            "- if the user asks what you can do, answer from this list",
            "- do not invent commands; /buscar and /search do not exist",
            "- when an exact command exists, suggest that exact command",
        ),
    },
}


def command_description(handler_name: str, locale: Locale) -> str:
    return COMMAND_DESCRIPTIONS[handler_name][locale]


def feature_text(key: str, locale: Locale | None = None) -> tuple[str, str]:
    return FEATURE_TEXT[key][locale or current_locale()]


def feature_examples(key: str, locale: Locale | None = None) -> tuple[str, ...]:
    translations = FEATURE_EXAMPLES.get(key)
    return translations[locale or current_locale()] if translations else ()


def weather_description(code: int, locale: Locale | None = None) -> str | None:
    translations = WEATHER_DESCRIPTIONS.get(code)
    return translations[locale or current_locale()] if translations else None


def category_name(key: str, locale: Locale | None = None) -> str:
    return CATEGORY_NAMES[key][locale or current_locale()]


def help_text(key: str, locale: Locale | None = None) -> str | tuple[str, ...]:
    return HELP_TEXT[locale or current_locale()][key]
