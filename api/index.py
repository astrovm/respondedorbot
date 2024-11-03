import hashlib
import json
import random
import re
import time
import traceback
import unicodedata
from datetime import datetime, timedelta, timezone
from math import log
from os import environ
from typing import Dict, List, Tuple, Callable, Union
import redis
import requests
from cryptography.fernet import Fernet
from flask import Flask, Request, request
from requests.exceptions import RequestException
import emoji
from anthropic import Anthropic

def config_redis(host=None, port=None, password=None):
    host = host or environ.get("REDIS_HOST", "localhost")
    port = port or environ.get("REDIS_PORT", 6379)
    password = password or environ.get("REDIS_PASSWORD", None)
    redis_client = redis.Redis(host=host, port=port, password=password, decode_responses=True)
    return redis_client

# request new data and save it in redis
def set_new_data(request, timestamp, redis_client, request_hash, hourly_cache):
    api_url = request["api_url"]
    parameters = request["parameters"]
    headers = request["headers"]
    response = requests.get(api_url, params=parameters,headers=headers, timeout=5)

    if response.status_code == 200:
        response_json = json.loads(response.text)
        redis_value = {"timestamp": timestamp, "data": response_json}
        redis_client.set(request_hash, json.dumps(redis_value))

        # if hourly_cache is True, save the data in redis with the current hour
        if hourly_cache:
            # get current date with hour
            current_hour = datetime.now().strftime("%Y-%m-%d-%H")
            redis_client.set(current_hour + request_hash, json.dumps(redis_value))

        return redis_value
    else:
        return None

# get cached data from previous hour
def get_cache_history(hours_ago, request_hash, redis_client):
    # subtract hours to current date
    timestamp = (datetime.now() - timedelta(hours=hours_ago)).strftime("%Y-%m-%d-%H")
    # get previous api data from redis cache
    cached_data = redis_client.get(timestamp + request_hash)

    if cached_data is None:
        return None
    else:
        cache_history = json.loads(cached_data)
        if cache_history is not None and "timestamp" not in cache_history: cache_history = None
        return cache_history

# generic proxy for caching any request
def cached_requests(
    api_url, parameters, headers, expiration_time, hourly_cache=False, get_history=False
):
    # create unique hash for the request
    arguments_dict = {"api_url": api_url, "parameters": parameters, "headers": headers}
    arguments_str = json.dumps(arguments_dict, sort_keys=True).encode()
    request_hash = hashlib.md5(arguments_str).hexdigest()

    # redis config
    redis_client = config_redis()

    # get previous api data from redis cache
    redis_response = redis_client.get(request_hash)

    if get_history is not False:
        cache_history = get_cache_history(get_history, request_hash, redis_client)
    else:
        cache_history = None

    # set current timestamp
    timestamp = int(time.time())

    # if there's no cached data request it
    if redis_response is None:
        new_data = set_new_data(arguments_dict, timestamp, redis_client, request_hash, hourly_cache)

        if cache_history is not None:
            new_data["history"] = cache_history

        return new_data
    else:
        # loads cached data
        cached_data = json.loads(redis_response)
        cached_data_timestamp = int(cached_data["timestamp"])

        if cache_history is not None:
            cached_data["history"] = cache_history

        # get new data if cache is older than expiration_time
        if timestamp - cached_data_timestamp > expiration_time:
            new_data = set_new_data(
                arguments_dict, timestamp, redis_client, request_hash, hourly_cache
            )

            if cache_history is not None:
                new_data["history"] = cache_history

            if "timestamp" in new_data:
                return new_data
            else:
                return cached_data
        else:
            return cached_data

def gen_random(name: str) -> str:
    time.sleep(random.uniform(0, 1))

    rand_res = random.randint(0, 1)
    rand_name = random.randint(0, 2)

    if rand_res:
        msg = "si"
    else:
        msg = "no"

    if rand_name == 1:
        msg = f"{msg} boludo"
        time.sleep(random.uniform(0, 1))
    elif rand_name == 2:
        msg = f"{msg} {name}"
        time.sleep(random.uniform(0, 1))

    return msg

def select_random(msg_text: str) -> str:
    try:
        # Check if the input is a comma-separated list of values
        values = [v.strip() for v in msg_text.split(",")]
        if len(values) >= 2:
            return random.choice(values)
    except ValueError:
        pass

    try:
        # Check if the input is a range of numbers
        start, end = [int(v.strip()) for v in msg_text.split("-")]
        if start < end:
            return str(random.randint(start, end))
    except ValueError:
        pass

    # Return an error message for invalid inputs
    return "Please enter a valid input. Use ',' to separate values or '-' to specify a range."


# check if prices are cached before request to api
def get_api_or_cache_prices(convert_to):
    # coinmarketcap api config
    api_url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    parameters = {"start": "1", "limit": "100", "convert": convert_to}
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": environ.get("COINMARKETCAP_KEY"),
    }

    cache_expiration_time = 300
    response = cached_requests(
        api_url, parameters, headers, cache_expiration_time)

    return response["data"]


# get crypto pices from coinmarketcap
def get_prices(msg_text: str) -> str:
    prices_number = 0
    # when the user asks for sats we need to ask for btc to the api and convert later
    convert_to = "USD"
    # here we keep the currency that we'll request to the api
    convert_to_parameter = "USD"

    # check if the user wants to convert the prices
    if "IN " in msg_text.upper():
        words = msg_text.upper().split()
        coins = [
            "XAU", "USD", "EUR", "KRW", "GBP", "AUD", "BRL", "CAD", "CLP", "CNY",
            "COP", "CZK", "DKK", "HKD", "ISK", "INR", "ILS", "JPY", "MXN", "TWD",
            "NZD", "PEN", "SGD", "SEK", "CHF", "UYU", "BTC", "SATS", "ETH", "XMR",
            "USDC", "USDT", "DAI", "BUSD",
        ]
        convert_to = words[-1]
        if convert_to in coins:
            if convert_to == "SATS":
                convert_to_parameter = "BTC"
            else:
                convert_to_parameter = convert_to
            msg_text = msg_text.upper().replace(f"IN {convert_to}", "").strip()
        else:
            return f"{convert_to} is not allowed"

    # get prices from api or cache
    prices = get_api_or_cache_prices(convert_to_parameter)

    # check if the user requested a custom number of prices
    if msg_text != "":
        numbers = msg_text.upper().replace(" ", "").split(",")

        for number in numbers:
            try:
                number = int(float(number))
                if number > prices_number:
                    prices_number = number
            except ValueError:
                # ignore items which aren't integers
                pass

    # check if the user requested a list of coins
    if msg_text.upper().isupper():
        new_prices = []
        coins = msg_text.upper().replace(" ", "").split(",")

        if "STABLES" in coins or "STABLECOINS" in coins:
            coins.extend([
                "BUSD", "DAI", "DOC", "EURT", "FDUSD", "FRAX", "GHO", "GUSD",
                "LUSD", "MAI", "MIM", "MIMATIC", "NUARS", "PAXG", "PYUSD", "RAI",
                "SUSD", "TUSD", "USDC", "USDD", "USDM", "USDP", "USDT", "UXD",
                "XAUT", "XSGD",
            ])

        for coin in prices["data"]:
            symbol = coin["symbol"].upper().replace(" ", "")
            name = coin["name"].upper().replace(" ", "")

            if symbol in coins or name in coins:
                new_prices.append(coin)
            elif prices_number > 0 and coin in prices["data"][:prices_number]:
                new_prices.append(coin)

        if not new_prices:
            return "ponzis no laburo"

        prices_number = len(new_prices)
        prices["data"] = new_prices

    # default number of prices
    if prices_number < 1:
        prices_number = 10

    # generate the message to answer the user
    msg = ""
    for coin in prices["data"][:prices_number]:
        if convert_to == "SATS":
            coin["quote"][convert_to_parameter]["price"] = (
                coin["quote"][convert_to_parameter]["price"] * 100000000
            )

        decimals = f"{coin['quote'][convert_to_parameter]['price']:.12f}".split(".")[-1]
        zeros = len(decimals) - len(decimals.lstrip("0"))

        ticker = coin["symbol"]
        price = f"{coin['quote'][convert_to_parameter]['price']:.{zeros+4}f}".rstrip("0").rstrip(".")
        percentage = f"{coin['quote'][convert_to_parameter]['percent_change_24h']:+.2f}".rstrip("0").rstrip(".")
        line = f"{ticker}: {price} {convert_to} ({percentage}% 24hs)"

        if prices["data"][0]["symbol"] == coin["symbol"]:
            msg = line
        else:
            msg = f"{msg}\n{line}"

    return msg

def sort_dollar_rates(dollar_rates):
    dollars = dollar_rates["data"]

    sorted_dollar_rates = [
        {
            "name": "Oficial",
            "price": dollars["oficial"]["price"],
            "history": dollars["oficial"]["variation"],
        },
        {
            "name": "Tarjeta",
            "price": dollars["tarjeta"]["price"],
            "history": dollars["tarjeta"]["variation"],
        },
        {
            "name": "MEP",
            "price": dollars["mep"]["al30"]["ci"]["price"],
            "history": dollars["mep"]["al30"]["ci"]["variation"],
        },
        {
            "name": "CCL",
            "price": dollars["ccl"]["al30"]["ci"]["price"],
            "history": dollars["ccl"]["al30"]["ci"]["variation"],
        },
        {
            "name": "Blue",
            "price": dollars["blue"]["ask"],
            "history": dollars["blue"]["variation"],
        },
        {
            "name": "Bitcoin",
            "price": dollars["cripto"]["ccb"]["ask"],
            "history": dollars["cripto"]["ccb"]["variation"],
        },
        {
            "name": "USDC",
            "price": dollars["cripto"]["usdc"]["ask"],
            "history": dollars["cripto"]["usdc"]["variation"],
        },
        {
            "name": "USDT",
            "price": dollars["cripto"]["usdt"]["ask"],
            "history": dollars["cripto"]["usdt"]["variation"],
        },
    ]

    sorted_dollar_rates.sort(key=lambda x: x["price"])

    return sorted_dollar_rates

def format_dollar_rates(dollar_rates: List[Dict], hours_ago: int) -> str:
    msg_lines = []
    for dollar in dollar_rates:
        price_formatted = f"{dollar['price']:.2f}".rstrip("0").rstrip(".")
        line = f"{dollar['name']}: {price_formatted}"
        if dollar["history"] is not None:
            percentage = dollar["history"]
            formatted_percentage = f"{percentage:+.2f}".rstrip("0").rstrip(".")
            line += f" ({formatted_percentage}% {hours_ago}hs)"
        msg_lines.append(line)

    return "\n".join(msg_lines)
def get_dollar_rates(msg_text: str) -> str:
    cache_expiration_time = 300
    # hours_ago = int(msg_text) if msg_text.isdecimal() and int(msg_text) >= 0 else 24
    api_url = "https://criptoya.com/api/dolar"

    dollars = cached_requests(api_url, None, None, cache_expiration_time, True)

    sorted_dollar_rates = sort_dollar_rates(dollars)

    return format_dollar_rates(sorted_dollar_rates, 24)
def get_devo(msg_text: str) -> str:
    try:
        fee = 0
        compra = 0

        if "," in msg_text:
            numbers = msg_text.replace(" ", "").split(",")
            fee = float(numbers[0]) / 100
            if len(numbers) > 1:
                compra = float(numbers[1])
        else:
            fee = float(msg_text) / 100

        if fee != fee or fee > 1 or compra != compra or compra < 0:
            return "Invalid input. Fee should be between 0 and 100, and purchase amount should be a positive number."

        cache_expiration_time = 300

        api_url = "https://criptoya.com/api/dolar"

        dollars = cached_requests(
            api_url, None, None, cache_expiration_time, True)

        usdt_ask = float(dollars["data"]["cripto"]["usdt"]["ask"])
        usdt_bid = float(dollars["data"]["cripto"]["usdt"]["bid"])
        usdt = (usdt_ask + usdt_bid) / 2
        oficial = float(dollars["data"]["oficial"]["price"])
        tarjeta = oficial * 1.6

        profit = -(fee * usdt + oficial - usdt) / tarjeta

        msg = f"""Profit: {f"{profit * 100:.2f}".rstrip("0").rstrip(".")}%

Fee: {f"{fee * 100:.2f}".rstrip("0").rstrip(".")}%
Oficial: {f"{oficial:.2f}".rstrip("0").rstrip(".")}
USDT: {f"{usdt:.2f}".rstrip("0").rstrip(".")}
Tarjeta: {f"{tarjeta:.2f}".rstrip("0").rstrip(".")}"""

        if compra > 0:
            compra_ars = compra * tarjeta
            compra_usdt = compra_ars / usdt
            ganancia_ars = compra_ars * profit
            ganancia_usdt = ganancia_ars / usdt
            msg = f"""{f"{compra:.2f}".rstrip("0").rstrip(".")} USD Tarjeta = {f"{compra_ars:.2f}".rstrip("0").rstrip(".")} ARS = {f"{compra_usdt:.2f}".rstrip("0").rstrip(".")} USDT
Ganarias {f"{ganancia_ars:.2f}".rstrip("0").rstrip(".")} ARS / {f"{ganancia_usdt:.2f}".rstrip("0").rstrip(".")} USDT
Total: {f"{compra_ars + ganancia_ars:.2f}".rstrip("0").rstrip(".")} ARS / {f"{compra_usdt + ganancia_usdt:.2f}".rstrip("0").rstrip(".")} USDT

{msg}"""

        return msg
    except ValueError:
        return "Invalid input. Usage: /devo <fee_percentage>[, <purchase_amount>]"

def powerlaw(msg_text: str) -> str:
    today = datetime.now(timezone.utc)
    since = datetime(day=4, month=1, year=2009).replace(tzinfo=timezone.utc)
    days_since = (today - since).days
    value = 10 ** (-17) * days_since ** 5.82

    api_response = get_api_or_cache_prices("USD")
    price = api_response["data"][0]["quote"]["USD"]["price"]

    percentage = ((price - value) / value) * 100
    if percentage > 0:
        percentage_txt = f"{percentage:.2f}% overvalued"
    else:
        percentage_txt = f"{abs(percentage):.2f}% undervalued"

    msg = f"Today's Bitcoin Power Law theoretical value is {value:.2f} USD ({percentage_txt})"
    return msg

def rainbow(msg_text: str) -> str:
    today = datetime.now(timezone.utc)
    since = datetime(day=9, month=1, year=2009).replace(tzinfo=timezone.utc)
    days_since = (today - since).days
    value = 10 ** (2.66167155005961 * log(days_since) - 17.9183761889864)

    api_response = get_api_or_cache_prices("USD")
    price = api_response["data"][0]["quote"]["USD"]["price"]

    percentage = ((price - value) / value) * 100
    if percentage > 0:
        percentage_txt = f"{percentage:.2f}% overvalued"
    else:
        percentage_txt = f"{abs(percentage):.2f}% undervalued"

    msg = f"Today's Bitcoin Rainbow theoretical value is {value:.2f} USD ({percentage_txt})"
    return msg
def convert_base(msg_text: str) -> str:
    try:
        # Parse input
        input_parts = msg_text.split(",")
        if len(input_parts) != 3:
            return "Invalid input. Usage: /convertbase <number>, <base_from>, <base_to>"
        number_str, base_from_str, base_to_str = map(str.strip, input_parts)
        base_from, base_to = map(int, (base_from_str, base_to_str))

        # Validate input
        if not all(c.isalnum() for c in number_str):
            return "Invalid input. Number must be alphanumeric."
        if not 2 <= base_from <= 36:
            return f"Invalid input. Base from '{base_from_str}' must be between 2 and 36."
        if not 2 <= base_to <= 36:
            return f"Invalid input. Base to '{base_to_str}' must be between 2 and 36."

        # Convert input to output base
        digits = []
        value = 0
        for digit in number_str:
            if digit.isdigit():
                digit_value = int(digit)
            else:
                digit_value = ord(digit.upper()) - ord("A") + 10
            value = value * base_from + digit_value
        while value > 0:
            digit_value = value % base_to
            if digit_value >= 10:
                digit = chr(digit_value - 10 + ord("A"))
            else:
                digit = str(digit_value)
            digits.append(digit)
            value //= base_to
        result = "".join(reversed(digits))

        # Format output string
        return f"{number_str} in base {base_from} equals to {result} in base {base_to}"
    except ValueError:
        return "Invalid input. Base and number must be integers."

def get_timestamp(msg_text: str) -> str:
    return f"{int(time.time())}"
def convert_to_command(msg_text: str) -> str:
    # Add spaces surrounding each emoji and convert it to its textual representation
    emoji_text = emoji.replace_emoji(
        msg_text, replace=lambda chars, data_dict: f" {data_dict['es']} "
    )

    # Convert to uppercase and replace Ñ
    replaced_ni_text = re.sub(
        r"\bÑ\b", "ENIE", emoji_text.upper()).replace("Ñ", "NI")

    # Normalize the text and remove consecutive spaces
    single_spaced_text = re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFD", replaced_ni_text)
        .encode("ascii", "ignore")
        .decode("utf-8"),
    )

    # Replace consecutive dots and specific punctuation marks
    translated_punctuation = re.sub(
        r"\.{3}", "_PUNTOSSUSPENSIVOS_", single_spaced_text
    ).translate(
        str.maketrans(
            {
                " ": "_",
                "\n": "_",
                "?": "_SIGNODEPREGUNTA_",
                "!": "_SIGNODEEXCLAMACION_",
                ".": "_PUNTO_",
            }
        )
    )

    # Remove non-alphanumeric characters and consecutive, trailing and leading underscores
    cleaned_text = re.sub(
        r"^_+|_+$",
        "",
        re.sub(r"[^A-Za-z0-9_]", "",
               re.sub(r"_+", "_", translated_punctuation)),
    )

    # If there are no remaining characters after processing, return an error
    if not cleaned_text:
        return "Invalid input. Usage: /comando <text>"

    # Add a forward slash at the beginning
    command = f"/{cleaned_text}"
    return command

def get_help(msg_text: str) -> str:
    return """
Available commands:

- /ask question: Returns the answer to the question

- /comando something: Convert the input to a command

- /convertbase 101, 2, 10: Convert a number from one base to another (e.g., binary 101 to decimal)

- /dolar: Dollar prices in Argentina

- /instance: Returns the name of the instance where the bot is running

- /prices: Prices of the top 10 cryptos in USD
- /prices in btc: Prices of the top 10 cryptos in BTC
- /prices 20: Prices of the top 20 cryptos in USD
- /prices 100 in eur: Prices of the top 100 cryptos in EUR
- /prices btc, eth, xmr: Prices of Bitcoin, Ethereum, and Monero in USD
- /prices dai in sats: Price of DAI in Satoshis
- /prices stables: Prices of stablecoins in USD

- /random pizza, meat, sushi: Chooses between the options
- /random 1-10: Returns a random number between 1 and 10

- /powerlaw: Get the theoretical value of Bitcoin Power Law and its overvaluation or undervaluation percentage
- /rainbow: Get the theoretical value of Bitcoin Rainbow and its overvaluation or undervaluation percentage

- /time: Returns the current Unix timestamp
"""

def get_instance_name(msg_text: str) -> str:
    return environ.get("FRIENDLY_INSTANCE_NAME")

def send_typing(token: str, chat_id: str) -> None:
    parameters = {"chat_id": chat_id, "action": "typing"}
    url = f"https://api.telegram.org/bot{token}/sendChatAction"
    requests.get(url, params=parameters, timeout=5)

def send_msg(token: str, chat_id: str, msg: str, msg_id: str = "") -> None:
    parameters = {"chat_id": chat_id, "text": msg}
    if msg_id != "":
        parameters["reply_to_message_id"] = msg_id
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.get(url, params=parameters, timeout=5)

def admin_report(token: str, message: str) -> None:
    instance_name = environ.get("FRIENDLY_INSTANCE_NAME")
    formatted_message = f"Admin report from {instance_name}: {message}"
    admin_chat_id = environ.get("ADMIN_CHAT_ID")
    send_msg(token, admin_chat_id, formatted_message)

def ask_claude(msg_text: str, first_name: str = "", username: str = "", chat_type: str = "") -> str:
    """Send a message to Claude and return the response in Atendedor style"""
    try:
        # Initialize Anthropic client with API key
        anthropic = Anthropic(api_key=environ.get("ANTHROPIC_API_KEY"))

        # Add context about the user and chat
        user_context = f"""
        Contexto del mensaje:
        - Nombre del usuario: {first_name}
        - Username: {username}
        - Tipo de chat: {chat_type}
        """

        # Add context to make Claude respond like el gordo
        taringuero_context = f"""
        Sos el gordo, un respondedor de boludos. Te dicen gordo pero sos el mismo atendedor de boludos del video.
        
        REGLAS IMPORTANTES:
        1. Respondé con UNA SOLA FRASE de hasta 140 caracteres, sin punto final
        2. Sos un tipo que se las sabe todas:
           - Te gusta explicar las cosas cuando te preguntan bien
           - A veces bardeás pero sin pasarte
           - En el fondo sos un tipazo que ayuda
           - A veces podés usar el nombre/username del que pregunta
        3. Si te preguntan algo técnico o difícil:
           - La mayoría de las veces contestás porque sabés todo
           - Explicás de forma simple y directa
        4. Cuando NO querés contestar, usá:
           - tomatelá
           - no te doy bola
           - preguntale a otro
           - quién te conoce?
           - me importa un carajo
           - atiendo boludos
           - ni en pedo
           - raja de acá
        5. Para respuestas burlonas usá:
           - kjjj
           - baiteado
           - domado
        6. IMPORTANTE: 
           - No uses comillas ni emojis ni exclamaciones ni punto final
           - NO USES más de una palabra de lunfardo por frase
           - Mantené el espíritu del video original pero sin exagerar
           - No te hagas el superado
        
        {user_context}
        
        RECORDÁ: Una sola frase de hasta 140 caracteres, sin comillas ni punto final.
        
        Respondé a esto: {msg_text}"""

        # Create a message and get response from Claude
        message = anthropic.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=140,
            messages=[{
                "role": "user",
                "content": taringuero_context
            }]
        )

        return message.content[0].text

    except Exception as e:
        return f"Se cayo el sistema: {str(e)}"

def initialize_commands() -> Dict[str, Callable]:
    return {
        "/ask": ask_claude,
        "/convertbase": convert_base,
        "/random": select_random,
        "/prices": get_prices,
        "/precios": get_prices,
        "/precio": get_prices,
        "/presios": get_prices,
        "/presio": get_prices,
        "/dolar": get_dollar_rates,
        "/dollar": get_dollar_rates,
        "/devo": get_devo,
        "/powerlaw": powerlaw,
        "/rainbow": rainbow,
        "/time": get_timestamp,
        "/comando": convert_to_command,
        "/command": convert_to_command,
        "/instance": get_instance_name,
        "/help": get_help,
    }

def save_message_to_redis(chat_id: str, message_id: str, text: str, redis_client: redis.Redis) -> None:
    """Save a message to Redis with expiration time"""
    key = f"msg:{chat_id}:{message_id}"
    redis_client.set(key, text, ex=7200)  # expire after 2 hours instead of 1

def get_message_from_redis(chat_id: str, message_id: str, redis_client: redis.Redis) -> str:
    """Get a message from Redis"""
    key = f"msg:{chat_id}:{message_id}"
    return redis_client.get(key)

def get_conversation_context(message: Dict, redis_client: redis.Redis) -> str:
    """Build context from previous messages"""
    context = []
    chat_id = str(message["chat"]["id"])
    
    # If replying to a message, get that message and its context
    if "reply_to_message" in message:
        reply_msg = message["reply_to_message"]
        reply_id = str(reply_msg["message_id"])
        
        # Get the original message from Redis
        original_msg = get_message_from_redis(chat_id, reply_id, redis_client)
        if original_msg:
            context.append(f"Mensaje previo del bot: {original_msg}")
            
            # Get the message that the original message was replying to (if any)
            if "reply_to_message" in reply_msg:
                previous_id = str(reply_msg["reply_to_message"]["message_id"])
                previous_msg = get_message_from_redis(chat_id, previous_id, redis_client)
                if previous_msg:
                    context.append(f"Mensaje anterior del usuario: {previous_msg}")
    
    # Add instructions for Claude to maintain consistency
    if context:
        context.append("\nInstrucción: Mantené consistencia con tus respuestas anteriores y el contexto de la conversación.")
    
    return "\n".join(context) if context else ""

def handle_msg(token: str, message: Dict) -> str:
    try:
        message_text = (
            message.get("text")
            or message.get("caption")
            or message.get("poll", {}).get("question")
            or ""
        )
        message_id = message.get("message_id", "")
        chat_id = str(message["chat"]["id"])
        chat_type = str(message["chat"]["type"])
        first_name = str(message["from"]["first_name"])
        username = str(message["from"].get("username", ""))

        # Initialize Redis client
        redis_client = config_redis()
        
        # Save current message to Redis
        if message_text:
            save_message_to_redis(chat_id, str(message_id), message_text, redis_client)

        # Get conversation context
        conversation_context = get_conversation_context(message, redis_client)

        response_msg = ""
        split_message = message_text.strip().split(" ")
        command = split_message[0].lower()
        sanitized_message_text = message_text.replace(split_message[0], "").strip()

        commands = initialize_commands()

        bot_name = f"@{environ.get('TELEGRAM_USERNAME')}"
        if bot_name in command:
            command = command.replace(bot_name, "")

        if command == "/comando" and not sanitized_message_text:
            if "reply_to_message" in message and "text" in message["reply_to_message"]:
                sanitized_message_text = message["reply_to_message"]["text"]

        # Check if message mentions "el gordo" or variations
        gordo_mentions = ["el gordo", "gordo", "gordito"]
        should_respond = any(mention in message_text.lower() for mention in gordo_mentions)

        if command in commands:
            if command == "/ask":
                full_context = f"{conversation_context}\n\nPregunta actual: {sanitized_message_text}" if conversation_context else sanitized_message_text
                response_msg = ask_claude(full_context, first_name, username, chat_type)
            else:
                response_msg = commands[command](sanitized_message_text)
        elif not command.startswith("/") and (
            should_respond 
            or chat_type == "private" 
            or bot_name in message_text 
            or (
                "reply_to_message" in message 
                and message["reply_to_message"]["from"]["username"] == environ.get("TELEGRAM_USERNAME")
            )
        ):
            send_typing(token, chat_id)
            full_context = f"{conversation_context}\n\nPregunta actual: {message_text}" if conversation_context else message_text
            response_msg = ask_claude(full_context, first_name, username, chat_type)
        else:
            return "ignored request"

        # Save bot's response to Redis
        if response_msg:
            save_message_to_redis(chat_id, "bot_" + str(message_id), response_msg, redis_client)

        send_msg(token, chat_id, response_msg, message_id)
        return "ok"
    except BaseException as error:
        print(f"Error from handle_msg: {error}")
        return "Error from handle_msg", 500

def decrypt_token(key: str, encrypted_token: str) -> str:
    fernet = Fernet(key.encode())
    return fernet.decrypt(encrypted_token.encode()).decode()

def get_telegram_webhook_info(decrypted_token: str) -> Dict[str, Union[str, dict]]:
    request_url = f"https://api.telegram.org/bot{decrypted_token}/getWebhookInfo"
    try:
        telegram_response = requests.get(request_url, timeout=5)
        telegram_response.raise_for_status()
    except RequestException as request_error:
        return {"error": str(request_error)}
    return telegram_response.json()["result"]

def set_telegram_webhook(
    decrypted_token: str, webhook_url: str, encrypted_token: str
) -> bool:
    secret_token = hashlib.sha256(Fernet.generate_key()).hexdigest()
    parameters = {
        "url": f"{webhook_url}?token={encrypted_token}",
        "allowed_updates": '["message"]',
        "secret_token": secret_token,
        "max_connections": 100,
    }
    request_url = f"https://api.telegram.org/bot{decrypted_token}/setWebhook"
    try:
        telegram_response = requests.get(
            request_url, params=parameters, timeout=5)
        telegram_response.raise_for_status()
    except RequestException:
        return False
    redis_client = config_redis()
    redis_response = redis_client.set(
        "X-Telegram-Bot-Api-Secret-Token", secret_token)
    return bool(redis_response)

def verify_webhook(decrypted_token: str, encrypted_token: str) -> bool:
    def set_main_webhook() -> bool:
        return set_telegram_webhook(decrypted_token, main_function_url, encrypted_token)

    webhook_info = get_telegram_webhook_info(decrypted_token)
    if "error" in webhook_info:
        return False

    main_function_url = environ.get("MAIN_FUNCTION_URL")
    current_function_url = environ.get("CURRENT_FUNCTION_URL")
    main_webhook_url = f"{main_function_url}?token={encrypted_token}"
    current_webhook_url = f"{current_function_url}?token={encrypted_token}"

    if main_function_url != current_function_url:
        try:
            function_response = requests.get(main_function_url, timeout=5)
            function_response.raise_for_status()
        except RequestException as request_error:
            if webhook_info.get("url") != current_webhook_url:
                error_message = f"Main webhook failed with error: {str(request_error)}"
                admin_report(decrypted_token, error_message)
                return set_telegram_webhook(
                    decrypted_token, current_function_url, encrypted_token
                )
            return True
    elif webhook_info.get("url") != main_webhook_url:
        set_main_webhook_success = set_main_webhook()
        if set_main_webhook_success:
            admin_report(decrypted_token, "Main webhook is up again")
        else:
            admin_report(decrypted_token, "Failed to set main webhook")
        return set_main_webhook_success

    return True

def is_secret_token_valid(request: Request) -> bool:
    secret_token = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
    redis_client = config_redis()
    redis_secret_token = redis_client.get("X-Telegram-Bot-Api-Secret-Token")
    return redis_secret_token == secret_token

def process_request_parameters(
    request: Request, decrypted_token: str, encrypted_token: str
) -> Tuple[str, int]:
    try:
        check_webhook = request.args.get("check_webhook") == "true"
        update_webhook = request.args.get("update_webhook") == "true"

        if check_webhook:
            return (
                ("Webhook checked", 200)
                if verify_webhook(decrypted_token, encrypted_token)
                else ("Webhook check error", 400)
            )

        if update_webhook:
            function_url = environ.get("CURRENT_FUNCTION_URL")
            return (
                ("Webhook updated", 200)
                if set_telegram_webhook(decrypted_token, function_url, encrypted_token)
                else ("Webhook update error", 400)
            )

        if not is_secret_token_valid(request):
            admin_report(decrypted_token, "Wrong secret token")
            return "Wrong secret token", 400

        request_json = request.get_json(silent=True)
        message = request_json.get("message")

        if not message:
            return "No message", 200

        handle_msg(decrypted_token, message)
        return "Ok", 200
    except BaseException as error:
        print(f"Error from process_request_parameters: {error}")
        return "Error from process_request_parameters", 500


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def responder() -> Tuple[str, int]:
    try:
        if request.args.get("update_dollars") == "true":
            get_dollar_rates("")
            return "Dollars updated", 200

        token = request.args.get("token")
        if not token:
            return "No token", 200

        encrypted_token = str(token)
        key = environ.get("TELEGRAM_TOKEN_KEY")
        decrypted_token = decrypt_token(key, encrypted_token)
        token_hash = hashlib.sha256(decrypted_token.encode()).hexdigest()

        if token_hash != environ.get("TELEGRAM_TOKEN_HASH"):
            return "Wrong token", 400

        response_message, status_code = process_request_parameters(
            request, decrypted_token, encrypted_token
        )
        return response_message, status_code
    except BaseException as error:
        traceback_info = traceback.format_exc()
        print(f"Error from responder: {error}")
        print(traceback_info)
        return "Error from responder", 500
