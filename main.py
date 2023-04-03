import hashlib
import json
import random
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from math import log
from os import environ
from typing import Dict, List, Tuple, Callable

import redis
import requests
from cryptography.fernet import Fernet
from flask import Request
from requests.exceptions import RequestException

import functions_framework


def _config_redis(host=None, port=None, password=None):
    """
    Configure a Redis client with the provided or default values and return it.
    """
    host = host or environ.get("REDIS_HOST", "localhost")
    port = port or environ.get("REDIS_PORT", 6379)
    password = password or environ.get("REDIS_PASSWORD", None)
    redis_client = redis.Redis(
        host=host, port=port, password=password, decode_responses=True)
    return redis_client


# request new data and save it in redis
def _set_new_data(request, timestamp, redis_client, request_hash, hourly_cache):
    api_url = request["api_url"]
    parameters = request["parameters"]
    headers = request["headers"]
    response = requests.get(api_url, params=parameters,
                            headers=headers, timeout=5)

    if response.status_code == 200:
        response_json = json.loads(response.text)
        redis_value = {"timestamp": timestamp, "data": response_json}
        redis_client.set(request_hash, json.dumps(redis_value))

        # if hourly_cache is True, save the data in redis with the current hour
        if hourly_cache:
            # get current date with hour
            current_hour = datetime.now().strftime("%Y-%m-%d-%H")
            redis_client.set(current_hour + request_hash,
                             json.dumps(redis_value))

        return redis_value
    else:
        return None


# get cached data from previous hour
def _get_cache_history(hours_ago, request_hash, redis_client):
    # subtract hours to current date
    timestamp = (datetime.now() - timedelta(hours=hours_ago)
                 ).strftime("%Y-%m-%d-%H")
    # get previous api data from redis cache
    cached_data = redis_client.get(timestamp + request_hash)

    if cached_data is None:
        return None
    else:
        cache_history = json.loads(cached_data)
        if cache_history is not None and "timestamp" not in cache_history:
            cache_history = None
        return cache_history


# generic proxy for caching any request
def _cached_requests(api_url, parameters, headers, expiration_time, hourly_cache=False, get_history=False):
    # create unique hash for the request
    arguments_dict = {"api_url": api_url,
                      "parameters": parameters,
                      "headers": headers}
    arguments_str = json.dumps(arguments_dict, sort_keys=True).encode()
    request_hash = hashlib.md5(arguments_str).hexdigest()

    # redis config
    redis_client = _config_redis()

    # check if environment variables for connecting to the redis backup are present
    if (
        "REDIS_HOST_BACKUP" in environ
        and "REDIS_PORT_BACKUP" in environ
        and "REDIS_PASSWORD_BACKUP" in environ
    ):
        redis_client_backup = _config_redis(
            environ.get("REDIS_HOST_BACKUP"),
            environ.get("REDIS_PORT_BACKUP"),
            environ.get("REDIS_PASSWORD_BACKUP"),
        )
    else:
        redis_client_backup = None

    # get previous api data from redis cache
    redis_response = redis_client.get(request_hash)

    if get_history is not False:
        cache_history = _get_cache_history(
            get_history, request_hash, redis_client)
        if cache_history is None and redis_client_backup is not None:
            cache_history = _get_cache_history(
                get_history, request_hash, redis_client_backup)
    else:
        cache_history = None

    # set current timestamp
    timestamp = int(time.time())

    # if there's no cached data request it
    if redis_response is None:
        new_data = _set_new_data(
            arguments_dict, timestamp, redis_client, request_hash, hourly_cache)

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
            new_data = _set_new_data(
                arguments_dict, timestamp, redis_client, request_hash, hourly_cache)

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
    """
    Given a message text, selects a random value or number based on the format.

    Args:
        msg_text (str): the input message text

    Returns:
        str: a random value or number, or an error message if the input is invalid
    """
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
def _get_api_or_cache_prices(convert_to):
    # coinmarketcap api config
    api_url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    parameters = {'start': '1', 'limit': '100', 'convert': convert_to}
    headers = {'Accepts': 'application/json',
               'X-CMC_PRO_API_KEY': environ.get("COINMARKETCAP_KEY")}

    cache_expiration_time = 300
    response = _cached_requests(
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
        coins = ["XAU", "USD", "EUR", "KRW", "GBP", "AUD", "BRL", "CAD", "CLP", "CNY", "COP", "CZK", "DKK", "HKD", "ISK", "INR", "ILS",
                 "JPY", "MXN", "TWD", "NZD", "PEN", "SGD", "SEK", "CHF", "UYU", "BTC", "SATS", "ETH", "XMR", "USDC", "USDT", "DAI", "BUSD"]
        convert_to = words[-1]
        if convert_to in coins:
            if convert_to == "SATS":
                convert_to_parameter = "BTC"
            else:
                convert_to_parameter = convert_to
            msg_text = msg_text.upper().replace(
                f"IN {convert_to}", "").strip()
        else:
            return f"{convert_to} is not allowed"

    # get prices from api or cache
    prices = _get_api_or_cache_prices(convert_to_parameter)

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
            coins.extend(["USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "USDD",
                         "GUSD", "DOC", "FRAX", "LUSD", "SUSD", "MIMATIC", "MIM", "MAI"])

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
            coin["quote"][convert_to_parameter
                          ]["price"] = coin["quote"][convert_to_parameter]["price"] * 100000000

        decimals = f"{coin['quote'][convert_to_parameter]['price']:.12f}".split(
            ".")[-1]
        zeros = len(decimals) - len(decimals.lstrip("0"))

        ticker = coin["symbol"]
        price = f"{coin['quote'][convert_to_parameter]['price']:.{zeros+4}f}".rstrip(
            "0").rstrip(".")
        percentage = f"{coin['quote'][convert_to_parameter]['percent_change_24h']:+.2f}".rstrip(
            "0").rstrip(".")
        line = f"{ticker}: {price} {convert_to} ({percentage}% 24hs)"

        if prices["data"][0]["symbol"] == coin["symbol"]:
            msg = line
        else:
            msg = f"""{msg}
{line}"""

    return msg


def _get_lowest(prices: Dict[str, Dict[str, float]]) -> float:
    lowest_price = float('inf')

    for exchange in prices:
        ask = float(prices[exchange]["totalAsk"])

        if ask == 0:
            continue

        if ask < lowest_price:
            lowest_price = ask

    return lowest_price


def _to_float(data):
    return {key: float(value) for key, value in data.items()}


def _add_derived_rates(data, base_key, multipliers):
    for name, multiplier in multipliers.items():
        data[name] = data[base_key] * multiplier
    return data


def _get_rate_history(data, key):
    return data.get(key) if key in data else None


def _sort_dollar_rates(dollar_rates, usdc_rates, dai_rates, usdt_rates):
    dollars = _to_float(dollar_rates["data"])
    derived_rates = {"solidario": 1.65, "tarjeta": 1.75, "qatar": 2}
    dollars = _add_derived_rates(dollars, "oficial", derived_rates)

    dollars["usdc"] = _get_lowest(usdc_rates["data"])
    dollars["dai"] = _get_lowest(dai_rates["data"])
    dollars["usdt"] = _get_lowest(usdt_rates["data"])

    dollars_history = {}
    if "history" in dollar_rates:
        dollars_history = _to_float(dollar_rates["history"]["data"])
        dollars_history = _add_derived_rates(
            dollars_history, "oficial", derived_rates)

        for rate_type in [("usdc", usdc_rates), ("dai", dai_rates), ("usdt", usdt_rates)]:
            if "history" in rate_type[1]:
                dollars_history[rate_type[0]] = _get_lowest(
                    rate_type[1]["history"]["data"])

    rate_names = [
        "oficial", "solidario", "tarjeta", "qatar", "mepgd30",
        "cclgd30", "ccb", "blue", "usdc", "dai", "usdt"
    ]

    rate_display_names = {
        "oficial": "Oficial", "solidario": "Solidario", "tarjeta": "Tarjeta", "qatar": "Qatar",
        "mepgd30": "MEP", "cclgd30": "CCL", "ccb": "Bitcoin", "blue": "Blue",
        "usdc": "USDC", "dai": "DAI", "usdt": "USDT"
    }

    sorted_dollar_rates = [
        {"name": rate_display_names[name], "price": dollars[name],
         "history": _get_rate_history(dollars_history, name)}
        for name in rate_names
    ]

    sorted_dollar_rates.sort(key=lambda x: x["price"])

    return sorted_dollar_rates


def _format_dollar_rates(dollar_rates: List[Dict], hours_ago: int) -> str:
    msg_lines = []
    for dollar in dollar_rates:
        price_formatted = f"{dollar['price']:.2f}".rstrip('0').rstrip('.')
        line = f"{dollar['name']}: {price_formatted}"
        if dollar["history"] is not None:
            percentage = (dollar['price'] / dollar["history"] - 1) * 100
            formatted_percentage = f"{percentage:+.2f}".rstrip('0').rstrip('.')
            line += f" ({formatted_percentage}% {hours_ago}hs)"
        msg_lines.append(line)

    return "\n".join(msg_lines)


def get_dollar_rates(msg_text: str) -> str:
    cache_expiration_time = 300
    hours_ago = int(msg_text) if msg_text.isdecimal() and int(
        msg_text) >= 0 else 24
    api_urls = [
        "https://criptoya.com/api/dolar",
        "https://criptoya.com/api/usdc/ars/1000",
        "https://criptoya.com/api/dai/ars/1000",
        "https://criptoya.com/api/usdt/ars/1000"
    ]

    with ThreadPoolExecutor(max_workers=5) as executor:
        api_results = [executor.submit(
            _cached_requests, url, None, None, cache_expiration_time, True, hours_ago) for url in api_urls]
        dollars, usdc, dai, usdt = [result.result() for result in api_results]

    sorted_dollar_rates = _sort_dollar_rates(dollars, usdc, dai, usdt)

    return _format_dollar_rates(sorted_dollar_rates, hours_ago)


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
        api_urls = [
            "https://criptoya.com/api/dolar",
            "https://criptoya.com/api/usdt/ars/1000"
        ]

        with ThreadPoolExecutor(max_workers=5) as executor:
            api_results = [executor.submit(
                _cached_requests, url, None, None, cache_expiration_time, True) for url in api_urls]
            dollars, usdt = [result.result() for result in api_results]

        dollars = _to_float(dollars["data"])
        dollars["usdt"] = _get_lowest(usdt["data"])

        qatar_tax = 2

        profit = -(fee * dollars["usdt"] + dollars["oficial"] -
                   dollars["usdt"]) / (dollars["oficial"] * qatar_tax)

        msg = f"""Profit: {f"{profit * 100:.2f}".rstrip("0").rstrip(".")}%

Fee: {f"{fee * 100:.2f}".rstrip("0").rstrip(".")}%
Oficial: {f"{dollars['oficial']:.2f}".rstrip("0").rstrip(".")}
USDT: {f"{dollars['usdt']:.2f}".rstrip("0").rstrip(".")}
Qatar: {f"{dollars['oficial'] * qatar_tax:.2f}".rstrip("0").rstrip(".")}"""

        if compra > 0:
            compra_ars = compra * (dollars["oficial"] * qatar_tax)
            compra_usdt = compra_ars / dollars["usdt"]
            ganancia_ars = compra_ars * profit
            ganancia_usdt = ganancia_ars / dollars["usdt"]
            msg = f"""{f"{compra:.2f}".rstrip("0").rstrip(".")} USD Qatar = {f"{compra_ars:.2f}".rstrip("0").rstrip(".")} ARS = {f"{compra_usdt:.2f}".rstrip("0").rstrip(".")} USDT
Ganarias {f"{ganancia_ars:.2f}".rstrip("0").rstrip(".")} ARS / {f"{ganancia_usdt:.2f}".rstrip("0").rstrip(".")} USDT
Total: {f"{compra_ars + ganancia_ars:.2f}".rstrip("0").rstrip(".")} ARS / {f"{compra_usdt + ganancia_usdt:.2f}".rstrip("0").rstrip(".")} USDT

{msg}"""

        return msg
    except ValueError:
        return "Invalid input. Usage: /devo <fee_percentage>[, <purchase_amount>]"


def rainbow(msg_text: str) -> str:
    today = datetime.now()
    since = datetime(day=9, month=1, year=2009)
    days_since = (today - since).days
    value = 10 ** (2.66167155005961 *
                   log(days_since) - 17.9183761889864)

    api_response = _get_api_or_cache_prices("USD")
    price = api_response["data"][0]["quote"]["USD"]["price"]

    percentage = ((price - value) / value)*100
    if percentage > 0:
        percentage_txt = f"{percentage:.2f}% overvalued"
    else:
        percentage_txt = f"{abs(percentage):.2f}% undervalued"

    msg = f"Today's Bitcoin theoretical value is {value:.2f} USD ({percentage_txt})"
    return msg


def convert_base(msg_text: str) -> str:
    """
    Convert a number from one base to another.

    Usage: /convertbase <number>, <base_from>, <base_to>

    Returns a string in the format "<number> in base <base_from> equals to <result> in base <base_to>",
    or an error message if the input is invalid.
    """
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
                digit_value = ord(digit.upper()) - ord('A') + 10
            value = value * base_from + digit_value
        while value > 0:
            digit_value = value % base_to
            if digit_value >= 10:
                digit = chr(digit_value - 10 + ord('A'))
            else:
                digit = str(digit_value)
            digits.append(digit)
            value //= base_to
        result = ''.join(reversed(digits))

        # Format output string
        return f"{number_str} in base {base_from} equals to {result} in base {base_to}"
    except ValueError:
        return "Invalid input. Base and number must be integers."


def get_timestamp(msg_text: str) -> str:
    return f"{int(time.time())}"


def convert_to_command(msg_text: str) -> str:
    # Convert to uppercase
    upper_text = msg_text.upper()

    # Replace standalone Ñ with ENIE, otherwise replace with NI
    replaced_text = re.sub(r'\bÑ\b', 'ENIE', upper_text)
    replaced_text = replaced_text.replace('Ñ', 'NI')

    # Normalize the text by removing accents and converting to ASCII
    normalized_text = unicodedata.normalize(
        'NFD', replaced_text).encode('ascii', 'ignore').decode('utf-8')

    # Replace specific punctuation marks with their respective words using str.translate
    punct_replacements = str.maketrans({
        ' ': '_',
        '?': '_SIGNODEPREGUNTA',
        '!': '_SIGNODEEXCLAMACION',
        '\n': '_',
    })
    translated_text = normalized_text.translate(punct_replacements)

    # Remove all characters except letters, numbers, and underscores
    alphanumeric_underscore = re.sub(r'[^A-Za-z0-9_]', '', translated_text)

    # Add a forward slash at the beginning
    command = '/' + alphanumeric_underscore

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

- /rainbow: Get the theoretical value of Bitcoin and its overvaluation or undervaluation percentage

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


def initialize_commands() -> Dict[str, Callable]:
    return {
        "/convertbase": convert_base,
        "/random": select_random,
        "/prices": get_prices,
        "/dolar": get_dollar_rates,
        "/devo": get_devo,
        "/rainbow": rainbow,
        "/time": get_timestamp,
        "/comando": convert_to_command,
        "/instance": get_instance_name,
        "/help": get_help,
    }


def handle_msg(token: str, message: Dict) -> str:
    message_text = message.get("text", "")
    message_id = message.get("message_id", "")
    chat_id = str(message["chat"]["id"])
    chat_type = str(message["chat"]["type"])
    first_name = str(message["from"]["first_name"])

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

    if command in commands:
        send_typing(token, chat_id)
        response_msg = commands[command](sanitized_message_text)
    elif not command.startswith("/") or command == "/ask":
        try:
            reply_to = str(message["reply_to_message"]["from"]["username"])
            if reply_to != environ.get("TELEGRAM_USERNAME") and bot_name not in message_text and not command.startswith("/ask"):
                return "ignored request"
        except KeyError:
            if chat_type != "private" and bot_name not in message_text and not command.startswith("/ask"):
                return "ignored request"

        send_typing(token, chat_id)
        response_msg = gen_random(first_name)

    send_msg(token, chat_id, response_msg, message_id)
    return "ok"


def decrypt_token(key: str, encrypted_token: str) -> str:
    fernet = Fernet(key.encode())
    return fernet.decrypt(encrypted_token.encode()).decode()


def set_telegram_webhook(decrypted_token: str, webhook_url: str, encrypted_token: str) -> bool:
    secret_token = hashlib.sha256(Fernet.generate_key()).hexdigest()
    parameters = {
        "url": f"{webhook_url}?token={encrypted_token}",
        "allowed_updates": ["message"],
        "secret_token": secret_token,
        "max_connections": 100
    }
    request_url = f"https://api.telegram.org/bot{decrypted_token}/setWebhook"
    telegram_response = requests.get(request_url, params=parameters, timeout=5)
    if telegram_response.status_code == 200:
        redis_client = _config_redis()
        redis_response = redis_client.set(
            "X-Telegram-Bot-Api-Secret-Token", secret_token)
        return bool(redis_response)
    else:
        return False


def handle_webhook_failure(decrypted_token: str, encrypted_token: str, error_message: str) -> bool:
    admin_report(decrypted_token, error_message)
    current_function_url = environ.get("CURRENT_FUNCTION_URL")
    webhook_result = set_telegram_webhook(
        decrypted_token, current_function_url, encrypted_token)
    return webhook_result


def verify_webhook(decrypted_token: str, encrypted_token: str) -> bool:
    main_function_url = environ.get("MAIN_FUNCTION_URL")
    parameters = {"update_webhook": "true", "token": encrypted_token}
    try:
        function_response = requests.get(
            main_function_url, params=parameters, timeout=5)
        function_response.raise_for_status()
    except RequestException as request_error:
        error_message = f"main webhook failed with error: {str(request_error)}"
        return handle_webhook_failure(decrypted_token, encrypted_token, error_message)
    return True


def is_secret_token_valid(request: Request) -> bool:
    secret_token = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
    redis_client = _config_redis()
    redis_secret_token = redis_client.get("X-Telegram-Bot-Api-Secret-Token")
    return redis_secret_token == secret_token


def process_request_parameters(request: Request, decrypted_token: str, encrypted_token: str) -> Tuple[str, int]:
    if request.args.get("check_webhook") == "true":
        if verify_webhook(decrypted_token, encrypted_token):
            return "webhook checked", 200
        else:
            return "webhook check error", 400

    if request.args.get("update_webhook") == "true":
        function_url = environ.get("CURRENT_FUNCTION_URL")
        if set_telegram_webhook(decrypted_token, function_url, encrypted_token):
            return "webhook updated", 200
        else:
            return "webhook update error", 400

    if not is_secret_token_valid(request):
        admin_report(decrypted_token, "wrong secret token")
        return "wrong secret token", 400

    request_json = request.get_json()
    if "message" not in request_json:
        return "not message", 200

    handle_msg(decrypted_token, request_json["message"])
    return "ok", 200


@functions_framework.http
def responder(request: Request) -> Tuple[str, int]:
    try:
        if request.args.get("update_dollars") == "true":
            get_dollar_rates("")
            return "dollars updated", 200

        encrypted_token = str(request.args.get("token"))
        key = environ.get("TELEGRAM_TOKEN_KEY")
        decrypted_token = decrypt_token(key, encrypted_token)
        token_hash = hashlib.sha256(decrypted_token.encode()).hexdigest()
        if token_hash != environ.get("TELEGRAM_TOKEN_HASH"):
            return "wrong token", 400

        response_message, status_code = process_request_parameters(
            request, decrypted_token, encrypted_token)
        return response_message, status_code
    except BaseException as error:
        print(f"error: {error}")
        return "error", 500
