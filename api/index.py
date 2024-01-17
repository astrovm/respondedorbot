import hashlib
import json
import random
import re
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from math import log
from os import environ
from typing import Dict, List, Tuple, Callable, Union
import redis
import requests
from cryptography.fernet import Fernet
from flask import Flask, Request, request
from requests.exceptions import RequestException
import emoji


def config_redis(host=None, port=None, password=None):
    host = host or environ.get("REDIS_HOST", "localhost")
    port = port or environ.get("REDIS_PORT", 6379)
    password = password or environ.get("REDIS_PASSWORD", None)
    redis_client = redis.Redis(
        host=host, port=port, password=password, decode_responses=True
    )
    return redis_client


# request new data and save it in redis
def set_new_data(request, timestamp, redis_client, request_hash, hourly_cache):
    api_url = request["api_url"]
    parameters = request["parameters"]
    headers = request["headers"]
    response = requests.get(api_url, params=parameters, headers=headers, timeout=5)

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
        if cache_history is not None and "timestamp" not in cache_history:
            cache_history = None
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
        new_data = set_new_data(
            arguments_dict, timestamp, redis_client, request_hash, hourly_cache
        )

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
    response = cached_requests(api_url, parameters, headers, cache_expiration_time)

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
            "XAU",
            "USD",
            "EUR",
            "KRW",
            "GBP",
            "AUD",
            "BRL",
            "CAD",
            "CLP",
            "CNY",
            "COP",
            "CZK",
            "DKK",
            "HKD",
            "ISK",
            "INR",
            "ILS",
            "JPY",
            "MXN",
            "TWD",
            "NZD",
            "PEN",
            "SGD",
            "SEK",
            "CHF",
            "UYU",
            "BTC",
            "SATS",
            "ETH",
            "XMR",
            "USDC",
            "USDT",
            "DAI",
            "BUSD",
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
            coins.extend(
                [
                    "USDT",
                    "USDC",
                    "BUSD",
                    "DAI",
                    "TUSD",
                    "USDP",
                    "USDD",
                    "GUSD",
                    "DOC",
                    "FRAX",
                    "LUSD",
                    "SUSD",
                    "MIMATIC",
                    "MIM",
                    "MAI",
                ]
            )

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
        price = f"{coin['quote'][convert_to_parameter]['price']:.{zeros+4}f}".rstrip(
            "0"
        ).rstrip(".")
        percentage = (
            f"{coin['quote'][convert_to_parameter]['percent_change_24h']:+.2f}".rstrip(
                "0"
            ).rstrip(".")
        )
        line = f"{ticker}: {price} {convert_to} ({percentage}% 24hs)"

        if prices["data"][0]["symbol"] == coin["symbol"]:
            msg = line
        else:
            msg = f"""{msg}
{line}"""

    return msg


def get_lowest(prices: Dict[str, Dict[str, float]]) -> float:
    lowest_price = float("inf")

    for exchange in prices:
        ask = float(prices[exchange]["totalAsk"])

        if ask == 0:
            continue

        if ask < lowest_price:
            lowest_price = ask

    return lowest_price


def to_float(data):
    return {key: float(value) for key, value in data.items()}


def add_derived_rates(data, base_key, multipliers):
    for name, multiplier in multipliers.items():
        data[name] = data[base_key] * multiplier
    return data


def get_rate_history(data, key):
    return data.get(key) if key in data else None


def sort_dollar_rates(dollar_rates, usdc_rates, dai_rates, usdt_rates):
    dollars = to_float(dollar_rates["data"])
    derived_rates = {"tarjeta": 1.6}
    dollars = add_derived_rates(dollars, "oficial", derived_rates)

    dollars["usdc"] = get_lowest(usdc_rates["data"])
    dollars["dai"] = get_lowest(dai_rates["data"])
    dollars["usdt"] = get_lowest(usdt_rates["data"])

    dollars_history = {}
    if "history" in dollar_rates:
        dollars_history = to_float(dollar_rates["history"]["data"])
        dollars_history = add_derived_rates(dollars_history, "oficial", derived_rates)

        for rate_type in [
            ("usdc", usdc_rates),
            ("dai", dai_rates),
            ("usdt", usdt_rates),
        ]:
            if "history" in rate_type[1]:
                dollars_history[rate_type[0]] = get_lowest(
                    rate_type[1]["history"]["data"]
                )

    rate_names = [
        "oficial",
        "tarjeta",
        "mep",
        "ccl",
        "ccb",
        "blue",
        "usdc",
        "dai",
        "usdt",
    ]

    rate_display_names = {
        "oficial": "Oficial",
        "tarjeta": "Tarjeta",
        "mep": "MEP",
        "ccl": "CCL",
        "ccb": "Bitcoin",
        "blue": "Blue",
        "usdc": "USDC",
        "dai": "DAI",
        "usdt": "USDT",
    }

    sorted_dollar_rates = [
        {
            "name": rate_display_names[name],
            "price": dollars[name],
            "history": get_rate_history(dollars_history, name),
        }
        for name in rate_names
    ]

    sorted_dollar_rates.sort(key=lambda x: x["price"])

    return sorted_dollar_rates


def format_dollar_rates(dollar_rates: List[Dict], hours_ago: int) -> str:
    msg_lines = []
    for dollar in dollar_rates:
        price_formatted = f"{dollar['price']:.2f}".rstrip("0").rstrip(".")
        line = f"{dollar['name']}: {price_formatted}"
        if dollar["history"] is not None:
            percentage = (dollar["price"] / dollar["history"] - 1) * 100
            formatted_percentage = f"{percentage:+.2f}".rstrip("0").rstrip(".")
            line += f" ({formatted_percentage}% {hours_ago}hs)"
        msg_lines.append(line)

    return "\n".join(msg_lines)


def get_dollar_rates(msg_text: str) -> str:
    cache_expiration_time = 300
    hours_ago = int(msg_text) if msg_text.isdecimal() and int(msg_text) >= 0 else 24
    api_urls = [
        "https://criptoya.com/api/dolar",
        "https://criptoya.com/api/usdc/ars/1000",
        "https://criptoya.com/api/dai/ars/1000",
        "https://criptoya.com/api/usdt/ars/1000",
    ]

    with ThreadPoolExecutor(max_workers=5) as executor:
        api_results = [
            executor.submit(
                cached_requests, url, None, None, cache_expiration_time, True, hours_ago
            )
            for url in api_urls
        ]
        dollars, usdc, dai, usdt = [result.result() for result in api_results]

    sorted_dollar_rates = sort_dollar_rates(dollars, usdc, dai, usdt)

    return format_dollar_rates(sorted_dollar_rates, hours_ago)


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
            "https://criptoya.com/api/usdt/ars/1000",
        ]

        with ThreadPoolExecutor(max_workers=5) as executor:
            api_results = [
                executor.submit(
                    cached_requests, url, None, None, cache_expiration_time, True
                )
                for url in api_urls
            ]
            dollars, usdt = [result.result() for result in api_results]

        dollars = to_float(dollars["data"])
        dollars["usdt"] = get_lowest(usdt["data"])

        tarjeta_tax = 1.6

        profit = -(fee * dollars["usdt"] + dollars["oficial"] - dollars["usdt"]) / (
            dollars["oficial"] * tarjeta_tax
        )

        msg = f"""Profit: {f"{profit * 100:.2f}".rstrip("0").rstrip(".")}%

Fee: {f"{fee * 100:.2f}".rstrip("0").rstrip(".")}%
Oficial: {f"{dollars['oficial']:.2f}".rstrip("0").rstrip(".")}
USDT: {f"{dollars['usdt']:.2f}".rstrip("0").rstrip(".")}
Tarjeta: {f"{dollars['oficial'] * tarjeta_tax:.2f}".rstrip("0").rstrip(".")}"""

        if compra > 0:
            compra_ars = compra * (dollars["oficial"] * tarjeta_tax)
            compra_usdt = compra_ars / dollars["usdt"]
            ganancia_ars = compra_ars * profit
            ganancia_usdt = ganancia_ars / dollars["usdt"]
            msg = f"""{f"{compra:.2f}".rstrip("0").rstrip(".")} USD Tarjeta = {f"{compra_ars:.2f}".rstrip("0").rstrip(".")} ARS = {f"{compra_usdt:.2f}".rstrip("0").rstrip(".")} USDT
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
    value = 10 ** (2.66167155005961 * log(days_since) - 17.9183761889864)

    api_response = get_api_or_cache_prices("USD")
    price = api_response["data"][0]["quote"]["USD"]["price"]

    percentage = ((price - value) / value) * 100
    if percentage > 0:
        percentage_txt = f"{percentage:.2f}% overvalued"
    else:
        percentage_txt = f"{abs(percentage):.2f}% undervalued"

    msg = f"Today's Bitcoin theoretical value is {value:.2f} USD ({percentage_txt})"
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
            return (
                f"Invalid input. Base from '{base_from_str}' must be between 2 and 36."
            )
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
    replaced_ni_text = re.sub(r"\bÑ\b", "ENIE", emoji_text.upper()).replace("Ñ", "NI")

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
        re.sub(r"[^A-Za-z0-9_]", "", re.sub(r"_+", "_", translated_punctuation)),
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

- /rainbow: Get the theoretical value of Bitcoin and its overvaluation or undervaluation percentage

- /time: Returns the current Unix timestamp
    """


def donation(msg_text: str) -> str:
    # Split the input string
    amount_currency = msg_text.split()

    # Check if the input has both amount and currency or just the amount
    if len(amount_currency) == 2:
        amount, currency = amount_currency
        currency = currency.upper()  # Convert currency to uppercase
    elif len(amount_currency) == 1:
        amount, currency = amount_currency[0], None  # Set currency to None
    else:
        return "Invalid input format. Please use either 'amount' in satoshis or 'amount fiat_currency'."

    # Check if amount is a valid number
    try:
        amount = float(amount) if currency else int(amount)
    except ValueError:
        return "Invalid amount. Please enter a valid number."

    # Reject negative amounts and zero
    if amount <= 0:
        return "Invalid amount. Please enter a positive number greater than zero."

    valid_fiat_currencies = [
        "AED",
        "AFN",
        "ALL",
        "AMD",
        "ANG",
        "AOA",
        "ARS",
        "AUD",
        "AWG",
        "AZN",
        "BAM",
        "BBD",
        "BDT",
        "BGN",
        "BHD",
        "BIF",
        "BMD",
        "BND",
        "BOB",
        "BRL",
        "BSD",
        "BTN",
        "BWP",
        "BYN",
        "BZD",
        "CAD",
        "CDF",
        "CHF",
        "CLF",
        "CLP",
        "CNH",
        "CNY",
        "COP",
        "CRC",
        "CUC",
        "CUP",
        "CVE",
        "CZK",
        "DJF",
        "DKK",
        "DOP",
        "DZD",
        "EGP",
        "ERN",
        "ETB",
        "EUR",
        "FJD",
        "FKP",
        "GBP",
        "GEL",
        "GGP",
        "GHS",
        "GIP",
        "GMD",
        "GNF",
        "GTQ",
        "GYD",
        "HKD",
        "HNL",
        "HRK",
        "HTG",
        "HUF",
        "IDR",
        "ILS",
        "IMP",
        "INR",
        "IQD",
        "IRR",
        "ISK",
        "JEP",
        "JMD",
        "JOD",
        "JPY",
        "KES",
        "KGS",
        "KHR",
        "KMF",
        "KPW",
        "KRW",
        "KWD",
        "KYD",
        "KZT",
        "LAK",
        "LBP",
        "LKR",
        "LRD",
        "LSL",
        "LYD",
        "MAD",
        "MDL",
        "MGA",
        "MKD",
        "MMK",
        "MNT",
        "MOP",
        "MRO",
        "MUR",
        "MVR",
        "MWK",
        "MXN",
        "MYR",
        "MZN",
        "NAD",
        "NGN",
        "NIO",
        "NOK",
        "NPR",
        "NZD",
        "OMR",
        "PAB",
        "PEN",
        "PGK",
        "PHP",
        "PKR",
        "PLN",
        "PYG",
        "QAR",
        "RON",
        "RSD",
        "RUB",
        "RWF",
        "SAR",
        "SBD",
        "SCR",
        "SDG",
        "SEK",
        "SGD",
        "SHP",
        "SLL",
        "SOS",
        "SRD",
        "SSP",
        "STD",
        "SVC",
        "SYP",
        "SZL",
        "THB",
        "TJS",
        "TMT",
        "TND",
        "TOP",
        "TRY",
        "TTD",
        "TWD",
        "TZS",
        "UAH",
        "UGX",
        "USD",
        "UYU",
        "UZS",
        "VES",
        "VND",
        "VUV",
        "WST",
        "XAF",
        "XAG",
        "XAU",
        "XCD",
        "XDR",
        "XOF",
        "XPD",
        "XPF",
        "XPT",
        "YER",
        "ZAR",
        "ZMW",
        "ZWL",
    ]

    # Check if currency is in the valid fiat currencies list
    if currency is not None and currency not in valid_fiat_currencies:
        return "Invalid currency. Please enter a valid 3-letter fiat currency code."

    url = "https://api.opennode.com/v1/charges"
    payload = {
        "amount": amount,
        "description": f"Donation to @{environ.get('TELEGRAM_USERNAME')}",
    }

    # Include currency in the payload if provided
    if currency is not None:
        payload["currency"] = currency

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": environ.get("OPENNODE_INVOICE_KEY"),
    }
    response = requests.post(url, json=payload, headers=headers, timeout=5)
    return response.json()["data"]["lightning_invoice"]["payreq"]


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
        "/donate": donation,
        "/help": get_help,
    }


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
                if (
                    reply_to != environ.get("TELEGRAM_USERNAME")
                    and bot_name not in message_text
                    and not command.startswith("/ask")
                ):
                    return "ignored request"
            except KeyError:
                if (
                    chat_type != "private"
                    and bot_name not in message_text
                    and not command.startswith("/ask")
                ):
                    return "ignored request"

            send_typing(token, chat_id)
            response_msg = gen_random(first_name)

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
        telegram_response = requests.get(request_url, params=parameters, timeout=5)
        telegram_response.raise_for_status()
    except RequestException:
        return False
    redis_client = config_redis()
    redis_response = redis_client.set("X-Telegram-Bot-Api-Secret-Token", secret_token)
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