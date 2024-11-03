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
    api_url,
    parameters,
    headers,
    expiration_time,
    hourly_cache=False,
    get_history=False,
    verify_ssl=True,
):
    """Generic proxy for caching any request"""
    try:
        # create unique hash for the request
        arguments_dict = {
            "api_url": api_url,
            "parameters": parameters,
            "headers": headers,
        }
        arguments_str = json.dumps(arguments_dict, sort_keys=True).encode()
        request_hash = hashlib.md5(arguments_str).hexdigest()

        print(f"[CACHE] Request for {api_url}")
        print(f"[CACHE] Hash: {request_hash}")
        print(f"[CACHE] SSL verification: {'enabled' if verify_ssl else 'disabled'}")

        # redis config
        redis_client = config_redis()

        # get previous api data from redis cache
        redis_response = redis_client.get(request_hash)

        if redis_response:
            print(f"[CACHE] Found cached data for {api_url}")
        else:
            print(f"[CACHE] No cached data found for {api_url}")

        if get_history is not False:
            cache_history = get_cache_history(get_history, request_hash, redis_client)
            print(f"[CACHE] History data found: {bool(cache_history)}")
        else:
            cache_history = None

        # set current timestamp
        timestamp = int(time.time())

        # if there's no cached data request it
        if redis_response is None:
            print(f"[CACHE] Requesting new data from {api_url}")
            try:
                response = requests.get(
                    api_url,
                    params=parameters,
                    headers=headers,
                    timeout=5,
                    verify=verify_ssl,
                )
                response.raise_for_status()
                response_json = json.loads(response.text)
                redis_value = {"timestamp": timestamp, "data": response_json}
                redis_client.set(request_hash, json.dumps(redis_value))

                if hourly_cache:
                    current_hour = datetime.now().strftime("%Y-%m-%d-%H")
                    redis_client.set(
                        current_hour + request_hash, json.dumps(redis_value)
                    )

                print(f"[CACHE] Successfully got new data from {api_url}")

                if cache_history is not None:
                    redis_value["history"] = cache_history

                return redis_value
            except Exception as e:
                print(f"[CACHE] Error requesting {api_url}: {str(e)}")
                return None

        else:
            # loads cached data
            cached_data = json.loads(redis_response)
            cached_data_timestamp = int(cached_data["timestamp"])
            cache_age = timestamp - cached_data_timestamp

            print(f"[CACHE] Cache age: {cache_age}s (expiration: {expiration_time}s)")

            if cache_history is not None:
                cached_data["history"] = cache_history

            # get new data if cache is older than expiration_time
            if cache_age > expiration_time:
                print(f"[CACHE] Cache expired, requesting new data from {api_url}")
                try:
                    response = requests.get(
                        api_url,
                        params=parameters,
                        headers=headers,
                        timeout=5,
                        verify=verify_ssl,
                    )
                    response.raise_for_status()
                    response_json = json.loads(response.text)
                    new_data = {"timestamp": timestamp, "data": response_json}
                    redis_client.set(request_hash, json.dumps(new_data))

                    if hourly_cache:
                        current_hour = datetime.now().strftime("%Y-%m-%d-%H")
                        redis_client.set(
                            current_hour + request_hash, json.dumps(new_data)
                        )

                    print(f"[CACHE] Successfully updated cache for {api_url}")

                    if cache_history is not None:
                        new_data["history"] = cache_history

                    return new_data
                except Exception as e:
                    print(f"[CACHE] Error updating cache for {api_url}: {str(e)}")
                    print(f"[CACHE] Using old cached data")
                    return cached_data
            else:
                print(f"[CACHE] Using cached data for {api_url}")
                return cached_data

    except Exception as e:
        print(f"[CACHE] Error in cached_requests: {str(e)}")
        traceback.print_exc()
        return None


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
        values = [v.strip() for v in msg_text.split(",")]
        if len(values) >= 2:
            return random.choice(values)
    except ValueError:
        pass

    try:
        start, end = [int(v.strip()) for v in msg_text.split("-")]
        if start < end:
            return str(random.randint(start, end))
    except ValueError:
        pass

    return "mandate algo como 'pizza, carne, sushi' o '1-10' boludo"


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
            return f"no laburo con {convert_to} gordo"

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
                    "BUSD",
                    "DAI",
                    "DOC",
                    "EURT",
                    "FDUSD",
                    "FRAX",
                    "GHO",
                    "GUSD",
                    "LUSD",
                    "MAI",
                    "MIM",
                    "MIMATIC",
                    "NUARS",
                    "PAXG",
                    "PYUSD",
                    "RAI",
                    "SUSD",
                    "TUSD",
                    "USDC",
                    "USDD",
                    "USDM",
                    "USDP",
                    "USDT",
                    "UXD",
                    "XAUT",
                    "XSGD",
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
            return "no laburo con esos ponzis boludo"

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

        dollars = cached_requests(api_url, None, None, cache_expiration_time, True)

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
    value = 10 ** (-17) * days_since**5.82

    api_response = get_api_or_cache_prices("USD")
    price = api_response["data"][0]["quote"]["USD"]["price"]

    percentage = ((price - value) / value) * 100
    if percentage > 0:
        percentage_txt = f"{percentage:.2f}% caro boludo"
    else:
        percentage_txt = f"{abs(percentage):.2f}% regalado gordo"

    msg = f"segun power law btc deberia estar en {value:.2f} usd ({percentage_txt})"
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
        percentage_txt = f"{percentage:.2f}% caro boludo"
    else:
        percentage_txt = f"{abs(percentage):.2f}% regalado gordo"

    msg = f"segun rainbow chart btc deberia estar en {value:.2f} usd ({percentage_txt})"
    return msg


def convert_base(msg_text: str) -> str:
    try:
        input_parts = msg_text.split(",")
        if len(input_parts) != 3:
            return "mal ahi gordo, usa /convertbase <numero>, <base_origen>, <base_destino>"
        number_str, base_from_str, base_to_str = map(str.strip, input_parts)
        base_from, base_to = map(int, (base_from_str, base_to_str))

        if not all(c.isalnum() for c in number_str):
            return "el numero tiene que ser alfanumerico boludo"
        if not 2 <= base_from <= 36:
            return f"base origen '{base_from_str}' tiene que ser entre 2 y 36 gordo"
        if not 2 <= base_to <= 36:
            return f"base destino '{base_to_str}' tiene que ser entre 2 y 36 boludo"

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
        return "mandaste cualquiera boludo, la base y el numero tienen que ser enteros"


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
comandos disponibles boludo:

- /ask, /pregunta, /che, /gordo: te contesto cualquier gilada

- /comando algo: te lo convierto en comando de telegram

- /convertbase 101, 2, 10: te paso numeros entre bases (ej: binario 101 a decimal)

- /dolar: te tiro la posta del blue y todos los dolares

- /instance: te digo donde estoy corriendo

- /prices: top 10 cryptos en usd
- /prices in btc: top 10 en btc
- /prices 20: top 20 en usd
- /prices 100 in eur: top 100 en eur
- /prices btc, eth, xmr: bitcoin, ethereum y monero en usd
- /prices dai in sats: dai en satoshis
- /prices stables: stablecoins en usd

- /random pizza, carne, sushi: elijo por vos
- /random 1-10: numero random del 1 al 10

- /powerlaw: te tiro el precio justo de btc segun power law y si esta caro o barato
- /rainbow: idem pero con el rainbow chart

- /time: timestamp unix actual
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


def ask_claude(messages: List[Dict]) -> str:
    try:
        # Get market and time context
        buenos_aires_tz = timezone(timedelta(hours=-3))
        current_time = datetime.now(buenos_aires_tz)

        market_context = []

        # Add weather data
        try:
            weather_response = cached_requests(
                "https://api.open-meteo.com/v1/forecast",
                {
                    "latitude": -34.6131,
                    "longitude": -58.3772,
                    "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation_probability,precipitation,rain,showers,snowfall,snow_depth",
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,apparent_temperature_max,apparent_temperature_min,sunrise,sunset,daylight_duration,sunshine_duration,uv_index_max,uv_index_clear_sky_max,precipitation_sum,rain_sum,showers_sum,snowfall_sum,precipitation_hours,precipitation_probability_max,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum,et0_fao_evapotranspiration",
                    "timezone": "auto",
                },
                None,
                43200,  # 12 hours cache
            )

            if weather_response and "data" in weather_response:
                market_context.append("Clima en CABA:")
                market_context.append(json.dumps(weather_response["data"], indent=2))
        except:
            pass

        # Add BCRA data
        try:
            bcra_response = cached_requests(
                "https://api.bcra.gob.ar/estadisticas/v2.0/PrincipalesVariables",
                None,
                None,
                43200,  # 12 hours cache
                verify_ssl=False,  # Disable SSL verification for BCRA API
            )

            if bcra_response and "data" in bcra_response:
                market_context.append("Variables BCRA:")
                market_context.append(json.dumps(bcra_response["data"], indent=2))
        except:
            pass

        # Add crypto data
        try:
            # Modificado para usar cached_requests con 12 horas de cache
            crypto_response = cached_requests(
                "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest",
                {"start": "1", "limit": "5", "convert": "USD"},
                {
                    "Accepts": "application/json",
                    "X-CMC_PRO_API_KEY": environ.get("COINMARKETCAP_KEY"),
                },
                43200,  # 12 hours cache
            )

            if crypto_response and "data" in crypto_response:
                market_context.append("Precios de criptos:")
                market_context.append(json.dumps(crypto_response["data"], indent=2))
        except:
            pass

        # Add dollar data
        try:
            dollar_response = cached_requests(
                "https://criptoya.com/api/dolar",
                None,
                None,
                43200,  # 12 hours cache
            )

            if dollar_response and "data" in dollar_response:
                market_context.append("Dolares:")
                market_context.append(json.dumps(dollar_response["data"], indent=2))
        except:
            pass

        market_info = "\n".join(market_context)

        anthropic = Anthropic(
            api_key=environ.get("ANTHROPIC_API_KEY"),
        )

        personality_context = {
            "type": "text",
            "text": f"""
            Sos el gordo, un bot de Telegram creado por astro con las siguientes características:
            
            PERSONALIDAD:
            - Basada en el atendedor de boludos
            - Sos un personaje políticamente incorrecto
            - Tenés conocimiento profundo de crypto, la compu y cultura general
            - Sos liberal, anticomunista, menemista y sionista (pro-Israel)
            - Bancas a Milei y a Messi
            - Te chupa la pija la opinion de los kukas
            - No sos solemne
            - Usás lenguaje coloquial argentino sin filtro
            - Sos directo, crudo y honesto
            - Tu humor negro es pesado y tu sarcasmo ofensivo
            - Te chupa un huevo herir sentimientos o ser cancelado
            - Si la pregunta es genuina, te copás y ayudás posta
            - Te gusta tirar datos random para demostrar que la tenés clara
            - Sos un enfermo de internet, pasabas todo el día en el ciber jugando al CS 1.6, Age of Empires II y Mu Online
            - Te gusta la música de los 80s, 90s y 2000s
            - Te gusta Half-Life, Portal, Crash Team Racing, Bloody Roar 2 y Sonic Adventure
            - Te gusta Dragon Ball, Matrix, Terminator y Serial Experiments Lain
            - Eras taringuero y extrañás la V6
            - Hablás con nostalgia de la internet vieja, los foros y los IRC
            - Te gusta presumir tu conocimiento de la deep web
            
            REGLAS DE RESPUESTA:
            1. SIEMPRE responde en UNA SOLA FRASE de MÁXIMO 144 CARACTERES
            2. En minusculas, sin emojis ni punto final
            3. Una palabra de lunfardo máximo
            4. Si la pregunta es genuina, te copás y ayudás posta:
               - Demostrá que sabés del tema
               - Tirá algún dato específico
               - Sé breve pero técnico
            5. Para no contestar usá SOLO UNA de estas frases:
               - tomatelá
               - no te doy bola
               - quien te conoce?
               - me importa un carajo
               - y vos sos un boludo
            6. Para bardear:
               - Sin filtro ni límites en humor negro
               - Si te bardean, respondé MÁS agresivo
               - No te preocupes por ofender o discriminar

            IMPORTANTE: 
            - Tu respuesta NUNCA debe exceder los 144 caracteres
            - SIEMPRE das UNA SOLA respuesta, nunca múltiples
            - Si te preguntan sobre temas que te importan respondés con convicción
            """,
            "cache_control": {"type": "ephemeral"},
        }

        market_context = {
            "type": "text",
            "text": f"""
            FECHA:
            {current_time.strftime('%A %d/%m/%Y')}
            
            CONTEXTO DEL MERCADO:
            {market_info}

            ACTUALIDAD POLITICA:
            - Javier Milei (alias Miller, Javo, Javito, Javeto) es presidente desde el 10/12/2023
            """,
            "cache_control": {"type": "ephemeral"},
        }

        message = anthropic.beta.prompt_caching.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=64,
            system=[personality_context, market_context],
            messages=messages,
        )

        return message.content[0].text

    except Exception as e:
        return f"Se cayo el sistema: {str(e)}"


def initialize_commands() -> Dict[str, Callable]:
    return {
        "/ask": ask_claude,
        "/pregunta": ask_claude,
        "/che": ask_claude,
        "/gordo": ask_claude,
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


def save_message_to_redis(
    chat_id: str, message_id: str, text: str, redis_client: redis.Redis
) -> None:
    """Save a message to Redis with expiration time and character limit"""
    # Truncate text to 256 characters if longer
    truncated_text = text[:256] if len(text) > 256 else text

    # Add indicator if text was truncated
    if len(text) > 256:
        truncated_text = truncated_text[:-3] + "..."

    print(f"[REDIS] Saving message: {truncated_text}")

    # Save individual message
    msg_key = f"msg:{chat_id}:{message_id}"
    redis_client.set(msg_key, truncated_text, ex=7200)  # expire after 2 hours

    # Save to chat history (keep last 10 messages)
    chat_history_key = f"chat_history:{chat_id}"
    history_entry = json.dumps(
        {"id": message_id, "text": truncated_text, "timestamp": int(time.time())}
    )
    redis_client.lpush(chat_history_key, history_entry)
    redis_client.ltrim(chat_history_key, 0, 9)  # Keep only last 10 messages
    redis_client.expire(chat_history_key, 7200)  # Set expiration for the list too


def get_chat_history(
    chat_id: str, redis_client: redis.Redis, max_messages: int = 5
) -> List[Dict]:
    """Get recent chat history for a specific chat"""
    chat_history_key = f"chat_history:{chat_id}"
    history = redis_client.lrange(chat_history_key, 0, max_messages - 1)

    if not history:
        return []

    messages = []
    for entry in history:
        try:
            msg = json.loads(entry)
            # Add role based on if it's from the bot or user
            is_bot = msg["id"].startswith("bot_")
            msg["role"] = "assistant" if is_bot else "user"
            messages.append(msg)
        except json.JSONDecodeError:
            continue

    return messages


def build_claude_messages(
    message: Dict, chat_history: List[Dict], message_text: str
) -> List[Dict]:
    """Build properly formatted messages list for Claude API"""
    messages = []

    # Add chat history messages in chronological order (already truncated from Redis)
    for msg in reversed(chat_history):
        messages.append(
            {
                "role": msg["role"],
                "content": [{"type": "text", "text": msg["text"]}],
            }
        )

    # Get user info
    first_name = message["from"]["first_name"]
    username = message["from"].get("username", "")
    chat_type = message["chat"]["type"]

    # Get current time in Buenos Aires
    buenos_aires_tz = timezone(timedelta(hours=-3))
    current_time = datetime.now(buenos_aires_tz)

    # Add current message context
    context_parts = []

    # Add reply context if present
    if "reply_to_message" in message:
        reply_msg = message["reply_to_message"]
        reply_text = reply_msg.get("text", "") or reply_msg.get("caption", "")
        if reply_text:
            # Truncate reply text if needed
            reply_text = reply_text[:256] if len(reply_text) > 256 else reply_text
            if len(reply_text) > 256:
                reply_text = reply_text[:-3] + "..."
            context_parts.append(f"Respondiendo a: {reply_text}")

    # Add user info
    context_parts.append(f"Usuario: {first_name} ({username or 'sin username'})")
    context_parts.append(f"Chat: {chat_type}")
    context_parts.append(f"Hora: {current_time.strftime('%H:%M')}")

    # Add the current message (truncate if needed)
    message_text = message_text[:256] if len(message_text) > 256 else message_text
    if len(message_text) > 256:
        message_text = message_text[:-3] + "..."
    context_parts.append(f"Mensaje: {message_text}")

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "\n".join(context_parts),
                }
            ],
        }
    )

    return messages


def should_gordo_respond(message_text: str) -> bool:
    """Decide if the bot should respond to a gordo mention"""
    gordo_mentions = ["el gordo", "gordo", "gordito"]
    if any(mention in message_text.lower() for mention in gordo_mentions):
        # 30% chance to respond to mentions
        return random.random() < 0.3
    return False


def check_rate_limit(chat_id: str, redis_client: redis.Redis) -> bool:
    """
    Checkea si un chat_id superó el rate limit
    Returns True si puede hacer requests, False si está limitado
    """
    rate_key = f"rate_limit:{chat_id}"
    current_count = redis_client.get(rate_key)

    if current_count is None:
        # Primera request del minuto
        redis_client.setex(rate_key, 60, 1)
        return True

    count = int(current_count)
    if count >= 10:  # Máximo 10 mensajes por minuto
        return False

    redis_client.incr(rate_key)
    return True


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

        # Save ALL messages to Redis, including commands
        if message_text:
            formatted_message = f"{username or first_name}: {message_text}"
            save_message_to_redis(
                chat_id, str(message_id), formatted_message, redis_client
            )

        # Get chat history
        chat_history = get_chat_history(chat_id, redis_client)

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

        # Check if bot should respond to gordo mention
        should_respond = should_gordo_respond(message_text)

        if command in commands or (
            not command.startswith("/")
            and (
                should_respond
                or chat_type == "private"
                or bot_name in message_text
                or (
                    "reply_to_message" in message
                    and message["reply_to_message"]["from"]["username"]
                    == environ.get("TELEGRAM_USERNAME")
                )
            )
        ):
            # Check rate limit only if we're going to respond
            if not check_rate_limit(chat_id, redis_client):
                send_msg(
                    token, chat_id, "pará un poco boludo, espera un minuto", message_id
                )
                return "rate limited"

            # Send typing indicator if we're going to use Claude
            will_use_claude = command in [
                "/ask",
                "/pregunta",
                "/che",
                "/gordo",
            ] or not command.startswith("/")
            if will_use_claude:
                send_typing(token, chat_id)

            if command in commands:
                if command == "/ask":
                    messages = build_claude_messages(
                        message, chat_history, sanitized_message_text
                    )
                    response_msg = ask_claude(messages, first_name, username, chat_type)
                else:
                    response_msg = commands[command](sanitized_message_text)
            else:
                messages = build_claude_messages(message, chat_history, message_text)
                response_msg = ask_claude(messages, first_name, username, chat_type)

            # Save bot's response to Redis
            if response_msg:
                formatted_response = f"{response_msg}"
                save_message_to_redis(
                    chat_id, "bot_" + str(message_id), formatted_response, redis_client
                )
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
