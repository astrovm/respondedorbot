import random
import json
import redis
import time
import requests
import functions_framework
from flask import Request
from typing import Dict
from os import environ
from math import log
from datetime import datetime, timedelta
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor


def _config_redis(host=None, port=None, password=None):
    """
    Configure a Redis client with the provided or default values and return it.
    """
    host = host or environ.get("REDIS_HOST", "localhost")
    port = port or environ.get("REDIS_PORT", 6379)
    password = password or environ.get("REDIS_PASSWORD", None)
    redis_client = redis.Redis(host=host, port=port, password=password)
    return redis_client


# request new data and save it in redis
def _set_new_data(api_url, parameters, headers, timestamp, redis_client, request_hash, hourly_cache):
    response = requests.get(api_url, params=parameters, headers=headers)

    if response.status_code == 200:
        response_json = json.loads(response.text)
        redis_value = {"timestamp": timestamp, "data": response_json}
        redis_client.set(request_hash, json.dumps(redis_value))

        # if hourly_cache is True, save the data in redis with the current hour
        if hourly_cache:
            # get current date with hour
            date = datetime.now().strftime("%Y-%m-%d-%H")
            redis_client.set(date+request_hash, json.dumps(redis_value))

        return redis_value
    else:
        return None


# get cached data from previous hour
def _get_cache_history(hours, request_hash, redis_client):
    # subtract hours to current date
    date = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d-%H")

    # get previous api data from redis cache
    redis_response = redis_client.get(date+request_hash)

    if redis_response is None:
        return None
    else:
        return json.loads(redis_response)


# generic proxy for caching any request
def _cached_requests(api_url, parameters, headers, expiration_time, hourly_cache=False, get_history=False):
    # create unique hash for the request
    arguments_dict = {"api_url": api_url,
                      "parameters": parameters,
                      "headers": headers}
    arguments_str = json.dumps(arguments_dict, sort_keys=True)
    request_hash = str(hash(arguments_str))

    # redis config
    redis_client = _config_redis()

    # get previous api data from redis cache
    redis_response = redis_client.get(request_hash)

    if get_history is not False:
        cache_history = _get_cache_history(
            get_history, request_hash, redis_client)
        if cache_history is not None and "timestamp" not in cache_history:
            cache_history = None
    else:
        cache_history = None

    # set current timestamp
    timestamp = int(time.time())

    # if there's no cached data request it
    if redis_response is None:
        new_data = _set_new_data(
            api_url, parameters, headers, timestamp, redis_client, request_hash, hourly_cache)

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
                api_url, parameters, headers, timestamp, redis_client, request_hash, hourly_cache)

            if cache_history is not None:
                new_data["history"] = cache_history

            if "timestamp" in new_data:
                return new_data
            else:
                return cached_data
        else:
            return cached_data


def gen_random(name: str) -> str:
    try:
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

    except Exception as e:
        print(f"Error: {e}")


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

        for n in numbers:
            try:
                number = int(float(n))
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

        if new_prices == []:
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

        decimals = "{:.12f}".format(
            coin["quote"][convert_to_parameter]["price"]).split(".")[-1]
        zeros = len(decimals) - len(decimals.lstrip("0"))

        ticker = coin["symbol"]
        price = "{:.{decimals_number}f}".format(
            coin["quote"][convert_to_parameter]["price"], decimals_number=zeros + 4).rstrip("0").rstrip(".")
        percentage = "{:+.2f}".format(
            coin["quote"][convert_to_parameter]["percent_change_24h"]).rstrip("0").rstrip(".")
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
        dollars_history = _to_float(dollar_rates["history"])
        dollars_history = _add_derived_rates(
            dollars_history, "oficial", derived_rates)

        for rate_type in [("usdc", usdc_rates), ("dai", dai_rates), ("usdt", usdt_rates)]:
            if "history" in rate_type[1]:
                dollars_history[rate_type[0]] = _get_lowest(
                    rate_type[1]["history"])

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


def get_dollars(msg_text: str) -> str:
    cache_expiration_time = 300
    with ThreadPoolExecutor(max_workers=5) as executor:
        dollars_thread = executor.submit(
            _cached_requests, "https://criptoya.com/api/dolar", None, None, cache_expiration_time, True, 24)
        usdc_thread = executor.submit(
            _cached_requests, "https://criptoya.com/api/usdc/ars/1000", None, None, cache_expiration_time, True, 24)
        dai_thread = executor.submit(
            _cached_requests, "https://criptoya.com/api/dai/ars/1000", None, None, cache_expiration_time, True, 24)
        usdt_thread = executor.submit(
            _cached_requests, "https://criptoya.com/api/usdt/ars/1000", None, None, cache_expiration_time, True, 24)

    dollars = _sort_dollar_rates(dollars_thread.result(), usdc_thread.result(),
                                 dai_thread.result(), usdt_thread.result())

    msg = ""
    for dollar in dollars:
        line = f"{dollar['name']}: {'{:.2f}'.format(dollar['price']).rstrip('0').rstrip('.')}"
        if dollar["history"] is not None:
            percentage = "{:+.2f}".format(
                (dollar['price']/dollar["history"]-1)*100).rstrip("0").rstrip(".")
            line += f" ({percentage}% 24hs)"

        if dollar == dollars[0]:
            msg = line
        else:
            msg += f"\n{line}"

    return msg


def get_devo(msg_text: str) -> str:
    fee = 0
    compra = 0

    if "," in msg_text:
        numbers = msg_text.replace(" ", "").split(",")
        fee = float(numbers[0]) / 100
        if len(numbers) > 1:
            compra = float(numbers[1])
    else:
        fee = float(msg_text) / 100

    if fee != fee or fee < 0 or fee > 1 or compra != compra or compra < 0:
        return "te voy a matar hijo de puta"

    cache_expiration_time = 300
    with ThreadPoolExecutor(max_workers=5) as executor:
        dollars_thread = executor.submit(
            _cached_requests, "https://criptoya.com/api/dolar", None, None, cache_expiration_time)
        usdt_thread = executor.submit(
            _cached_requests, "https://criptoya.com/api/usdt/ars/1000", None, None, cache_expiration_time)

    dollars = {key: float(value)
               for key, value in dollars_thread.result()["data"].items()}
    usdt = usdt_thread.result()["data"]
    dollars["usdt"] = _get_lowest(usdt)

    tarjeta_tax = 1.75
    qatar_tax = 2

    profit = -(fee * dollars["usdt"] + dollars["oficial"] -
               dollars["usdt"]) / (dollars["oficial"] * qatar_tax)

    msg = f"""Profit: {"{:.2f}".format(profit * 100).rstrip("0").rstrip(".")}%

Fee: {"{:.2f}".format(fee * 100).rstrip("0").rstrip(".")}%
Oficial: {"{:.2f}".format(dollars["oficial"]).rstrip("0").rstrip(".")}
USDT: {"{:.2f}".format(dollars["usdt"]).rstrip("0").rstrip(".")}
Qatar: {"{:.2f}".format(dollars["oficial"] * qatar_tax).rstrip("0").rstrip(".")}
Tarjeta: {"{:.2f}".format(dollars["oficial"] * tarjeta_tax).rstrip("0").rstrip(".")}"""

    if compra > 0:
        compra_ars = compra * (dollars["oficial"] * qatar_tax)
        compra_usdt = compra_ars / dollars["usdt"]
        ganancia_ars = compra_ars * profit
        ganancia_usdt = ganancia_ars / dollars["usdt"]
        msg = f"""{"{:.2f}".format(compra).rstrip("0").rstrip(".")} USD Qatar = {"{:.2f}".format(compra_ars).rstrip("0").rstrip(".")} ARS = {"{:.2f}".format(compra_usdt).rstrip("0").rstrip(".")} USDT
Ganarias {"{:.2f}".format(ganancia_ars).rstrip("0").rstrip(".")} ARS / {"{:.2f}".format(ganancia_usdt).rstrip("0").rstrip(".")} USDT
Total: {"{:.2f}".format(compra_ars + ganancia_ars).rstrip("0").rstrip(".")} ARS / {"{:.2f}".format(compra_usdt + ganancia_usdt).rstrip("0").rstrip(".")} USDT

{msg}"""

    return msg


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


def get_help(msg_text: str) -> str:
    return """
Available commands:

- /ask question: returns the answer to the question

- /random pizza, meat, sushi: chooses between the options
- /random 1-10: returns a random number between 1 and 10

- /prices: prices of the top 10 cryptos in usd
- /prices in btc: prices of the top 10 cryptos in btc
- /prices 20: prices of the top 20 cryptos in usd
- /prices 100 in eur: prices of the top 100 cryptos in eur
- /prices btc: price of bitcoin in usd
- /prices btc, eth, xmr: prices of bitcoin, ethereum and monero in usd
- /prices dai in sats: price of dai in satoshis
- /prices btc, eth, xmr in sats: price of bitcoin, ethereum and monero in satoshis

- /dolar: dollar prices in argentina

- /time: returns the current unix timestamp
    """


def send_typing(token: str, chat_id: str):
    url = "https://api.telegram.org/bot" + token + \
        "/sendChatAction?chat_id=" + chat_id + "&action=typing"
    requests.get(url)


def send_msg(token: str, chat_id: str, msg_id: str, msg: str):
    url = "https://api.telegram.org/bot" + token + "/sendMessage?chat_id=" + \
        chat_id + "&reply_to_message_id=" + \
        msg_id + "&text=" + quote(msg, safe='/')
    requests.get(url)


def handle_msg(start_time: float, token: str, req: Dict) -> str:
    """Handle incoming messages and return a response."""
    msg_text = str(req["message"]["text"])
    sanitized_msg_text = msg_text
    msg_id = str(req["message"]["message_id"])
    chat_id = str(req["message"]["chat"]["id"])
    chat_type = str(req["message"]["chat"]["type"])
    first_name = str(req["message"]["from"]["first_name"])
    typing = False
    msg_to_send = ""

    if sanitized_msg_text.startswith("/exectime"):
        sanitized_msg_text = sanitized_msg_text.replace(
            "/exectime", "").strip()
    else:
        start_time = False

    split_msg = sanitized_msg_text.strip().split(" ")
    lower_cmd = split_msg[0].lower()
    sanitized_msg_text = sanitized_msg_text.replace(split_msg[0], "").strip()

    # Map commands to functions
    commands = {
        "/convertbase": convert_base,
        "/random": select_random,
        "/prices": get_prices,
        "/dolar": get_dollars,
        "/devo": get_devo,
        "/rainbow": rainbow,
        "/time": get_timestamp,
        "/help": get_help
    }

    # Check for the bot's name in the command, if applicable
    bot_name = "@respondedorbot"
    if bot_name in lower_cmd:
        lower_cmd = lower_cmd.replace(bot_name, "")

    if lower_cmd in commands:
        send_typing(token, chat_id)
        typing = True
        msg_to_send = commands[lower_cmd](sanitized_msg_text)

    else:
        try:
            reply_to = str(
                req["message"]["reply_to_message"]["from"]["username"])
            if reply_to != "respondedorbot" and bot_name not in msg_text and not lower_cmd.startswith("/ask"):
                return "ignored request"
        except KeyError:
            if chat_type != "private" and bot_name not in msg_text and not lower_cmd.startswith("/ask"):
                return "ignored request"

        send_typing(token, chat_id)
        typing = True
        msg_to_send = gen_random(first_name)

    if start_time:
        exec_time = round(time.time() - start_time, 4)
        msg_to_send = f"{msg_to_send}\n\nExecution time: {exec_time:.4f} secs"

    send_msg(token, chat_id, msg_id, msg_to_send)


@functions_framework.http
def responder(request: Request) -> str:
    start_time = time.time()
    try:
        token = str(request.args.get("token"))
        if token != environ.get("TELEGRAM_TOKEN"):
            return "wrong token"

        req = request.get_json()

        handle_msg(start_time, token, req)

        return "ok"
    except BaseException:
        return "unexpected error"
