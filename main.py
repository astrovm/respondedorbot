import json
import redis
import time
from os import environ
from math import log
from random import randint, uniform
from requests import get
from datetime import datetime
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor


def config_redis(host=environ.get("REDIS_HOST"), port=environ.get("REDIS_PORT"), password=environ.get("REDIS_PASSWORD")):
    r = redis.Redis(
        host=host,
        port=port,
        password=password)
    return r


def gen_random(name):
    time.sleep(uniform(0, 1))

    randRes = randint(0, 1)
    randName = randint(0, 2)

    if randRes == 1:
        msg = "si"
    else:
        msg = "no"

    if randName == 1:
        msg = f"{msg} boludo"
        time.sleep(uniform(0, 1))
    elif randName == 2:
        msg = f"{msg} {name}"
        time.sleep(uniform(0, 1))

    return msg


def select_random(msg_text):
    if "," in msg_text:
        if msg_text.startswith(",") or msg_text.endswith(","):
            return "habla bien idiota"

        split_msg = msg_text.split(",")
        values = len(split_msg)

        randValue = randint(0, values - 1)
        strip_value = split_msg[randValue].strip()

        return strip_value

    if "-" in msg_text:
        if msg_text.startswith("-") or msg_text.endswith("-"):
            return "habla bien idiota"

        split_msg = msg_text.split("-")
        values = len(split_msg)

        if values == 2:
            strip_start = int(split_msg[0].strip())
            strip_end = int(split_msg[1].strip())
            rand_num = str(randint(strip_start, strip_end))

            return rand_num

    return "habla bien idiota"


# request new prices and save them in redis
def _set_new_prices(api_url, parameters, headers, timestamp, response, r):
    response = get(api_url, params=parameters, headers=headers)
    prices = json.loads(response.text)

    r.set(f"{api_url}{parameters['convert']}", json.dumps(
        {"timestamp": timestamp, "prices": prices}))

    return prices


def get_prices(msg_text):
    # coinmarketcap api config
    api_url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    convert_to = "USD"
    parameters = {
        'start': '1',
        'limit': '100',
        'convert': convert_to
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': environ.get("COINMARKETCAP_KEY"),
    }
    prices_number = 0

    # check if the user wants to convert the prices
    if "IN " in msg_text.upper():
        words = msg_text.upper().split()
        coins = ["XAU", "USD", "EUR", "KRW", "GBP", "AUD", "BRL", "CAD", "CLP", "CNY", "COP", "CZK", "DKK", "HKD", "ISK", "INR", "ILS",
                 "JPY", "MXN", "TWD", "NZD", "PEN", "SGD", "SEK", "CHF", "UYU", "BTC", "SATS", "ETH", "XMR", "USDC", "USDT", "DAI", "BUSD"]
        convert_to = words[-1]
        if convert_to in coins:
            if convert_to == "SATS":
                parameters["convert"] = "BTC"
            else:
                parameters["convert"] = convert_to
            msg_text = msg_text.upper().replace(
                f"IN {convert_to}", "").strip()
        else:
            return f"{convert_to} is not allowed"

    # redis config for cache
    r = config_redis()

    # get previous api response from redis cache
    redis_response = r.get(f"{api_url}{parameters['convert']}")

    # set current timestamp
    timestamp = int(time.time())

    # if there's no cached prices request them
    if redis_response is None:
        prices = _set_new_prices(api_url, parameters,
                                 headers, timestamp, response, r)
    else:
        # loads cached prices
        response = json.loads(redis_response)
        response_timestamp = int(response["timestamp"])

        # get new prices if cached prices are older than 200 seconds
        if timestamp - response_timestamp > 200:
            prices = _set_new_prices(
                api_url, parameters, headers, timestamp, response, r)
        # use cached prices if they are recent
        else:
            prices = response["prices"]

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
            coin["quote"][parameters["convert"]
                          ]["price"] = coin["quote"][parameters["convert"]]["price"] * 100000000

        decimals = "{:.12f}".format(
            coin["quote"][parameters["convert"]]["price"]).split(".")[-1]
        zeros = len(decimals) - len(decimals.lstrip("0"))

        ticker = coin["symbol"]
        price = "{:.{decimals_number}f}".format(
            coin["quote"][parameters["convert"]]["price"], decimals_number=zeros + 4).rstrip("0").rstrip(".")
        percentage = "{:+.2f}".format(
            coin["quote"][parameters["convert"]]["percent_change_24h"]).rstrip("0").rstrip(".")
        line = f"{ticker}: {price} {convert_to} ({percentage}% 24hs)"

        if prices["data"][0]["symbol"] == coin["symbol"]:
            msg = line
        else:
            msg = f"""{msg}
{line}"""

    return msg


def _get_lowest(prices):
    lowest_price = float('inf')

    for exchange in prices:
        ask = float(prices[exchange]["totalAsk"])

        if ask == 0:
            continue

        if ask < lowest_price:
            lowest_price = ask

    return lowest_price


def get_dolar():
    executor = ThreadPoolExecutor(max_workers=5)
    dollars_thread = executor.submit(get, "https://criptoya.com/api/dolar")
    usdc_thread = executor.submit(
        get, "https://criptoya.com/api/usdc/ars/1000")
    dai_thread = executor.submit(get, "https://criptoya.com/api/dai/ars/1000")
    usdt_thread = executor.submit(
        get, "https://criptoya.com/api/usdt/ars/1000")

    dollars = dollars_thread.result().json()
    usdc = usdc_thread.result().json()
    dai = dai_thread.result().json()
    usdt = usdt_thread.result().json()

    for dollar in dollars:
        dollars[dollar] = float(dollars[dollar])

    dollars["usdc"] = _get_lowest(usdc)
    dollars["dai"] = _get_lowest(dai)
    dollars["usdt"] = _get_lowest(usdt)

    dollars = [
        {"name": "Oficial", "price": dollars["oficial"]},
        {"name": "Solidario", "price": dollars["oficial"] * 1.65},
        {"name": "Tarjeta", "price": dollars["oficial"] * 1.75},
        {"name": "Qatar", "price": dollars["oficial"] * 2},
        {"name": "MEP", "price": dollars["mepgd30"]},
        {"name": "CCL", "price": dollars["cclgd30"]},
        {"name": "Bitcoin", "price": dollars["ccb"]},
        {"name": "Blue", "price": dollars["blue"]},
        {"name": "USDC", "price": dollars["usdc"]},
        {"name": "DAI", "price": dollars["dai"]},
        {"name": "USDT", "price": dollars["usdt"]},
    ]

    dollars.sort(key=lambda x: x.get("price"))

    for dollar in dollars:
        line = f"""{dollar["name"]}: {"{:.2f}".format(dollar["price"]).rstrip("0").rstrip(".")}"""

        if dollar["name"] == dollars[0]["name"]:
            msg = line
        else:
            msg = f"""{msg}
{line}"""

    return msg


def get_devo(msg_text):
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

    executor = ThreadPoolExecutor(max_workers=5)
    dollars_thread = executor.submit(get, "https://criptoya.com/api/dolar")
    usdt_thread = executor.submit(
        get, "https://criptoya.com/api/usdt/ars/1000")

    dollars = dollars_thread.result().json()
    usdt = usdt_thread.result().json()

    for dollar in dollars:
        dollars[dollar] = float(dollars[dollar])

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


def rainbow():
    today = datetime.now()
    since = datetime(day=9, month=1, year=2009)
    days_since = (today - since).days
    value = 10 ** (2.66167155005961 *
                   log(days_since) - 17.9183761889864)

    api_request = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    api_response = get(api_request)

    r = config_redis()

    if api_response.status_code == 200:
        price = api_response.json()
        r.set(api_request, json.dumps(price))
    else:
        redis_response = r.get(api_request)
        if redis_response is None:
            return f"error {api_response.status_code}"
        price = json.loads(redis_response)

    percentage = ((price["bitcoin"]["usd"] - value) / value)*100
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


def get_timestamp():
    return f"{int(time.time())}"


def get_help():
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


def send_typing(token, chat_id):
    url = "https://api.telegram.org/bot" + token + \
        "/sendChatAction?chat_id=" + chat_id + "&action=typing"
    get(url)


def send_msg(token, chat_id, msg_id, msg):
    url = "https://api.telegram.org/bot" + token + "/sendMessage?chat_id=" + \
        chat_id + "&reply_to_message_id=" + \
        msg_id + "&text=" + quote(msg, safe='/')
    get(url)


def handle_msg(start_time, token, req):
    msg_text = str(req["message"]["text"])
    msg_id = str(req["message"]["message_id"])
    chat_id = str(req["message"]["chat"]["id"])
    chat_type = str(req["message"]["chat"]["type"])
    first_name = str(req["message"]["from"]["first_name"])
    typing = False
    msg_to_send = ""

    if msg_text.startswith("/exectime"):
        msg_text = msg_text.replace("/exectime", "").strip()
    else:
        start_time = False

    split_msg = msg_text.strip().split(" ")
    lower_cmd = split_msg[0].lower()
    msg_text = msg_text.replace(split_msg[0], "").strip()

    if lower_cmd.startswith("/convertbase"):
        send_typing(token, chat_id)
        typing = True
        msg_to_send = convert_base(msg_text)

    if lower_cmd.startswith("/random"):
        send_typing(token, chat_id)
        typing = True
        msg_to_send = select_random(msg_text)

    if lower_cmd.startswith("/prices"):
        send_typing(token, chat_id)
        typing = True
        msg_to_send = get_prices(msg_text)

    if lower_cmd.startswith("/dolar"):
        send_typing(token, chat_id)
        typing = True
        msg_to_send = get_dolar()

    if lower_cmd.startswith("/devo"):
        send_typing(token, chat_id)
        typing = True
        msg_to_send = get_devo(msg_text)

    if lower_cmd.startswith("/rainbow"):
        send_typing(token, chat_id)
        typing = True
        msg_to_send = rainbow()

    if lower_cmd.startswith("/time"):
        send_typing(token, chat_id)
        typing = True
        msg_to_send = get_timestamp()

    if lower_cmd.startswith("/help"):
        send_typing(token, chat_id)
        typing = True
        msg_to_send = get_help()

    if not typing:
        try:
            reply_to = str(
                req["message"]["reply_to_message"]["from"]["username"])

            if reply_to != "respondedorbot" and lower_cmd.startswith(
                    "/ask") == False:
                return "ignored request"
        except BaseException:
            if chat_type != "private" and lower_cmd.startswith(
                    "/ask") == False:
                return "ignored request"

        send_typing(token, chat_id)
        typing = True
        msg_to_send = gen_random(first_name)

    if start_time:
        exec_time = round(time.time() - start_time, 4)
        msg_to_send = f"""{msg_to_send}

Execution time: {str(exec_time).rstrip("0").rstrip(".")} secs"""

    send_msg(token, chat_id, msg_id, msg_to_send)


def responder(request):
    start_time = time.time()
    try:
        if request.method == "POST":
            token = str(request.args.get("token"))
            if token != environ.get("TELEGRAM_TOKEN"):
                return "wrong token"

            req = request.get_json()

            handle_msg(start_time, token, req)

            return "ok"
        else:
            return "bad request"
    except BaseException:
        return "unexpected error"
