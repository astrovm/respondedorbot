from math import floor, log
from time import sleep, time
from random import randint, uniform
from requests import get
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor


def gen_random(name):
    sleep(uniform(0, 1))

    randRes = randint(0, 1)
    randName = randint(0, 2)

    if randRes == 1:
        msg = "si"
    else:
        msg = "no"

    if randName == 1:
        msg = f"{msg} boludo"
        sleep(uniform(0, 1))
    elif randName == 2:
        msg = f"{msg} {name}"
        sleep(uniform(0, 1))

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


def get_prices(msg_text):
    try:
        prices = []
        per_page = 10
        vs_currency = "USD"
        vs_currency_api = "USD"

        if "IN " in msg_text.upper():
            words = msg_text.upper().split()

            if words[-1] == "SATS":
                vs_currency = "sats"
                vs_currency_api = "BTC"
            else:
                vs_currency = words[-1]
                vs_currency_api = words[-1]

            msg_text = msg_text.upper().replace(
                f"""IN {words[-1]}""", "").strip()

        if msg_text.upper().isupper():  # if the message doesn't contain numbers
            per_page = 100
        elif not msg_text == "":
            custom_number = int(float(msg_text))

            if custom_number > 0 and custom_number < 101:
                per_page = custom_number

        prices = get(
            f"""https://api.coingecko.com/api/v3/coins/markets?vs_currency={vs_currency_api}&order=market_cap_desc&per_page={per_page}&page=1&sparkline=false&price_change_percentage=24h""").json()

        if prices["error"]:
            return prices["error"]

        if msg_text.upper().isupper():
            new_prices = []
            coins = msg_text.upper().replace(" ", "").split(",")

            for coin in prices:
                symbol = coin["symbol"].upper().replace(" ", "")
                id = coin["id"].upper().replace(" ", "")
                name = coin["name"].upper().replace(" ", "")

                if symbol in coins or id in coins or name in coins:
                    new_prices.append(coin)
            if new_prices == []:
                return "ponzis no laburo"
            prices = new_prices

        msg = ""

        for coin in prices:
            if vs_currency == "sats":
                coin["current_price"] = coin["current_price"] * 100000000

            ticker = coin["symbol"].upper()
            price = "{:.8f}".format(
                coin["current_price"]).rstrip("0").rstrip(".")
            percentage = "{:+.2f}".format(
                coin["price_change_percentage_24h"]).rstrip("0").rstrip(".")

            line = f"""{ticker}: {price} {vs_currency} ({percentage}% 24hs)"""

            if prices[0]["symbol"] == coin["symbol"]:
                msg = line
            else:
                msg = f"""{msg}
{line}"""
    except BaseException as e:
        msg = str(e)

    return msg


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

    def get_lowest(prices):
        lowest_price = 0

        for exchange in prices:
            ask = float(prices[exchange]["totalAsk"])

            if ask == 0:
                continue

            if ask < lowest_price or lowest_price == 0:
                lowest_price = float(prices[exchange]["totalAsk"])

        return lowest_price

    dollars["usdc"] = get_lowest(usdc)
    dollars["dai"] = get_lowest(dai)
    dollars["usdt"] = get_lowest(usdt)

    dollars = [
        {"name": "Oficial", "price": dollars["oficial"]},
        {"name": "Solidario", "price": dollars["solidario"]},
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


def convert_base(msg_text):
    user_input = msg_text.split(",")

    user_input_number = user_input[0].strip()
    base_from = int(user_input[1].strip())
    decimal = 0
    index_counter = len(user_input_number) - 1

    for digit in user_input_number:
        digit_in_ascii = ord(digit)
        if digit_in_ascii >= 65:
            digit = digit_in_ascii - 65 + 10
        decimal += int(digit) * base_from ** index_counter
        index_counter -= 1

    base_to_convert = int(user_input[2].strip())
    max_index = floor(log(decimal) / log(base_to_convert))
    result = ""

    for index in range(max_index, -1, -1):
        for digit in range(base_to_convert, -1, -1):
            index_value = digit * base_to_convert**index
            if decimal - index_value >= 0:
                if digit > 9:
                    result += chr(digit - 10 + 65)
                else:
                    result += str(digit)
                decimal -= index_value
                break

    return f"""{user_input_number} in base {base_from} equals to {result} in base {base_to_convert}"""


def get_timestamp():
    now = str(int(time()))
    return now


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

    if lower_cmd.startswith("/time"):
        send_typing(token, chat_id)
        typing = True
        msg_to_send = get_timestamp()

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
        exec_time = round(time() - start_time, 4)
        msg_to_send = f"""{msg_to_send}

Execution time: {str(exec_time).rstrip("0").rstrip(".")} secs"""

    send_msg(token, chat_id, msg_id, msg_to_send)


def responder(request):
    start_time = time()
    try:
        if request.method == "POST":
            token = str(request.args.get("token"))
            req = request.get_json()

            handle_msg(start_time, token, req)

            return "ok"
        else:
            return "bad request"
    except BaseException:
        return "unexpected error"
