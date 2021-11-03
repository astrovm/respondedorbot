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
    clean_text = msg_text.replace("/random", "").strip()

    if "," in clean_text:
        if clean_text.startswith(",") or clean_text.endswith(","):
            return "habla bien idiota"

        split_msg = clean_text.split(",")
        values = len(split_msg)

        randValue = randint(0, values - 1)
        strip_value = split_msg[randValue].strip()

        return strip_value

    if "-" in clean_text:
        if clean_text.startswith("-") or clean_text.endswith("-"):
            return "habla bien idiota"

        split_msg = clean_text.split("-")
        values = len(split_msg)

        if values == 2:
            strip_start = int(split_msg[0].strip())
            strip_end = int(split_msg[1].strip())
            rand_num = str(randint(strip_start, strip_end))

            return rand_num

    return "habla bien idiota"


def get_prices(msg_text):
    try:
        split_msg = msg_text.strip().split(" ")
        per_page = 10

        if len(split_msg) > 1 and int(
            float(
                split_msg[1])) > 0 and int(
            float(
                split_msg[1])) < 101:
            per_page = int(float(split_msg[1]))

        prices = get(
            f"""https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page={per_page}&page=1&sparkline=false&price_change_percentage=24h""").json()

        msg = ""

        for coin in prices:
            ticker = coin["symbol"].upper()
            price = "{:.8f}".format(
                coin["current_price"]).rstrip("0").rstrip(".")
            percentage = "{:+.2f}".format(
                coin["price_change_percentage_24h"]).rstrip("0").rstrip(".")

            line = f"""{ticker}: {price} USD ({percentage}% 24hs)"""

            if prices[0]["symbol"] == coin["symbol"]:
                msg = line
            else:
                msg = f"""{msg}
{line}"""
    except BaseException:
        msg = "que no sabes lo que es un numero boludito"

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


def generic_orderbook_match(amount, coin, pair, orderbook):
    match_with = "bids" if pair.removesuffix(coin) == pair else "asks"
    remaining = amount
    matched = 0

    for order in orderbook[match_with]:
        order_price = float(order[0])
        order_amount = float(order[1])
        amount_to_match = remaining / order_price if match_with == "asks" else remaining

        if amount_to_match > order_amount:
            matched += order_amount if match_with == "asks" else order_amount * order_price
            remaining -= order_amount * order_price if match_with == "asks" else order_amount
        else:
            matched += amount_to_match if match_with == "asks" else amount_to_match * order_price
            remaining -= remaining
            break

    return matched


def kraken_orderbook_match(amount, coin, pair):
    orderbook = get(
        "https://api.kraken.com/0/public/Depth?pair=" +
        pair +
        "&count=500").json()
    pair_key = list(orderbook["result"].keys())[0]

    formatted_orderbook = {"asks": orderbook["result"][pair_key]["asks"],
                           "bids": orderbook["result"][pair_key]["bids"]}

    matched = generic_orderbook_match(amount, coin, pair, formatted_orderbook)

    return matched


def binance_orderbook_match(amount, coin, pair):
    orderbook = get(
        "https://www.binance.com/api/v3/depth?symbol=" +
        pair +
        "&limit=500").json()

    matched = generic_orderbook_match(amount, coin, pair, orderbook)

    return matched


def kraken_to_binance(msg_text):
    clean_msg = msg_text.replace("/krakentobinance", "").strip()
    user_input = clean_msg.split(",")

    usd_amount = float(user_input[0].strip())
    bank_fee = float(user_input[1].strip())

    kraken_fee = 0.998
    kraken_usdt_withdrawal = 2.5
    kraken_usdc_withdrawal = 7.5
    kraken_dai_withdrawal = 15

    kraken_executor = ThreadPoolExecutor(max_workers=5)
    kraken_usdt_thread = kraken_executor.submit(
        kraken_orderbook_match, usd_amount, "USD", "USDTUSD")
    kraken_usdc_thread = kraken_executor.submit(
        kraken_orderbook_match, usd_amount, "USD", "USDCUSD")
    kraken_dai_thread = kraken_executor.submit(
        kraken_orderbook_match, usd_amount, "USD", "DAIUSD")

    kraken_usdt_amount = kraken_usdt_thread.result()
    kraken_usdc_amount = kraken_usdc_thread.result()
    kraken_dai_amount = kraken_dai_thread.result()

    buy_usdt = round(
        kraken_usdt_amount * kraken_fee,
        4) - kraken_usdt_withdrawal
    buy_usdc = round(
        kraken_usdc_amount * kraken_fee,
        4) - kraken_usdc_withdrawal
    buy_dai = round(kraken_dai_amount * kraken_fee, 4) - kraken_dai_withdrawal

    binance_executor = ThreadPoolExecutor(max_workers=5)
    binance_usdtbusd_thread = binance_executor.submit(
        binance_orderbook_match, buy_usdt, "USDT", "BUSDUSDT")
    binance_usdcbusd_thread = binance_executor.submit(
        binance_orderbook_match, buy_usdc, "USDC", "USDCBUSD")
    binance_daibusd_thread = binance_executor.submit(
        binance_orderbook_match, buy_dai, "DAI", "BUSDDAI")

    binance_usdtbusd_amount = binance_usdtbusd_thread.result()
    binance_usdcbusd_amount = binance_usdcbusd_thread.result()
    binance_daibusd_amount = binance_daibusd_thread.result()

    buy_usdtbusd = round(binance_usdtbusd_amount, 4)
    buy_usdcbusd = round(binance_usdcbusd_amount, 4)
    buy_daibusd = round(binance_daibusd_amount, 4)

    usdt_percentage = round(
        (buy_usdt / (usd_amount + bank_fee) * 100 - 100) * (-1), 4)
    usdc_percentage = round(
        (buy_usdc / (usd_amount + bank_fee) * 100 - 100) * (-1), 4)
    dai_percentage = round(
        (buy_dai / (usd_amount + bank_fee) * 100 - 100) * (-1), 4)

    usdtbusd_percentage = round(
        (buy_usdtbusd / (usd_amount + bank_fee) * 100 - 100) * (-1), 4)
    usdcbusd_percentage = round(
        (buy_usdcbusd / (usd_amount + bank_fee) * 100 - 100) * (-1), 4)
    daibusd_percentage = round(
        (buy_daibusd / (usd_amount + bank_fee) * 100 - 100) * (-1), 4)

    results = [{"buy": "USDT",
                "get": "USDT",
                "amount": buy_usdt,
                "fee": usdt_percentage},
               {"buy": "USDC",
                "get": "USDC",
                "amount": buy_usdc,
                "fee": usdc_percentage},
               {"buy": "DAI",
                "get": "DAI",
                "amount": buy_dai,
                "fee": dai_percentage},
               {"buy": "USDT",
                "get": "BUSD",
                "amount": buy_usdtbusd,
                "fee": usdtbusd_percentage},
               {"buy": "USDC",
                "get": "BUSD",
                "amount": buy_usdcbusd,
                "fee": usdcbusd_percentage},
               {"buy": "DAI",
                "get": "BUSD",
                "amount": buy_daibusd,
                "fee": daibusd_percentage}]

    results.sort(key=lambda x: x.get("amount"), reverse=True)

    msg = f"""You sent {str(usd_amount).rstrip("0").rstrip(".")} USD to Kraken paying {str(bank_fee).rstrip("0").rstrip(".")} USD fee to your bank, spending {str(usd_amount + bank_fee).rstrip("0").rstrip(".")} USD in total.\n"""

    for coin in results:
        msg = f"""{msg}
Buying {coin["buy"]} in Kraken you can get {str(coin["amount"]).rstrip("0").rstrip(".")} {coin["get"]} in Binance ({str(coin["fee"]).rstrip("0").rstrip(".")}% total fee)."""

    return msg


def convert_base(msg_text):
    clean_msg = msg_text.replace("/convertbase", "").strip()
    user_input = clean_msg.split(",")

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

    if msg_text.startswith("/convertbase"):
        send_typing(token, chat_id)
        typing = True
        msg_to_send = convert_base(msg_text)

    if msg_text.startswith("/random"):
        send_typing(token, chat_id)
        typing = True
        msg_to_send = select_random(msg_text)

    if msg_text.startswith("/prices"):
        send_typing(token, chat_id)
        typing = True
        msg_to_send = get_prices(msg_text)

    if msg_text.startswith("/dolar"):
        send_typing(token, chat_id)
        typing = True
        msg_to_send = get_dolar()

    if msg_text.startswith("/krakentobinance"):
        send_typing(token, chat_id)
        typing = True
        msg_to_send = kraken_to_binance(msg_text)

    if not typing:
        try:
            reply_to = str(
                req["message"]["reply_to_message"]["from"]["username"])

            if reply_to != "respondedorbot" and msg_text.startswith(
                    "/ask") == False:
                return "ignored request"
        except BaseException:
            if chat_type != "private" and msg_text.startswith(
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
