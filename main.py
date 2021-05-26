from time import sleep
from random import randint, uniform
from requests import get
from urllib.parse import quote


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
    dollars = get("https://criptoya.com/api/dolar").json()
    usdc = get("https://criptoya.com/api/usdc/ars/1000").json()
    dai = get("https://criptoya.com/api/dai/ars/1000").json()
    usdt = get("https://criptoya.com/api/usdt/ars/1000").json()

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
        {"name": "MEP", "price": dollars["mep"]},
        {"name": "CCL", "price": dollars["ccl"]},
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


def send_typing(token, chat_id):
    url = "https://api.telegram.org/bot" + token + \
        "/sendChatAction?chat_id=" + chat_id + "&action=typing"
    get(url)


def send_msg(token, chat_id, msg_id, msg):
    url = "https://api.telegram.org/bot" + token + "/sendMessage?chat_id=" + \
        chat_id + "&reply_to_message_id=" + \
        msg_id + "&text=" + quote(msg, safe='/')
    get(url)


def responder(request):
    try:
        if request.method == "POST":
            token = str(request.args.get("token"))

            req = request.get_json()

            msg_text = str(req["message"]["text"])
            msg_id = str(req["message"]["message_id"])
            chat_id = str(req["message"]["chat"]["id"])
            chat_type = str(req["message"]["chat"]["type"])
            first_name = str(req["message"]["from"]["first_name"])

            if msg_text.startswith("/random"):
                send_typing(token, chat_id)
                msg = select_random(msg_text)
                send_msg(token, chat_id, msg_id, msg)
                return "ok"

            if msg_text.startswith("/prices"):
                send_typing(token, chat_id)
                msg = get_prices(msg_text)
                send_msg(token, chat_id, msg_id, msg)
                return "ok"

            if msg_text.startswith("/dolar"):
                send_typing(token, chat_id)
                msg = get_dolar()
                send_msg(token, chat_id, msg_id, msg)
                return "ok"

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
            msg = gen_random(first_name)
            send_msg(token, chat_id, msg_id, msg)
            return "ok"
        else:
            return "bad request"
    except BaseException:
        return "unexpected error"
