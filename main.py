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


def get_prices():
    prices = get(
        "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin%2Cethereum&vs_currencies=usd").json()

    btc = round(float(prices["bitcoin"]["usd"]))
    eth = round(float(prices["ethereum"]["usd"]))

    msg = f"""BTC: {btc} USD
ETH: {eth} USD"""
    return msg


def get_dolar():
    dolar = get("https://criptoya.com/api/dolar").json()
    dai = get("https://criptoya.com/api/dai/ars/1000").json()
    dai_ask = 0

    for exchange in dai:
        if float(dai[exchange]["totalAsk"]) < float(dai_ask) or dai_ask == 0:
            dai_ask = str(dai[exchange]["totalAsk"])

    msg = f"""Oficial: {dolar["oficial"]}
Solidario: {dolar["solidario"]}
MEP: {dolar["mep"]}
CCL: {dolar["ccl"]}
Bitcoin: {dolar["ccb"]}
Blue: {dolar["blue"]}
DAI: {dai_ask}"""

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
                msg = get_prices()
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

                if reply_to != "respondedorbot" and msg_text.startswith("/ask") == False:
                    return "ignored request"
            except:
                if chat_type != "private" and msg_text.startswith("/ask") == False:
                    return "ignored request"

            send_typing(token, chat_id)
            msg = gen_random(first_name)
            send_msg(token, chat_id, msg_id, msg)
            return "ok"
        else:
            return "bad request"
    except:
        return "unexpected error"
