from time import sleep
from random import randint, uniform
from requests import get


def gen_random(name):
    sleep(uniform(0, 1))

    randRes = randint(0, 1)
    randName = randint(0, 2)

    if randRes == 1:
        msg = "si"
    else:
        msg = "no"

    if randName == 1:
        msg = msg + " boludo"
        sleep(uniform(0, 1))
    elif randName == 2:
        msg = msg + " " + name
        sleep(uniform(0, 1))

    return msg


def get_prices():
    prices = get(
        "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin%2Cethereum&vs_currencies=usd").json()

    btc = round(float(prices["bitcoin"]["usd"]))
    eth = round(float(prices["ethereum"]["usd"]))

    msg = f"""BTC: {btc} USD
ETH: {eth} USD"""
    return msg


def send_typing(token, chat_id):
    url = "https://api.telegram.org/bot" + token + \
        "/sendChatAction?chat_id=" + chat_id + "&action=typing"
    get(url)


def send_msg(token, chat_id, msg_id, msg):
    url = "https://api.telegram.org/bot" + token + "/sendMessage?chat_id=" + \
        chat_id + "&reply_to_message_id=" + msg_id + "&text=" + msg
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

            if msg_text.startswith("/prices"):
                send_typing(token, chat_id)
                msg = get_prices()
                send_msg(token, chat_id, msg_id, msg)
                return "ok"

            try:
                reply_to = str(
                    req["message"]["reply_to_message"]["from"]["username"])

                if reply_to != "respondedorbot":
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
