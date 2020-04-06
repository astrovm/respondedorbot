from time import sleep
from random import randint, uniform
from requests import get


def gen_random(name):
    sleep(uniform(0, 1))

    randRes = randint(0, 1)
    randName = randint(0, 2)

    if randRes == 1:
        msj = "si"
    else:
        msj = "no"

    if randName == 1:
        msj = msj + " boludo"
        sleep(uniform(0, 1))
    elif randName == 2:
        msj = msj + " " + name
        sleep(uniform(0, 1))

    return msj


def responder(request):
    try:
        if request.method == "POST":
            token = str(request.args.get("token"))

            req = request.get_json()
            chat_id = str(req["message"]["chat"]["id"])

            url = 'https://api.telegram.org/bot' + token + \
                '/sendChatAction?chat_id=' + chat_id + '&action=typing'

            get(url)

            message_id = str(req["message"]["message_id"])
            first_name = str(req["message"]["from"]["first_name"])

            msj = gen_random(first_name)

            url = 'https://api.telegram.org/bot' + token + '/sendMessage?chat_id=' + \
                chat_id + '&reply_to_message_id=' + message_id + '&text=' + msj

            get(url)

            return "ok"
        else:
            return "bad request"
    except:
        return "unexpected error"
