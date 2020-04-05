from random import randint
from requests import get


def gen_random(name):
    randRes = randint(0, 1)
    randName = randint(0, 2)

    if randRes == 1:
        msj = "si"
    else:
        msj = "no"

    if randName == 1:
        msj = msj + " boludo"
    elif randName == 2:
        msj = msj + " " + name

    return msj


def responder(request):
    if request.method == "POST":
        req = request.get_json()
        token = str(request.args.get("token"))
        chat_id = str(req["message"]["chat"]["id"])
        message_id = str(req["message"]["message_id"])
        first_name = str(req["message"]["from"]["first_name"])
        msj = gen_random(first_name)
        url = 'https://api.telegram.org/bot' + token + '/sendMessage?chat_id=' + \
            chat_id + '&reply_to_message_id=' + message_id + '&text=' + msj
        get(url)
    return "ok"
