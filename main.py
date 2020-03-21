import os
import telegram
import random


def gen_random():
    rand = random.randint(0, 1)
    if rand == 1:
        msj = "si"
    else:
        msj = "no"
    return msj


def responder(request):
    if request.method == "POST" and request.args.get('token') == os.environ["TELEGRAM_TOKEN"]:
        bot = telegram.Bot(token=os.environ["TELEGRAM_TOKEN"])
        update = telegram.Update.de_json(request.get_json(force=True), bot)
        chat_id = update.message.chat.id
        msj = gen_random()
        bot.sendMessage(chat_id=chat_id, text=msj)
    return "ok"
