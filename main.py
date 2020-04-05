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
    if request.method == "POST":
        bot = telegram.Bot(token=request.args.get('token'))
        update = telegram.Update.de_json(request.get_json(force=True), bot)
        chat_id = update.message.chat.id
        message_id = update.message.message_id
        msj = gen_random()
        bot.sendMessage(chat_id=chat_id,
                        reply_to_message_id=message_id, text=msj)
    return "ok"
