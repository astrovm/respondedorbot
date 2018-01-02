'use strict';

const http = require('http'),
  TelegramBot = require('node-telegram-bot-api'),
  respondedorbot = new TelegramBot(process.env.RESPONDEDORBOT_TELE_TOKEN, { polling: true }),
  secwalbot = new TelegramBot(process.env.SECWALRBOT_TELE_TOKEN, { polling: true }),
  sectmbot = new TelegramBot(process.env.SECTMBOT_TELE_TOKEN, { polling: true }),
  secaybot = new TelegramBot(process.env.SECAYBOT_TELE_TOKEN, { polling: true }),
  secpubot = new TelegramBot(process.env.SECPUBOT_TELE_TOKEN, { polling: true })

respondedorbot.on('message', function (msg) {
  const chatId = msg.chat.id;
  const opts = {
    reply_to_message_id: msg.message_id
  };
  const random = Math.round(Math.random());
  if (random === 1) {
    respondedorbot.sendMessage(chatId, 'si', opts)
  } else {
    respondedorbot.sendMessage(chatId, 'no', opts)
  };
});


secwalbot.on('message', function (msg) {
  const chatId = msg.chat.id;
  const opts = {
    reply_to_message_id: msg.message_id
  }
  const random = Math.round(Math.random());
  if (random === 1) {
    secwalbot.sendMessage(chatId, 'si', opts)
  } else {
    secwalbot.sendMessage(chatId, 'no', opts)
  }
})

sectmbot.on('message', function (msg) {
  const chatId = msg.chat.id;
  const opts = {
    reply_to_message_id: msg.message_id
  }
  const random = Math.round(Math.random());
  if (random === 1) {
    sectmbot.sendMessage(chatId, 'guau si', opts)
  } else {
    sectmbot.sendMessage(chatId, 'guau no', opts)
  }
})

secaybot.on('message', function (msg) {
  const chatId = msg.chat.id;
  const opts = {
    reply_to_message_id: msg.message_id
  }
  const random = Math.round(Math.random());
  if (random === 1) {
    secaybot.sendMessage(chatId, 'miau si', opts)
  } else {
    secaybot.sendMessage(chatId, 'miau no', opts)
  }
})


secpubot.on('message', function (msg) {
  const chatId = msg.chat.id;
  const opts = {
    reply_to_message_id: msg.message_id
  }
  const random = Math.round(Math.random());
  if (random === 1) {
    secpubot.sendMessage(chatId, 'RAFRAFAYAYASNJCRA si', opts)
  } else {
    secpubot.sendMessage(chatId, 'RAFRAFAYAYASNJCRA no', opts)
  }
})

const port = Number(process.env.PORT || 5000);
http.createServer(function (req, res) {
  res.writeHead(200, {'Content-Type': 'text/plain'})
  res.end('The G-Man is watching you.\n')
}).listen(port, function() {
  console.log('Listening on ' + port)
})
