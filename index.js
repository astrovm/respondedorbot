'use strict';

const request = require('request')
const schedule = require('node-schedule')
const TelegramBot = require('node-telegram-bot-api')
const respondedorbot = new TelegramBot(process.env.RESPONDEDORBOT_TELE_TOKEN, { polling: true })
const secwalbot = new TelegramBot(process.env.SECWALRBOT_TELE_TOKEN, { polling: true })
const sectmbot = new TelegramBot(process.env.SECTMBOT_TELE_TOKEN, { polling: true })
const secaybot = new TelegramBot(process.env.SECAYBOT_TELE_TOKEN, { polling: true })
const secpubot = new TelegramBot(process.env.SECPUBOT_TELE_TOKEN, { polling: true })

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

const poke = schedule.scheduleJob('*/5 * * * *', function(){
  request('https://bitfees.now.sh/')
  request('https://thegman.now.sh/')
})

module.exports = () => 'wake up mr. freeman'
