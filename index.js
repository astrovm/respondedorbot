'use strict'

const TelegramBot = require('node-telegram-bot-api')
const schedule = require('node-schedule')
const request = require('request')

class AddBot {
  constructor (token, yes, no) {
    this.token = token
    this.yes = yes
    this.no = no
  }

  init () {
    const bot = new TelegramBot(this.token, { polling: true })
    bot.on('message', (msg) => {
      const chatId = msg.chat.id
      const opts = {
        reply_to_message_id: msg.message_id
      }
      const random = Math.round(Math.random())
      if (random === 1) {
        bot.sendMessage(chatId, this.yes, opts)
      } else {
        bot.sendMessage(chatId, this.no, opts)
      }
    })
  }
}

const respondedorbot = new AddBot(process.env.RESPONDEDORBOT_TELE_TOKEN, 'si', 'no')
const secwalbot = new AddBot(process.env.SECWALRBOT_TELE_TOKEN, 'si', 'no')
const sectmbot = new AddBot(process.env.SECTMBOT_TELE_TOKEN, 'guau si', 'guau no')
const secaybot = new AddBot(process.env.SECAYBOT_TELE_TOKEN, 'miau si', 'miau no')
const secpubot = new AddBot(process.env.SECPUBOT_TELE_TOKEN, 'RAFRAFAYAYASNJCRA si', 'RAFRAFAYAYASNJCRA no')

respondedorbot.init()
secwalbot.init()
sectmbot.init()
secaybot.init()
secpubot.init()

schedule.scheduleJob('*/5 * * * *', () => {
  request('https://bitfees.now.sh/')
  request('https://thegman.now.sh/')
  request('https://astro-bots.now.sh/')
})

module.exports = () => 'wake up mr. freeman'
