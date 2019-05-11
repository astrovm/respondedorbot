const express = require('express')
const request = require('request')
const bodyParser = require('body-parser')

let bots = {}
bots[process.env.RESPONDEDORBOT_TELE_TOKEN] = { yes: 'si', no: 'no' }
bots[process.env.SECAYBOT_TELE_TOKEN] = { yes: 'miau si', no: 'miau no' }
bots[process.env.SECKIBOT_TELE_TOKEN] = { yes: 'miau si', no: 'miau no' }
bots[process.env.SECPUBOT_TELE_TOKEN] = { yes: 'RAFRAFAYAYASNJCRA si', no: 'RAFRAFAYAYASNJCRA no' }
bots[process.env.SECTMBOT_TELE_TOKEN] = { yes: 'guau si', no: 'guau no' }
bots[process.env.SECWALRBOT_TELE_TOKEN] = { yes: 'si', no: 'no' }

const app = express()
module.exports = app

app.use(bodyParser.json())

app.post('*', (req, res) => {
  const token = req.query.token
  const answer = bots[token]
  if (answer) {
    const chatId = req.body.message.chat.id
    const random = Math.round(Math.random())
    if (random === 1) {
      request.post({
        uri: `https://api.telegram.org/bot${token}/sendMessage`,
        json: true,
        body: { text: answer.yes, chat_id: chatId }
      })
    } else {
      request.post({
        uri: `https://api.telegram.org/bot${token}/sendMessage`,
        json: true,
        body: { text: answer.no, chat_id: chatId }
      })
    }
  }
  res.status(200).send('boludo')
})
