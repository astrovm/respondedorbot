const express = require('express')
const request = require('request')
const bodyParser = require('body-parser')
const TOKEN = process.env.TELEGRAM_TOKEN

const app = express()
module.exports = app

app.use(bodyParser.json())

// curl --request POST --url https://api.telegram.org/botTOKEN/setWebhook --header 'content-type: application/json' --data '{"url": "https://URL.now.sh/bot?token=TOKEN"}'
app.post('*', (req, res) => {
  if (req.query.token === TOKEN) {
    const chatId = req.body.message.chat.id
    const text = req.body.message.text
    console.log(chatId)
    console.log(text)
    request.post({
      uri: `https://api.telegram.org/bot${TOKEN}/sendMessage`,
      json: true,
      body: { text: 'hola', chat_id: chatId }
    })
    console.log(req.body)
  }
  res.status(200).send('boludo')
})
