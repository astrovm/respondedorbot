'use strict';

let http = require('http'),
  schedule = require('node-schedule'),
  request = require('request'),
  nodemailer = require('nodemailer'),
  transporter = nodemailer.createTransport(),
  TelegramBot = require('node-telegram-bot-api'),
  respondedorbot = new TelegramBot(process.env.RESPONDEDORBOT_TELE_TOKEN, { polling: true }),
  Heroku = require('heroku-client'),
  heroku = new Heroku({ token: process.env.HEROKU_API_TOKEN }),
  rulerestart = new schedule.RecurrenceRule(),
  check15 = new schedule.RecurrenceRule(),
  check = function (title, app, url) {
    request(url, function (err, res) {
      if (!err && res.statusCode == 200) {
      } else {
        heroku.apps(app).dynos().restartAll();
        transporter.sendMail({
          from: process.env.AM,
          to: process.env.MM,
          text: title + ' DOWN'
        });
        console.log(title + ' DOWN');
      };
    });
  };

respondedorbot.on('message', function (msg) {
  let chatId = msg.chat.id;
  let text = msg.text;
  let opts = {
    reply_to_message_id: msg.message_id
  };
  let random = Math.round(Math.random());
  if (text.match(/^\//)) {
    if (text.match(/^\/ask/)) {
      if (random === 1) {
        respondedorbot.sendMessage(chatId, 'si', opts);
      } else {
        respondedorbot.sendMessage(chatId, 'no', opts);
      }
    }
  } else {
    if (random === 1) {
      respondedorbot.sendMessage(chatId, 'si', opts);
    } else {
      respondedorbot.sendMessage(chatId, 'no', opts);
    }
  }
});

rulerestart.minute = [25, 55];
check15.second = [0, 15, 30, 45];

let autorulerestart = schedule.scheduleJob(rulerestart, function(){
  request('http://thegman.herokuapp.com');
  heroku.apps('tuxifeed').dynos().restartAll();
});

let autocheck15 = schedule.scheduleJob(check15, function(){
  request('http://thegman.herokuapp.com');
  check('TuxiFeed', 'tuxifeed', 'http://tuxifeed.herokuapp.com');
  check('Linky', 'linkyurl', 'http://linkyurl.herokuapp.com');
});

let port = Number(process.env.PORT || 5000);
http.createServer(function (req, res) {
  res.writeHead(200, {'Content-Type': 'text/plain'});
  res.end('The G-Man is watching you.\n');
}).listen(port, function() {
  console.log('Listening on ' + port);
});
