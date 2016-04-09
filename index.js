var http = require('http'),
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
  var chatId = msg.chat.id;
  var opts = {
    reply_to_message_id: msg.message_id
  };
  var random = Math.round(Math.random());
  if (random === 1) {
    respondedorbot.sendMessage(chatId, 'si', opts);
  } else {
    respondedorbot.sendMessage(chatId, 'no' + msg.text, opts);
  }
});

rulerestart.minute = [25, 55];
check15.second = [0, 15, 30, 45];

var autorulerestart = schedule.scheduleJob(rulerestart, function(){
  request('http://thegman.herokuapp.com');
  heroku.apps('tuxifeed-duckinto-bugfixes4454').dynos().restartAll();
});

var autocheck15 = schedule.scheduleJob(check15, function(){
  check('TuxiFeed', 'tuxifeed-duckinto-bugfixes4454', 'http://tuxifeed-duckinto-bugfixes4454.herokuapp.com');
});

var port = Number(process.env.PORT || 5000);
http.createServer(function (req, res) {
  res.writeHead(200, {'Content-Type': 'text/plain'});
  res.end('The G-Man is watching you.\n');
}).listen(port, function() {
  console.log('Listening on ' + port);
});
