# respondedorbot

soy el gordo, un bot basado en el atendedor de boludos que labura 24/7 respondiendo boludeces en telegram

## che gordo como te agrego?

mandate aca capo [@respondedorbot](https://t.me/respondedorbot)

## que onda, que sabes hacer?

- te contesto cualquier gilada que me preguntes con /ask, /pregunta, /che o /gordo
- te tiro la posta de los precios crypto con /prices (o /precio, /precios, /presio, /presios)
- te digo como esta el dolar blue y toda la movida con /dolar o /dollar
- te calculo el arbitraje entre tarjeta y crypto con /devo
- te tiro el precio justo de btc segun powerlaw y rainbow con /powerlaw y /rainbow
- te elijo random entre pizza o hamburguesa con /random
- te convierto texto a comando telegram con /comando o /command
- te paso numeros entre bases con /convertbase
- te tiro la hora unix con /time
- y bocha de giladas mas, mandate /help y te cuento todo

## como te deployas?

necesitas estas variables de entorno:

```
TELEGRAM_TOKEN: el token del bot de telegram
TELEGRAM_USERNAME: mi nombre de usuario para saber cuando me hablan
ADMIN_CHAT_ID: el chat id del admin para mandarle reports
COINMARKETCAP_KEY: key de la api de coinmarketcap para los precios crypto
REDIS_HOST: el host de redis
REDIS_PORT: el puerto de redis
REDIS_PASSWORD: el password de redis si tenes
CURRENT_FUNCTION_URL: la url donde estoy corriendo
MAIN_FUNCTION_URL: la url principal donde deberia estar
FRIENDLY_INSTANCE_NAME: un nombre piola para identificarme en los reports
OPENROUTER_API_KEY: la key de openrouter para que te pueda bardear como corresponde
CLOUDFLARE_API_KEY: la key de cloudflare workers ai para fallback cuando openrouter falla
CLOUDFLARE_ACCOUNT_ID: el account id de cloudflare para acceder a workers ai
GORDO_KEY: key para autenticar los requests al webhook
```

## como seteas el webhook?

mandate esta url master:

`{function_url}/?update_webhook=true&key={gordo_key}`

o si queres checkear que onda:

`{function_url}/?check_webhook=true&key={gordo_key}`

## licencia

hace lo que se te cante el orto, total es todo codigo choreado de stackoverflow

## creditos

creditos al gordo astro que me programo mientras se bajaba una birra y puteaba a javascript
