# respondedorbot

Telegram bot running on a Raspberry Pi that answers questions and provides useful information. In case the Raspberry Pi goes down, the bot automatically switches to an instance in Vercel.

<https://t.me/respondedorbot>

## Set webhook

To set the webhook, use this URL:

`{function_url}/?update_webhook=true&token={encrypted_token}`

## Environment variables

The bot requires several environment variables to function properly:

```
TELEGRAM_TOKEN_HASH: The SHA256 hash of the decrypted Telegram token, used for verifying the authenticity of the token.
TELEGRAM_TOKEN_KEY: The key used to decrypt the Fernet encrypted Telegram token.
TELEGRAM_USERNAME: The username of the bot, used to identify if a command is directed at the bot.
ADMIN_CHAT_ID: The chat ID of the bot administrator to receive any admin reports or notifications.
COINMARKETCAP_KEY: The API key for accessing CoinMarketCap's API for cryptocurrency data.
REDIS_HOST: The hostname or IP address of the Redis server.
REDIS_PORT: The port number on which the Redis server is running.
REDIS_PASSWORD: The password for authenticating with the Redis server (if required).
CURRENT_FUNCTION_URL: The URL of the current instance.
MAIN_FUNCTION_URL: The URL of the primary instance that should be used if it's not down.
FRIENDLY_INSTANCE_NAME: A human-readable name for the current instance, used in admin reports and notifications to identify the source of the message.
```

## License

[MIT License](/LICENSE)
