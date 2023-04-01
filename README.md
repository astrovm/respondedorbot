# respondedorbot

Telegram bot running on a Raspberry Pi that answers questions and provides useful information

https://t.me/respondedorbot

## Set webhook

https://api.telegram.org/bot{bot_token}/setWebhook?url={function_url}?token={encrypted_bot_token}

## Environment variables

- TELEGRAM_TOKEN_HASH: The SHA256 hash of the decrypted Telegram token, used for verifying the authenticity of the token.
- TELEGRAM_TOKEN_KEY: The key used to decrypt the Fernet encrypted Telegram token.
- TELEGRAM_USERNAME: The username of the bot, used to identify if a command is directed at the bot.
- ADMIN_CHAT_ID: The chat ID of the bot administrator to receive any admin reports or notifications.
- COINMARKETCAP_KEY: The API key for accessing CoinMarketCap's API for cryptocurrency data.
- REDIS_HOST: The hostname or IP address of the Redis server.
- REDIS_PORT: The port number on which the Redis server is running.
- REDIS_PASSWORD: The password for authenticating with the Redis server (if required).

## License

[MIT License](/LICENSE)
