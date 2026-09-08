# Клиент GigaChat: обёртка над SDK, кеш по хешу запроса.


class GigaChatClient:
    async def complete(self, system: str, messages: list[dict]) -> str:
        raise NotImplementedError
