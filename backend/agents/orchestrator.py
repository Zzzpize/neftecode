# Порядок цикла:
#   data_agent -> quality + reliability -> optimizer -> ранжирование -> ответ.
# Приоритет: безопасность, потом качество, потом экономика.


class Orchestrator:
    async def decide(self, timestamp) -> dict:
        raise NotImplementedError
