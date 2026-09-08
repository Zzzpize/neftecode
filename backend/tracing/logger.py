# Один цикл принятия решения = один decision_id, все сообщения агентов связаны им.


class AgentTracer:
    def record(self, decision_id: str, agent: str, input_hash: str, output: dict, duration_ms: float) -> None:
        raise NotImplementedError

    def get_trace(self, decision_id: str) -> list[dict]:
        raise NotImplementedError
