# Пока прокси-метрики: отклонения от нормальных зон, скорости изменения температур.


class ReliabilityAgent:
    def assess(self, state: dict) -> dict:
        raise NotImplementedError
