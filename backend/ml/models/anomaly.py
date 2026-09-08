# Аномалии: Isolation Forest + rolling z-score, детектор застывших сигналов и out-of-envelope.


class AnomalyDetector:
    def score(self, state: dict) -> dict:
        raise NotImplementedError
