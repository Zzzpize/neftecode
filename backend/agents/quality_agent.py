class QualityAgent:
    def forecast(self, state: dict) -> dict:
        raise NotImplementedError

    def forecast_after_action(self, state: dict, action: dict) -> dict:
        raise NotImplementedError
