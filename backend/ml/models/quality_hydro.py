# Прогноз серы, T50, T90, D15 после гидроочистки.
# Таргет - ПАК Mg.Sulfur.Q (10-мин), ЛИМС используется как калибратор.


class QualityHydroModel:
    def predict(self, state: dict) -> dict:
        raise NotImplementedError

    def predict_after_action(self, state: dict, action: dict) -> dict:
        raise NotImplementedError

    def explain(self, state: dict) -> dict:
        raise NotImplementedError
