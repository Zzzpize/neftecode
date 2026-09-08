# Прогноз T50, T90, D15, ПТФ для фракции 240-350 на выходе АВТ.
# База - формулы ВАК, поверх LightGBM на резидуалах, квантильная регрессия.


class QualityAVTModel:
    def predict(self, state: dict) -> dict:
        raise NotImplementedError

    def predict_after_action(self, state: dict, action: dict) -> dict:
        raise NotImplementedError

    def explain(self, state: dict) -> dict:
        raise NotImplementedError
