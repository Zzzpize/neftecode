# Парето-оптимизатор через Optuna. Метрики: сера, выход, энергия, тяжесть режима.


class ParetoOptimizer:
    def find_pareto(self, state: dict, constraints: dict, n_variants: int = 10) -> list[dict]:
        raise NotImplementedError
