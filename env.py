import numpy as np

class PowerMarketEnv:
    def __init__(self, n_generators=2, n_retailers=2, n_prices=10, n_demands=10,
                 gen_types=None, startup_costs=None, variable_costs=None):
        self.n_generators = n_generators
        self.n_retailers = n_retailers
        self.n_prices = n_prices
        self.n_demands = n_demands
        # 発電機タイプ（"thermal" or "nuclear"）
        self.gen_types = gen_types if gen_types is not None else ["thermal"] * n_generators
        # 起動費（例: thermal=3.0, nuclear=5.0）
        self.startup_costs = startup_costs if startup_costs is not None else [3.0] * n_generators
        # 変動費（例: thermal=2.0, nuclear=0.5）
        self.variable_costs = variable_costs if variable_costs is not None else [2.0] * n_generators
        self.state = None
        self.prev_gen_on = [0] * n_generators  # 前コマの稼働状態（0:停止, 1:稼働）
        self.reset()

    def reset(self):
        self.state = np.zeros(self.n_generators + self.n_retailers)
        self.prev_gen_on = [0] * self.n_generators
        return self.state

    def step(self, gen_actions, ret_actions):
        # PV予測誤差（例: 平均0, 標準偏差1.0の正規分布ノイズ）
        pv_errors = np.random.normal(loc=0.0, scale=1.0, size=len(gen_actions))
        actual_supply = sum([max(a + e, 0) for a, e in zip(gen_actions, pv_errors)])
        demand = sum(ret_actions)
        # Market clearing price: if actual_supply >= demand, lowest generator price wins
        if actual_supply >= demand:
            market_price = min(gen_actions)
        else:
            market_price = max(gen_actions)
        # 稼働状態判定（価格>0なら稼働とみなす）
        gen_on = [1 if a > 0 else 0 for a in gen_actions]
        # Rewards: generators get profit, retailers get negative cost
        gen_rewards = []
        for i, a in enumerate(gen_actions):
            reward = max(market_price - a, 0)
            # 起動費考慮
            if gen_on[i] == 1 and self.prev_gen_on[i] == 0:
                reward -= self.startup_costs[i]
            # 変動費考慮（発電量に応じてコスト）
            reward -= self.variable_costs[i] * a
            gen_rewards.append(reward)
        ret_rewards = [-market_price * d for d in ret_actions]
        next_state = np.array(list(gen_actions) + list(ret_actions))
        self.prev_gen_on = gen_on  # 状態更新
        done = False
        return next_state, gen_rewards, ret_rewards, done, {}
