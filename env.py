import numpy as np

class PowerMarketEnv:
    def __init__(self, n_generators=2, n_retailers=2, n_prices=10, n_demands=10):
        self.n_generators = n_generators
        self.n_retailers = n_retailers
        self.n_prices = n_prices
        self.n_demands = n_demands
        self.state = None
        self.reset()

    def reset(self):
        self.state = np.zeros(self.n_generators + self.n_retailers)
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
        # Rewards: generators get profit, retailers get negative cost
        gen_rewards = [max(market_price - a, 0) for a in gen_actions]
        ret_rewards = [-market_price * d for d in ret_actions]
        next_state = np.array(list(gen_actions) + list(ret_actions))
        done = False
        return next_state, gen_rewards, ret_rewards, done, {}
