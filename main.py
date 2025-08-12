import numpy as np
import matplotlib.pyplot as plt
from env import PowerMarketEnv
from agent import DQNAgent

# パラメータ
n_generators = 2
n_retailers = 2
n_prices = 10
n_demands = 10
n_episodes = 5000
n_slots = 48  # 1日48コマ
window = 100
threshold = 0.01  # 標準偏差の閾値

# 環境・エージェント初期化
env = PowerMarketEnv(n_generators=n_generators, n_retailers=n_retailers, n_prices=n_prices, n_demands=n_demands)
gen_agents = [DQNAgent(state_dim=1, action_dim=n_prices, epsilon=0.05, lr=1e-4) for _ in range(n_generators)]
ret_agents = [DQNAgent(state_dim=1, action_dim=n_demands, epsilon=0.05, lr=1e-4) for _ in range(n_retailers)]

avg_prices = []
avg_demands = []
avg_gen_rewards = []
avg_ret_rewards = []
slot_price_history = []  # コマごとの市場価格
slot_demand_history = [] # コマごとの需要

for episode in range(n_episodes):
    # 1日分の需要曲線（例: sin波＋ノイズ）
    base_demand = 5 + 3 * np.sin(np.linspace(0, 2 * np.pi, n_slots))
    noise = np.random.normal(0, 0.5, n_slots)
    demand_curve = base_demand + noise
    demand_curve = np.clip(demand_curve, 1, n_demands-1)

    state = env.reset()
    total_price = 0
    total_demand = 0
    total_gen_reward = 0
    total_ret_reward = 0
    slot_prices = []
    slot_demands = []
    for t in range(n_slots):
        # 小売事業者の需要行動を需要曲線で決定
        ret_actions = [int(demand_curve[t])] * n_retailers
        gen_actions = [agent.select_action([state[i]]) for i, agent in enumerate(gen_agents)]
        next_state, gen_rewards, ret_rewards, done, _ = env.step(gen_actions, ret_actions)
        for i, agent in enumerate(gen_agents):
            agent.store([state[i]], gen_actions[i], gen_rewards[i], [next_state[i]])
            agent.update()
        for j, agent in enumerate(ret_agents):
            agent.store([state[n_generators + j]], ret_actions[j], ret_rewards[j], [next_state[n_generators + j]])
            agent.update()
        state = next_state
        # 記録
        market_price = min(gen_actions) if sum(gen_actions) >= sum(ret_actions) else max(gen_actions)
        slot_prices.append(market_price)
        slot_demands.append(sum(ret_actions))
        total_price += market_price
        total_demand += sum(ret_actions)
        total_gen_reward += np.mean(gen_rewards)
        total_ret_reward += np.mean(ret_rewards)
    avg_prices.append(total_price / n_slots)
    avg_demands.append(total_demand / n_slots)
    avg_gen_rewards.append(total_gen_reward / n_slots)
    avg_ret_rewards.append(total_ret_reward / n_slots)
    slot_price_history.append(slot_prices)
    slot_demand_history.append(slot_demands)
    if episode % 100 == 0:
        print(f"Episode {episode}: Last Slot GenActions {gen_actions}, RetActions {ret_actions}")
        print(f"GenRewards {gen_rewards}, RetRewards {ret_rewards}")

# 収束判定機能
def judge_convergence(data, name, threshold):
    if len(data) < window:
        print(f"Not enough data for convergence check: {name}")
        return False
    std = np.std(data[-window:])
    print(f"Convergence check for {name}: std={std:.5f}")
    return std < threshold

price_converged = judge_convergence(avg_prices, "Price", threshold)
demand_converged = judge_convergence(avg_demands, "Demand", threshold)
gen_reward_converged = judge_convergence(avg_gen_rewards, "Generator Reward", threshold)
ret_reward_converged = judge_convergence(avg_ret_rewards, "Retailer Reward", threshold)

if price_converged and demand_converged and gen_reward_converged and ret_reward_converged:
    print("収束判定: すべての指標が収束しました！")
else:
    print("収束判定: いずれかの指標が収束していません。")

# グラフ描画
plt.figure(figsize=(14,10))
plt.subplot(2,2,1)
plt.plot(avg_prices)
plt.title('Average Market Price per Episode')
plt.xlabel('Episode')
plt.ylabel('Price')

plt.subplot(2,2,2)
plt.plot(avg_demands)
plt.title('Average Total Demand per Episode')
plt.xlabel('Episode')
plt.ylabel('Demand')

plt.subplot(2,2,3)
plt.plot(avg_gen_rewards)
plt.title('Average Generator Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Reward')

plt.subplot(2,2,4)
plt.plot(avg_ret_rewards)
plt.title('Average Retailer Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Reward')

plt.tight_layout()
plt.savefig("market_simulation_results.png")
print("Saved graph as market_simulation_results.png")

# 1日分のコマごとの市場価格推移（最終エピソード）
plt.figure(figsize=(10,4))
plt.plot(slot_price_history[-1], label='Market Price')
plt.plot(slot_demand_history[-1], label='Total Demand')
plt.title('Market Price and Demand per Slot (Last Episode)')
plt.xlabel('Slot (30min)')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.savefig("market_price_per_slot.png")
print("Saved graph as market_price_per_slot.png")
