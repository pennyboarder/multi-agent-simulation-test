
# --- 新しい分割構成 ---
from simulation import run_simulation
from visualization import plot_results, plot_generation_mix
from utils import judge_convergence

# パラメータ
n_generators = 20
n_retailers = 2
n_prices = 10
n_demands = 10
days_per_episode = 7
n_slots_per_day = 48
n_episodes = 500
window = 100
threshold = 0.01
gen_types = ["thermal"]*11 + ["nuclear"]*3 + ["pumped_storage"]*3 + ["solar"]*3
startup_costs = [30]*11 + [100]*3 + [10]*3 + [0]*3
variable_costs = [8]*11 + [1]*3 + [0.8]*3 + [0.05]*3

avg_prices, avg_demands, avg_gen_rewards, avg_ret_rewards, slot_price_history, slot_demand_history, slot_gen_actions_history, n_slots = run_simulation(
    n_generators, n_retailers, n_prices, n_demands,
    days_per_episode, n_slots_per_day, n_episodes,
    gen_types, startup_costs, variable_costs, window, threshold
)

price_converged = judge_convergence(avg_prices, "Price", window, threshold)
demand_converged = judge_convergence(avg_demands, "Demand", window, threshold)
gen_reward_converged = judge_convergence(avg_gen_rewards, "Generator Reward", window, threshold)
ret_reward_converged = judge_convergence(avg_ret_rewards, "Retailer Reward", window, threshold)

if price_converged and demand_converged and gen_reward_converged and ret_reward_converged:
    print("収束判定: すべての指標が収束しました！")
else:
    print("収束判定: いずれかの指標が収束していません。")

plot_results(avg_prices, avg_demands, avg_gen_rewards, avg_ret_rewards)
plot_generation_mix(slot_gen_actions_history, gen_types, n_slots)
