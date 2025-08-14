# --- 新しい分割構成 ---

import argparse
import os
import pickle

import yaml

from simulation import run_simulation
from utils import judge_convergence
from visualization import plot_generation_mix, plot_results

parser = argparse.ArgumentParser()
parser.add_argument(
    "--result_dir", type=str, default="result", help="Directory to save results"
)
args = parser.parse_args()
result_dir = args.result_dir
os.makedirs(result_dir, exist_ok=True)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
n_generators = config["n_generators"]
n_retailers = config["n_retailers"]
n_prices = config["n_prices"]
n_demands = config["n_demands"]
days_per_episode = config["days_per_episode"]
n_slots_per_day = config["n_slots_per_day"]
n_episodes = config["n_episodes"]
window = config["window"]
threshold = config["threshold"]
gen_types = config["gen_types"]
startup_costs = config["startup_costs"]
variable_costs = config["variable_costs"]

(
    avg_prices,
    avg_demands,
    avg_gen_rewards,
    avg_ret_rewards,
    slot_price_history,
    slot_demand_history,
    slot_gen_actions_history,
    n_slots,
) = run_simulation(
    n_generators,
    n_retailers,
    n_prices,
    n_demands,
    days_per_episode,
    n_slots_per_day,
    n_episodes,
    gen_types,
    startup_costs,
    variable_costs,
    window,
    threshold,
)


with open(os.path.join(result_dir, "simulation_results.pkl"), "wb") as f:
    pickle.dump(
        {
            "avg_prices": avg_prices,
            "avg_demands": avg_demands,
            "avg_gen_rewards": avg_gen_rewards,
            "avg_ret_rewards": avg_ret_rewards,
            "slot_price_history": slot_price_history,
            "slot_demand_history": slot_demand_history,
            "slot_gen_actions_history": slot_gen_actions_history,
            "n_slots": n_slots,
        },
        f,
    )

price_converged = judge_convergence(avg_prices, "Price", window, threshold)
demand_converged = judge_convergence(avg_demands, "Demand", window, threshold)
gen_reward_converged = judge_convergence(
    avg_gen_rewards, "Generator Reward", window, threshold
)
ret_reward_converged = judge_convergence(
    avg_ret_rewards, "Retailer Reward", window, threshold
)

if (
    price_converged
    and demand_converged
    and gen_reward_converged
    and ret_reward_converged
):
    print("収束判定: すべての指標が収束しました！")
else:
    print("収束判定: いずれかの指標が収束していません。")

plot_results(avg_prices, avg_demands, avg_gen_rewards, avg_ret_rewards, result_dir)
plot_generation_mix(slot_gen_actions_history, gen_types, n_slots, result_dir)
