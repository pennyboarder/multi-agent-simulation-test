import numpy as np
from tqdm import tqdm
from env import PowerMarketEnv
from agent import DQNAgent

def run_simulation(
    n_generators, n_retailers, n_prices, n_demands,
    days_per_episode, n_slots_per_day, n_episodes,
    gen_types, startup_costs, variable_costs, window, threshold
):
    env = PowerMarketEnv(
        n_generators=n_generators,
        n_prices=n_prices,
        n_demands=n_demands,
        gen_types=gen_types,
        startup_costs=startup_costs,
        variable_costs=variable_costs
    )
    gen_agents = [DQNAgent(state_dim=1, action_dim=n_prices, epsilon=0.05, lr=1e-4, batch_size=128) for _ in range(n_generators)]
    ret_agents = [DQNAgent(state_dim=1, action_dim=n_demands, epsilon=0.05, lr=1e-4, batch_size=128) for _ in range(n_retailers)]

    avg_prices = []
    avg_demands = []
    avg_gen_rewards = []
    avg_ret_rewards = []
    slot_price_history = []
    slot_demand_history = []
    slot_gen_actions_history = []
    n_slots = days_per_episode * n_slots_per_day

    # Generatorインスタンス生成
    from generator import Generator
    generator_list = []
    for gt in gen_types:
        if gt == "thermal":
            generator_list.append(Generator(gt, rated_output=10, min_rate=0.3, max_rate=1.0, min_up_time=3, ramp_rate=0.2))
        elif gt == "nuclear":
            generator_list.append(Generator(gt, rated_output=15, min_rate=0.3, max_rate=1.0, min_up_time=3, ramp_rate=0.2))
        elif gt == "pumped_storage":
            generator_list.append(Generator(gt, rated_output=8, min_rate=0.0, max_rate=1.0))
        elif gt == "solar":
            generator_list.append(Generator(gt, rated_output=5, min_rate=0.0, max_rate=1.0))
        else:
            generator_list.append(Generator(gt, rated_output=10))
    for episode in tqdm(range(n_episodes), desc='Training Episodes'):
        demand_curve = []
        slot_gen_actions = []
        for day in range(days_per_episode):
            base_demand = 17500 + 7500 * -np.cos(np.linspace(0, 2 * np.pi, n_slots_per_day))
            noise = np.random.normal(0, 1000, n_slots_per_day)
            day_curve = base_demand + noise
            day_curve = np.clip(day_curve, 10000, 25000)
            demand_curve.extend(day_curve)
        state = env.reset()
        total_price = 0
        total_demand = 0
        total_gen_reward = 0
        total_ret_reward = 0
        slot_prices = []
        slot_demands = []
        for t in range(n_slots):
            ret_actions = [int(demand_curve[t])] * n_retailers
            gen_actions = [agent.select_action(state[i]) for i, agent in enumerate(gen_agents)]
            for i, gt in enumerate(gen_types):
                hour = (t % n_slots_per_day) / 2
                current_day = t // n_slots_per_day
                action_val = gen_actions[i]
                gen_actions[i] = generator_list[i].compute_output(action_val, n_prices, hour=hour, current_day=current_day)
            total_gen = sum(gen_actions)
            total_demand = sum(ret_actions)
            if total_gen < total_demand:
                deficit = total_demand - total_gen
                adjustable_indices = [i for i, gt in enumerate(gen_types) if gen_actions[i] > 0 or gt != "solar"]
                if adjustable_indices:
                    add_per_gen = deficit // len(adjustable_indices)
                    remainder = deficit % len(adjustable_indices)
                    for idx in adjustable_indices:
                        gen_actions[idx] += add_per_gen
                    for i in range(remainder):
                        gen_actions[adjustable_indices[i]] += 1
            slot_gen_actions.append(gen_actions)
            next_state, gen_rewards, ret_rewards, done, _ = env.step(gen_actions, ret_actions)
            for i, agent in enumerate(gen_agents):
                agent.store(state[i], gen_actions[i], gen_rewards[i], next_state[i])
                agent.update()
            for j, agent in enumerate(ret_agents):
                clipped_action = min(max(int(ret_actions[j]), 0), n_demands-1)
                agent.store(state[n_generators + j], clipped_action, ret_rewards[j], next_state[n_generators + j])
                agent.update()
            state = next_state
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
        slot_gen_actions_history.append(slot_gen_actions)
    return avg_prices, avg_demands, avg_gen_rewards, avg_ret_rewards, slot_price_history, slot_demand_history, slot_gen_actions_history, n_slots
