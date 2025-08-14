import matplotlib.pyplot as plt
import numpy as np

def plot_results(avg_prices, avg_demands, avg_gen_rewards, avg_ret_rewards):
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

def plot_generation_mix(slot_gen_actions_history, gen_types, n_slots):
    gen_type_list = ["thermal", "nuclear", "pumped_storage", "solar", "pumped_storage_charge"]
    gen_type_indices = {t: [i for i, gt in enumerate(gen_types) if gt == t] for t in ["thermal", "nuclear", "pumped_storage", "solar"]}
    gen_type_generation = {t: [] for t in gen_type_list}
    last_gen_actions = slot_gen_actions_history[-1]
    for slot in range(len(last_gen_actions)):
        for t in ["thermal", "nuclear", "solar"]:
            indices = gen_type_indices[t]
            total_gen = sum([last_gen_actions[slot][i] for i in indices if i < len(last_gen_actions[slot])])
            gen_type_generation[t].append(total_gen)
        indices = gen_type_indices["pumped_storage"]
        discharge = sum([last_gen_actions[slot][i] for i in indices if i < len(last_gen_actions[slot]) and last_gen_actions[slot][i] > 0])
        charge = -sum([last_gen_actions[slot][i] for i in indices if i < len(last_gen_actions[slot]) and last_gen_actions[slot][i] < 0])
        gen_type_generation["pumped_storage"].append(discharge)
        gen_type_generation["pumped_storage_charge"].append(charge)
    fig, ax = plt.subplots(figsize=(12,5))
    bottom = np.zeros(n_slots)
    colors = {"thermal": "#d62728", "nuclear": "#1f77b4", "pumped_storage": "#2ca02c", "solar": "#ff7f0e", "pumped_storage_charge": "#9467bd"}
    for t in ["thermal", "nuclear", "pumped_storage", "solar"]:
        ax.bar(np.arange(n_slots), gen_type_generation[t], bottom=bottom, label=t.capitalize(), color=colors[t])
        bottom += np.array(gen_type_generation[t])
    ax.bar(np.arange(n_slots), -np.array(gen_type_generation["pumped_storage_charge"]), bottom=bottom, label="PumpedStorage Charge", color=colors["pumped_storage_charge"])
    ax.set_xlabel('Slot (30min)')
    ax.set_ylabel('Generation (Market Price x Count, approx)')
    ax.set_title('Generation Mix per Slot (Stacked, Last Episode)')
    ax.legend()
    fig.tight_layout()
    fig.savefig("generation_mix_per_slot.png")
    print("Saved graph as generation_mix_per_slot.png")
