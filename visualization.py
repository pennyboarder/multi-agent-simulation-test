import matplotlib.pyplot as plt
import numpy as np

def plot_results(avg_prices, avg_demands, avg_gen_rewards, avg_ret_rewards, result_dir='result'):
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
    outpath = f"{result_dir}/market_simulation_results.png"
    plt.savefig(outpath)
    print(f"Saved graph as {outpath}")

def plot_generation_mix(slot_gen_actions_history, gen_types, n_slots, result_dir='result'):
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
    # x軸を24時間表記に
    slot_ticks = np.arange(0, n_slots+1, 4)  # 2時間ごと（48コマ/日）
    hour_labels = [(i//2)%24 for i in slot_ticks]
    ax.set_xticks(slot_ticks)
    ax.set_xticklabels(hour_labels)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Generation (Market Price x Count, approx)')
    ax.set_title('Generation Mix per Slot (Stacked, Last Episode)')
    ax.legend()
    ax.grid(True, which='both', axis='x', linestyle='--', alpha=0.5)
    fig.tight_layout()
    outpath = f"{result_dir}/generation_mix_per_slot.png"
    fig.savefig(outpath)
    print(f"Saved graph as {outpath}")
