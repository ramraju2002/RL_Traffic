# import time
# import gymnasium as gym
# from traffic_environment import TrafficEnv
# import rl_planners
# import numpy as np

# # define rewards function
# rewards = {"state": 0}

# # initialize the environment with rewards and max_steps
# env = TrafficEnv(rewards = rewards, max_steps=1000)

# # set the RL algorithm to plan or train an agent
# rl_algo = "Value Iteration"

# # initialize the agent and train it
# if rl_algo == "Value Iteration":
#     agent = rl_planners.ValueIterationPlanner(env)
# elif rl_algo == "Policy Iteration":
#     agent = rl_planners.PolicyIterationPlanner(env)


# # TODO: Initialize variables to track performance metrics
# # Metrics to include:
# # 1. Count of instances where car count exceeds critical thresholds (N total cars or M in any direction)
# # 2. Average number of cars waiting at the intersection in all directions during a time period
# # 3. Maximum continuous time where car count remains below critical thresholds


# # reset the environment and get the initial observation
# observation, info = env.reset(seed=42), {}
# np.random.seed(42)
# env.action_space.seed(42)

# # TODO: Initialize variables to track environment metrics
# # Example: cumulative rewards, episode duration, etc.

# # set light state variables
# RED, GREEN = 0, 1

# # run the environment until terminated or truncated
# terminated, truncated = False, False
# while (not terminated and not truncated):
#     # use the agent's policy to choose an action
#     action = agent.choose_action(observation)
#     # step through the environment with the chosen action
#     observation, reward, terminated, truncated, info = env.step(action)

#     # TODO: Update variables to calculate performance and environment metrics based on the new observation

#     # unpack the state to get the number of cars and traffic light state
#     ns, ew, light = tuple(observation)
#     light_color = "GREEN" if light == GREEN else "RED"
#     # print the current state
#     print(f"Step: x, NS Cars: {ns}, EW Cars: {ew}, Light NS: {light_color}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
#     # print(f"Step: {_}, NS Cars: {ns}, EW Cars: {ew}, Light NS: {light_color}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
#     # render the environment at each step
#     env.render()
#     # add a delay to slow down the rendering for better visualization
#     time.sleep(0.8)

#     # reset the environment if terminated or truncated
#     if terminated or truncated:
#         print("\nTERMINATED OR TRUNCATED, RESETTING...\n")

#         # TODO: Update metrics for completed episode

#         observation, info = env.reset(), {}

#         # TODO: Reset tracking variables for the new episode

#         terminated, truncated = False, False

# # close the environment
# env.render(close=True)

# # TODO: Evaluate performance based on high-level metrics

# print("\n=== PERFORMANCE EVALUATION ===")
# # TODO: Print performance metrics


import time
import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from traffic_environment import TrafficEnv
import rl_planners



BASE_DIR = "experiment_images"
os.makedirs(BASE_DIR, exist_ok=True)


experiments = [
    {"gamma":0.90, "theta":1e-3},
    {"gamma":0.95, "theta":1e-4},
    {"gamma":0.97, "theta":1e-5},
    {"gamma":0.99, "theta":1e-6},
    {"gamma":0.80, "theta":1e-3}
]

MAX_STEPS = 100
SAFE_LIMIT = 10


results_avg_wait = []
results_violations = []
results_safe_duration = []
results_rewards = []



for exp_id, exp in enumerate(experiments):

    gamma = exp["gamma"]
    theta = exp["theta"]

    print("\n=================================")
    print(f"Experiment {exp_id+1}")
    print(f"Gamma={gamma}  Theta={theta}")
    print("=================================")

    exp_dir = os.path.join(BASE_DIR, f"experiment_{exp_id+1}")
    os.makedirs(exp_dir, exist_ok=True)

    rewards = {"state":0}

    env = TrafficEnv(rewards=rewards, max_steps=MAX_STEPS)

    agent = rl_planners.ValueIterationPlanner(env, gamma=gamma, theta=theta)

    observation, info = env.reset(seed=42), {}

    np.random.seed(42)
    env.action_space.seed(42)

    RED, GREEN = 0, 1

    step = 0

    waiting_cars = []
    rewards_list = []
    ns_cars = []
    ew_cars = []
    total_cars = []

    violations = 0
    safe_duration = 0
    max_safe_duration = 0


    while step < MAX_STEPS:

        action = agent.choose_action(observation)

        observation, reward, terminated, truncated, info = env.step(action)

        ns, ew, light = tuple(observation)

        total = ns + ew

        waiting_cars.append(total)
        rewards_list.append(reward)

        ns_cars.append(ns)
        ew_cars.append(ew)

        total_cars.append(total)

        if total > SAFE_LIMIT:
            violations += 1
            safe_duration = 0
        else:
            safe_duration += 1
            max_safe_duration = max(max_safe_duration, safe_duration)

        light_color = "GREEN" if light == GREEN else "RED"

        print(f"Step:{step} NS:{ns} EW:{ew} Light:{light_color} Reward:{reward}")

        env.render()

        step += 1
        time.sleep(0.1)


    avg_wait = np.mean(waiting_cars)
    total_reward = np.sum(rewards_list)

    results_avg_wait.append(avg_wait)
    results_violations.append(violations)
    results_safe_duration.append(max_safe_duration)
    results_rewards.append(total_reward)

    env.render(close=True)

    steps = np.arange(len(waiting_cars))


    plt.figure()
    plt.plot(steps, waiting_cars)
    plt.title("Cars vs Step")
    plt.xlabel("Step")
    plt.ylabel("Total Cars")

    plt.savefig(os.path.join(exp_dir,"cars_vs_step.png"))
    plt.close()

    plt.figure()
    plt.plot(steps, rewards_list)

    plt.title("Reward Over Time")
    plt.xlabel("Step")
    plt.ylabel("Reward")

    plt.savefig(os.path.join(exp_dir,"reward_over_time.png"))
    plt.close()

    plt.figure()
    plt.plot(steps, ns_cars, label="NS")
    plt.plot(steps, ew_cars, label="EW")

    plt.legend()
    plt.title("Traffic NS vs EW")

    plt.savefig(os.path.join(exp_dir,"traffic_ns_vs_ew.png"))
    plt.close()

    plt.figure()
    plt.plot(steps, total_cars)

    plt.title("Traffic Over Time")
    plt.xlabel("Step")
    plt.ylabel("Cars")

    plt.savefig(os.path.join(exp_dir,"traffic_over_time.png"))
    plt.close()

labels = [f"E{i+1}" for i in range(len(experiments))]


plt.figure()
plt.bar(labels, results_avg_wait)
plt.title("Average Waiting Cars Comparison")
plt.ylabel("Average Cars")

plt.savefig(os.path.join(BASE_DIR,"comparison_avg_wait.png"))
plt.close()



plt.figure()
plt.bar(labels, results_violations)
plt.title("Traffic Violations Comparison")
plt.ylabel("Violations")

plt.savefig(os.path.join(BASE_DIR,"comparison_violations.png"))
plt.close()



plt.figure()
plt.bar(labels, results_safe_duration)
plt.title("Maximum Safe Duration Comparison")
plt.ylabel("Duration")

plt.savefig(os.path.join(BASE_DIR,"comparison_safe_duration.png"))
plt.close()



plt.figure()
plt.bar(labels, results_rewards)
plt.title("Total Reward Comparison")
plt.ylabel("Reward")

plt.savefig(os.path.join(BASE_DIR,"comparison_rewards.png"))
plt.close()


print("\n=========== FINAL RESULTS ===========")

for i in range(len(experiments)):

    print(f"\nExperiment {i+1}")
    print("Gamma:",experiments[i]["gamma"])
    print("Theta:",experiments[i]["theta"])
    print("Average Waiting Cars:",results_avg_wait[i])
    print("Traffic Violations:",results_violations[i])
    print("Max Safe Duration:",results_safe_duration[i])
    print("Total Reward:",results_rewards[i])


print("\nAll graphs saved inside:", BASE_DIR)