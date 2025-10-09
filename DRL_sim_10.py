import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

# ------------------ Custom ORAN Traffic Environment ------------------
class ORANTrafficEnv(gym.Env):
    def __init__(self):
        super(ORANTrafficEnv, self).__init__()
        print("[Init] Setting up ORAN environment.")

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        self.max_steps = 10
        self.current_step = 0
        self.state = None
        self.reset()

    def normalize(self, packet_size, freq, protocol, bandwidth):
        packet_norm = (packet_size - 256) / (1024 - 256)
        freq_norm = (freq - 2.5) / (4.0 - 2.5)
        protocol_norm = protocol / 2
        bandwidth_norm = (bandwidth - 10) / (50 - 10)
        return np.array([packet_norm, freq_norm, protocol_norm, bandwidth_norm], dtype=np.float32)

    def step(self, action):
        raw_packet, raw_freq, raw_protocol, raw_bandwidth = self.raw_state
        path = action + 1

        # Latency calculations
        latency_ru = 5 + random.uniform(0, 0.2)
        latency_du = 10 + 0.02 * raw_packet - 0.6 * raw_freq + random.uniform(0, 0.2)
        latency_cu = 15 + raw_protocol * 3 - 0.3 * raw_bandwidth + random.uniform(0, 0.2)

        # STRONG path penalty to separate latency better
        if path == 1:
            path_penalty = 0
        elif path == 2:
            path_penalty = 10
        else:
            path_penalty = 25

        total_latency = latency_ru + latency_du + latency_cu + path_penalty

        # Exponential reward scaling
        reward = np.exp(-total_latency / 50.0)

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        info = {"latency": total_latency, "path": path}

        self.raw_state = self._sample_raw_state()
        self.state = self.normalize(*self.raw_state)

        return self.state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.raw_state = self._sample_raw_state()
        self.state = self.normalize(*self.raw_state)
        return self.state, {}

    def _sample_raw_state(self):
        packet_size = random.choice([256, 512, 1024])
        freq = random.uniform(2.5, 4.0)
        protocol = random.choice([0, 1, 2])
        bandwidth = random.choice([10, 20, 30, 40, 50])
        return (packet_size, freq, protocol, bandwidth)

    def render(self, mode='human'):
        pass

# ------------------ Initialize and Check Environment ------------------
env = ORANTrafficEnv()
check_env(env)

# ------------------ Baseline Policy ------------------
def baseline_policy(raw_state):
    packet_size, freq, protocol, bandwidth = raw_state
    if bandwidth >= 30 and protocol == 0:
        return 0
    elif freq >= 3.5:
        return 1
    else:
        return 2

baseline_latencies = []
for i in range(500):
    obs, _ = env.reset(seed=i)
    episode_latency = 0
    for _ in range(env.max_steps):
        action = baseline_policy(env.raw_state)
        obs, _, terminated, _, info = env.step(action)
        episode_latency += info['latency']
        if terminated:
            break
    baseline_latencies.append(episode_latency / env.max_steps)

# ------------------ Train DRL Model ------------------
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    n_steps=1024,
    batch_size=128,
    gae_lambda=0.92,
    gamma=0.995,
    ent_coef=0.005,
    vf_coef=0.5,
    max_grad_norm=0.5,
    n_epochs=10,
)

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./models/",
    name_prefix="oran_ppo"
)

model.learn(total_timesteps=700000, callback=checkpoint_callback)
model.save("oran_drl_model")

# ------------------ Evaluate DRL Model ------------------
model = PPO.load("oran_drl_model")
drl_latencies = []

for i in range(500):
    obs, _ = env.reset(seed=i)
    episode_latency = 0
    for _ in range(env.max_steps):
        action, _ = model.predict(obs)
        obs, _, terminated, _, info = env.step(action)
        episode_latency += info['latency']
        if terminated:
            break
    drl_latencies.append(episode_latency / env.max_steps)

# ------------------ Calculate and Display Improvement ------------------
avg_baseline = np.mean(baseline_latencies)
avg_drl = np.mean(drl_latencies)
improvement = ((avg_baseline - avg_drl) / avg_baseline) * 100

print(f"\nAverage Baseline Latency: {avg_baseline:.2f} ms")
print(f"Average DRL Latency: {avg_drl:.2f} ms")
print(f"Latency Improvement: {improvement:.2f}%")

# ------------------ Plot Results ------------------
plt.figure(figsize=(12, 6))
plt.plot(baseline_latencies, label='Baseline Policy (No DRL)', color='red', linestyle='--')
plt.plot(drl_latencies, label='DRL (PPO)', color='green')
plt.title("Latency Comparison: Baseline vs DRL Optimized Routing")
plt.xlabel("Simulation Instance")
plt.ylabel("Avg Latency per Episode (ms)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
