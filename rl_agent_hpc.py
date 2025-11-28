from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch as th
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class HPCBatteryEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self, 
        df, 
        threshold=400000,
        battery_capacity=400000,
        max_charge_rate=400000,      # Wh/h
        max_discharge_rate=400000

    ):
        super().__init__()

        ## LOG ##
        self.episode_idx = -1
        self.battery_history = []
        self.time_history = []
        self.cost_history = []
        ## LOG ##


        self.df = df.reset_index(drop=True)
        self.N = len(self.df)

        self.threshold = threshold
        self.capacity = battery_capacity
        
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate

        # -------------------------------
        # Observation: [P_ratio, P_peak, battery_norm, time_norm, price_base_norm]
        # -------------------------------
        self.observation_space = spaces.Box(
            low=np.array([0., 0., 0., 0., 0.], dtype=np.float32),
            high=np.array([3., 3., 1., 1., 1.], dtype=np.float32)
        )

        # -------------------------------
        # Action = singola variabile [-1, 1]
        # a < 0 → scarica
        # a > 0 → carica
        # -------------------------------
        self.action_space = spaces.Box(
            low=np.array([-1.], dtype=np.float32),
            high=np.array([1.], dtype=np.float32)
        )
 

        self.reset()

    def _get_obs(self):
        t = min(self.t, self.N - 1)
        P = float(self.df.loc[t, "power"])

        P_ratio = np.clip(P / self.threshold, 0, 3)
        
        P_peak  = np.clip(max(P - self.threshold, 0) / self.threshold, 0, 3)
        battery_norm = self.battery / self.capacity
        time_norm = t / (self.N - 1)

        # prezzi normalizzati
        price_base = float(self.df.loc[t, "price_base"])
        price_base_norm = price_base / self.df["price_base"].max()

        # print(np.array([
        #     P_ratio,
        #     P_peak,
        #     battery_norm,
        #     time_norm
        # ], dtype=np.float32))

        return np.array([
            P_ratio,
            P_peak,
            battery_norm,
            time_norm,
            price_base_norm
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        ## START PLOT LOG ##
        #print(self.battery_history)
        #print(self.cost_history)
        if self.episode_idx > 0:
            plt.figure(figsize=(14, 5))
            plt.plot(mdates.date2num(self.time_history), self.battery_history)
            plt.xlabel("Time")
            plt.ylabel("Battery Charge (Wh)")
            plt.title("Battery State of Charge Over Time")
            plt.grid(True)
            plt.tight_layout()
            filename = f"plots/battery_plot_ep{self.episode_idx}.png"
            plt.savefig(filename)
            plt.close()   


        self.episode_idx += 1
        self.battery_history = [self.capacity]
        self.time_history =  [self.df.loc[0, "time"]]
        self.cost_history = [0]
        ## END START PLOT LOG ##

        self.t = 0
        self.battery = self.capacity  # inizialmente piena
        return self._get_obs(), {}

    def step(self, action):

        if self.t >= self.N - 1:
            return self._get_obs(), 0.0, True, False, {}
        
        
        a = float(action[0])   # a > 0 → carica, a < 0 → scarica

        t = self.t
        P = float(self.df.loc[t, "power"])
        dt = float(self.df.loc[t, "dt_hours"])

        if dt == 0:
            self.t += 1
            terminated = self.t >= self.N - 1
            return self._get_obs(), 0.0, terminated, False, {}

        # -----------------------------------
        # 1. ENERGIA AZIONE (libera)
        # -----------------------------------
        # NB se fisso a = 0, la simulazione diventa diventa come quella deterministica senza batteria.
        # a=0
        if a > 0:
            # CARICA
            E_charge_req = a * self.max_charge_rate * dt
            E_charge = min(E_charge_req, self.capacity - self.battery)
            E_discharge = 0.0
        else:
            # SCARICA
            E_discharge_req = -a * self.max_discharge_rate * dt
            E_discharge = min(E_discharge_req, self.battery)
            E_charge = 0.0

        # -----------------------------------
        # 2. POTENZA VISTA DALLA RETE (libera)
        # -----------------------------------
        # P_charge = potenza assorbita per caricare la batteria
        # P_discharge = potenza "negativa" vista dalla rete
        P_charge     = E_charge     / dt
        P_discharge  = E_discharge  / dt

        # Potenza effettiva che si chiede alla rete
        P_grid = P + P_charge - P_discharge
        P_grid = max(P_grid, 0)  # Non può essere negativa

        # -----------------------------------
        # 3. ENERGIA DA RETE IN QUESTO STEP
        # -----------------------------------
        E_grid = P_grid * dt

        # Divisa in:
        E_base  = min(P_grid, self.threshold) * dt
        E_peak  = max(P_grid - self.threshold, 0) * dt

        # -----------------------------------
        # 4. COSTO (modello generale)
        # -----------------------------------
        price_base  = self.df.loc[t, "price_base"]
        price_high = self.df.loc[t, "price_high"]

        cost = (E_base * price_base) + (E_peak * price_high)

        # -----------------------------------
        # 5. AGGIORNA BATTERIA
        # -----------------------------------
        self.battery += E_charge
        self.battery -= E_discharge
        self.battery = np.clip(self.battery, 0, self.capacity)

        # -----------------------------------
        # 6. REWARD
        # -----------------------------------
        # reward = -cost * 1000

        reward = 0.0

        # 1) Reward base: costo negativo
        reward -= cost * 1000

        # # 2) Penalità per caricare quando il prezzo è alto
        # if a > 0:   # sta caricando
        #     reward -= price_base * E_charge * 0.5

        # # 3) Bonus per scaricare quando il prezzo è alto (arbitraggio)
        # price_mean = self.df["price_base"].mean()
        # if a < 0:   # sta scaricando
        #     reward += max(price_base - price_mean, 0) * E_discharge * 2.0

        # # 4) Bonus finale per batteria piena
        # if self.t == self.N - 1:   # episodio finito
        #     reward += (self.battery / self.capacity) * 50.0

        # -----------------------------------
        # 7. TEMPO
        # -----------------------------------
        self.t += 1
        terminated = self.t >= self.N - 1

        ## LOG ##
        self.battery_history.append(self.battery)
        self.cost_history.append(cost)
        self.time_history.append(self.df.loc[t, "time"])
        ## LOG ##


        return self._get_obs(), reward, terminated, False, {}                  


# -------------------------------------------------------
# CALLBACK PER COSTO EPISODIO
# -------------------------------------------------------
class EpisodeCostCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_cost = 0
        self.episode_idx = 1

    def _on_step(self):
        reward = self.locals["rewards"][0]
        self.episode_cost += -reward 

        done = self.locals["dones"][0]
        if done:
            self.episode_cost = self.episode_cost / 10000  # nomralizzo
            print(f"[Episode {self.episode_idx}] total cost = {self.episode_cost:.4f}")
            self.episode_idx += 1
            self.episode_cost = 0
        return True

def dynamic_low_price(timestamp):
    hour = timestamp.hour
    if 0 <= hour < 6:
        return 5
    elif 6 <= hour < 22:
        return 12
    else:
        return 7
        
if __name__ == "__main__":
    # -------------------------------------------------------
    # RUN TRAINING
    # -------------------------------------------------------
    df = pd.read_csv("cluster_power_only_nodes_10days.csv")

    df["time"] = pd.to_datetime(df["time"])
    df["dt_hours"] = df["time"].diff().dt.total_seconds() / 3600
    df["dt_hours"] = df["dt_hours"].fillna(0)
    
    df["price_base"] = df["time"].apply(dynamic_low_price)
    df["price_high"] = df["price_base"] * 3  


    env = HPCBatteryEnv(df)
    callback = EpisodeCostCallback()

    policy_kwargs = dict(
        net_arch=[256, 256, 128],
        activation_fn=th.nn.ReLU
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=4096,          # raccolgo questa quantità di step prima di fare un update
        batch_size=64,
        ent_coef=0.01,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=0,
        device="cpu"
    )
    model.learn(total_timesteps=1_000_000, callback=callback)

    model.save("ppo_battery_hpc")