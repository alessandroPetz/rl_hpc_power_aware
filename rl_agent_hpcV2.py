from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch as th
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

from renewable_models import RenewableModels

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_powercap")
VEC_PATH = os.path.join(MODEL_DIR, "vecnormalize.pkl")


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
        self.curtailment_history= []
        ## LOG ##


        self.df = df.reset_index(drop=True)
        self.N = len(self.df)

        self.threshold = threshold
        self.capacity = battery_capacity
        
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate

        self.prev_action = 0

        # -------------------------------
        # Observation: [P_ratio, P_peak, battery_norm, time_left, price_base_norm, hour_sin, hour_cos, prev_a, P_ren_norm, forecast_ren_norm1h, forecast_ren_norm6h]
        # -------------------------------
        self.observation_space = spaces.Box(
            low=np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32),
            high=np.array([3., 3., 1., 1., 1., 1., 1., 1., 1., 1., 0.1], dtype=np.float32)
        )

        # -------------------------------
        # Action = singola variabile [-1, 1]
        # a < 0 → scarica
        # a > 0 → carica
        # -------------------------------
        # # action discreta
        # self.action_space = spaces.Box(
        #     low=np.array([-1.], dtype=np.float32),
        #     high=np.array([1.], dtype=np.float32)
        # )
        # action discreta
        self.action_levels = np.array([-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0], dtype=np.float32)
        #self.action_levels = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action_levels))
 

        self.reset()

    def _get_obs(self):
        t = min(self.t, self.N - 1)
        P = float(self.df.loc[t, "power"])          # W
        P_ratio = np.clip(P / self.threshold, 0, 3)
        P_peak  = np.clip(max(P - self.threshold, 0) / self.threshold, 0, 3)
        battery_norm = float(self.battery / self.capacity)   # 0..1

        ts = self.df.loc[t, "time"]
        hour = ts.hour + ts.minute / 60.0
        hour_sin = np.sin(2 * np.pi * hour / 24.)
        hour_cos = np.cos(2 * np.pi * hour / 24.)

        price_base = float(self.df.loc[t, "price_base"])
        price_base_norm = price_base / (self.df["price_base"].max() + 1e-9)

        time_left = 1.0 - (t / (self.N - 1))
        prev_a = getattr(self, "prev_action", 0.0)

        P_ren = float(self.df.loc[t, "P_ren"])
        P_ren_norm = P_ren / (self.threshold + 1e-9)   # può superare 1

        E_forecast_1h = float(self.df.loc[t, "forecast_P_ren_1h"])  # qui è energia Wh (sum)
        E_forecast_6h = float(self.df.loc[t, "forecast_P_ren_6h"])

        # normalizzo forecast su capacità (Wh)
        E_forecast_1h_norm = E_forecast_1h / (self.capacity + 1e-9)
        E_forecast_6h_norm = E_forecast_6h / (6.0 * self.capacity + 1e-9)  # opzionale scaling

        obs = np.array([
            P_ratio, P_peak, battery_norm, time_left,
            price_base_norm, hour_sin, hour_cos, prev_a,
            P_ren_norm, E_forecast_1h_norm, E_forecast_6h_norm
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        ## START PLOT LOG ##
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

            print(f"[Episode {self.episode_idx}] total cost = {sum(self.cost_history) :.4f}")


        self.episode_idx += 1
        self.battery_history = [self.capacity / 2]
        self.time_history =  [self.df.loc[0, "time"]]
        self.cost_history = [0]
        self.curtailment_history= [0]
        ## END PLOT LOG ##

        self.t = 0
        self.battery = self.capacity / 2  # inizialmente piena
        return self._get_obs(), {}

    def step(self, action):

        # if self.t >= self.N - 1:
        #     return self._get_obs(), 0.0, True, False, {}
        
        # # action continua
        # a = float(action[0])   # a > 0 → carica, a < 0 → scarica
        # action discreta
        a = float(self.action_levels[action])
        self.prev_action = a 

        t = self.t
        P_load = float(self.df.loc[t, "power"])     # carico HPC
        P_ren  = float(self.df.loc[t, "P_ren"])     # rinnovabile

        dt = float(self.df.loc[t, "dt_hours"])

        if dt == 0:
            self.t += 1
            terminated = self.t > self.N - 1
            return self._get_obs(), 0.0, terminated, False, {}

        # -----------------------------------
        # 1. ENERGIA AZIONE (libera)
        # -----------------------------------
        
        # ------START MANIPULATING ACTION ----
        # ------------------------------------

        # 1) come in simulazione senza batteria
        # a=0
        #
        # 2) come in simulaizone con batteria, 
        # la differenza è che qui carico o scarico sempre al massimo, in simulazione sono limitato dal threshold
        #
        # if P <= self.threshold :
        #     # carico la batteria
        #     a = 1
        
        # else:
        #     # uso  la batteria
        #     a = -1
        #
        #
        # 3) innesto manualmente regola per comportarmi più similmente alla sim deterministica con batteria
        # se sono sotto threshold non posso usare la batteria
        # se sono sopra threshold non posso caricarla.
        # difff con 2) qui posso avere a in percentuale
        # 
        # if P <= self.threshold :
        #     a = max (a,0)
        # else:
        #     a = min (a,0)
        
        # 3) forzare PRIORITÀ ALLA RINNOVABILE
        # if P_ren > P_load and self.battery < self.capacity:
        #     a = max(a, 0)   # forzo carica


        # ----------------------------------- 
        # ----- END MANIPULATING ACTION -----


    # azione -> energia richiesta (Wh) per questo step
        if a > 0:
            # richiesta energia per caricare (da SURPLUS solo)
            P_charge_req = a * self.max_charge_rate   # W
            # massimo power che può provenire da surplus rinnovabile:
            P_surplus_available = max(P_ren - P_load, 0.0)
            # limita la potenza di carica al surplus (policy A)
            P_charge_eff = min(P_charge_req, P_surplus_available)
            E_charge = P_charge_eff * dt
            E_discharge = 0.0
        else:
            # scarico
            P_discharge_req = -a * self.max_discharge_rate
            # energia disponibile dalla batteria (Wh)
            E_discharge_req = P_discharge_req * dt
            E_discharge = min(E_discharge_req, self.battery)
            E_charge = 0.0

        # Potenze corrispondenti
        P_charge = E_charge / dt
        P_discharge = E_discharge / dt

        # Potenza vista dalla rete (prima: load - ren)
        P_net = P_load - P_ren
        # effetto batteria: carica aumenta richiesta, scarica riduce richiesta
        P_grid = P_net + P_charge - P_discharge
        P_grid = max(P_grid, 0.0)

        # energy from grid this step
        E_grid = P_grid * dt

        # split base/peak on grid import
        E_base = min(P_grid, self.threshold) * dt
        E_peak = max(P_grid - self.threshold, 0) * dt

        price_base = float(self.df.loc[t, "price_base"])
        price_high = float(self.df.loc[t, "price_high"])
        cost = E_base * price_base + E_peak * price_high

        # update battery
        self.battery += E_charge
        self.battery -= E_discharge
        self.battery = float(np.clip(self.battery, 0.0, self.capacity))

        # curtailment (energia rinnovabile sprecata)
        P_surplus_after_charge = max(P_ren - P_load - P_charge, 0.0)
        E_curtail = P_surplus_after_charge * dt

        # reward: vogliamo MAXIMIZE -> reward = -cost + shaping
        reward = -cost
        # shaping: penalizza fortemente spreco; incentiva uso rinnovabile
        reward -= 5.0 * E_curtail
        E_ren_used = min(P_ren, P_load + P_charge) * dt
        reward += 2.0 * E_ren_used

        # aggiorna log
        self.battery_history.append(self.battery)
        self.cost_history.append(cost)
        self.curtailment_history.append(E_curtail)
        self.time_history.append(self.df.loc[t, "time"])

        self.t += 1
        terminated = self.t > self.N - 1

        return self._get_obs(), float(reward), terminated, False, {}                 



def dynamic_low_price(timestamp):
    hour = timestamp.hour
    if 0 <= hour < 6:
        return 0.0005
    elif 6 <= hour < 22:
        return 0.0012
    else:
        return 0.0007
        
if __name__ == "__main__":
    # -------------------------------------------------------
    # RUN TRAINING
    # -------------------------------------------------------
    df = pd.read_csv("cluster_power_only_nodes_10days.csv")[8641:]
    # df = pd.read_csv("cluster_power_only_nodes.csv")

    df["time"] = pd.to_datetime(df["time"])
    df["dt_hours"] = df["time"].diff().dt.total_seconds() / 3600
    df["dt_hours"] = df["dt_hours"].fillna(0)
    
    df["price_base"] = df["time"].apply(dynamic_low_price)
    df["price_high"] = df["price_base"] * 3

    rm = RenewableModels(seed=42)

    df["P_solar"] = rm.solar_cloudy(df) 
    df["P_wind"]  = rm.wind_stochastic_daily(df)
    df["P_ren"]   = df["P_solar"] + df["P_wind"]

    # previsioni del tempo
    steps_per_hour = int(3600 / 20)
    df["forecast_P_ren_1h"] = (
        df["P_ren"]
        .rolling(window=steps_per_hour, min_periods=1)
        .sum()
        .shift(-steps_per_hour)
        .fillna(0)
    )
    df["forecast_P_ren_6h"] = (
        df["P_ren"]
        .rolling(window=steps_per_hour*6, min_periods=1)
        .sum()
        .shift(-steps_per_hour)
        .fillna(0)
    )  


    vec_env = DummyVecEnv([lambda: HPCBatteryEnv(df)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_reward=10.0)


    policy_kwargs = dict(
        net_arch=[256, 256, 128],
        activation_fn=th.nn.ReLU
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=16384,          # raccolgo questa quantità di step prima di fare un update
        #n_steps=4096,
        batch_size=256,
        #batch_size=64,
        ent_coef=0.03,    # aumentare per esplorare di più (se si blocca in locale minimo)
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=0,
        device="cpu"
    )
    model.learn(total_timesteps=10_000_000)

    # Save final model trained
    model.save(MODEL_PATH)
    vec_env.save(VEC_PATH)