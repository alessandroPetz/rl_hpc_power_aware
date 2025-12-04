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
        ## LOG ##


        self.df = df.reset_index(drop=True)
        self.N = len(self.df)

        self.threshold = threshold
        self.capacity = battery_capacity
        
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate

        self.prev_action = 0

        # -------------------------------
        # Observation: [P_ratio, P_peak, battery_norm, time_norm, price_base_norm]
        #              [P_ratio, P_peak, battery_norm, time_left, price_base_norm, hour_sin, hour_cos, prev_a]
        # -------------------------------
        self.observation_space = spaces.Box(
            low=np.array([0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32),
            high=np.array([3., 3., 1., 1., 1., 1., 1., 1.], dtype=np.float32)
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


        # TODO dare informazione di tutta la fascia oraria e prezzo

        t = min(self.t, self.N - 1)
        P = float(self.df.loc[t, "power"])
        # existing
        P_ratio = np.clip(P / self.threshold, 0, 3)
        P_peak  = np.clip(max(P - self.threshold, 0) / self.threshold, 0, 3)
        battery_norm = self.battery / self.capacity
        # time cyclical
        ts = self.df.loc[t, "time"]
        hour = ts.hour + ts.minute/60.0
        hour_sin = np.sin(2 * np.pi * hour / 24.)
        hour_cos = np.cos(2 * np.pi * hour / 24.)
        # price normalized
        price_base = float(self.df.loc[t, "price_base"])
        price_base_norm = price_base / self.df["price_base"].max()
        # remaining proportion of episode
        time_left = 1.0 - (t / (self.N - 1))
        # previous action (keep as scalar normalized -1..1)
        prev_a = getattr(self, "prev_action", 0.0)
        return np.array([P_ratio, P_peak, battery_norm, time_left, price_base_norm, hour_sin, hour_cos, prev_a], dtype=np.float32)

    # def _get_obs(self):
    #     t = min(self.t, self.N - 1)
    #     P = float(self.df.loc[t, "power"])

    #     P_ratio = np.clip(P / self.threshold, 0, 3)
        
    #     P_peak  = np.clip(max(P - self.threshold, 0) / self.threshold, 0, 3)
    #     battery_norm = self.battery / self.capacity
    #     time_norm = t / (self.N - 1)

    #     # prezzi normalizzati
    #     price_base = float(self.df.loc[t, "price_base"])
    #     price_base_norm = price_base / self.df["price_base"].max()


        # return np.array([
        #     P_ratio,
        #     P_peak,
        #     battery_norm,
        #     time_norm,
        #     price_base_norm
        # ], dtype=np.float32)

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
        P = float(self.df.loc[t, "power"])
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
        
        # ----------------------------------- 
        # ----- END MANIPULATING ACTION -----



        if a > 0:
            # CARICO LA BATTERIA
            E_charge_req = a * self.max_charge_rate * dt
            E_charge = min(E_charge_req, self.capacity - self.battery)
            #E_charge = min(E_charge_req, self.capacity - self.battery, (self.threshold - P) * dt)   # LIMITA la batteria a non caricarsi con costo HIGH
                                                                                                    # aggiunta al 2, comportamente come sim con batteria 
            E_discharge = 0.0
        else:
            # USO LA BATTERIA
            E_discharge_req = -a * self.max_discharge_rate * dt
            E_discharge = min(E_discharge_req, self.battery)
            # E_discharge = min(E_discharge_req, self.battery, max(P - self.threshold, 0) * dt)   # LIMITA la batteria a coprire al massimo il picco
                                                                                                # aggiunta al 2, comportamente come sim con batteria  
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
        #print(cost)

        # -----------------------------------
        # 5. AGGIORNA BATTERIA
        # -----------------------------------
        self.battery += E_charge
        self.battery -= E_discharge
        self.battery = np.clip(self.battery, 0, self.capacity)

        # -----------------------------------
        # 6. REWARD
        # -----------------------------------

        # 1) Reward base: costo negativo
        reward = cost


        # 2) Penalità per caricare quando il prezzo è alto
        # if a > 0:   # sta caricando
        #     reward -= price_base * E_charge * 0.5

        # 3) Bonus per scaricare quando il prezzo è alto (arbitraggio)
        # price_mean = self.df["price_base"].mean()
        # if a < 0:   # sta scaricando
        #     reward += max(price_base - price_mean, 0) * E_discharge * 2.0

        # --- MALUS basato sulla logica richiesta ---

        # # caso 1) SCARICA senza picco (P <= threshold)
        # if a < 0 and P <= self.threshold:
        #     # malus proporzionale al prezzo base
        #     reward -= E_discharge * price_base

        # # caso 2) CARICA durante il picco (P > threshold)
        # if a > 0 and P > self.threshold:
        #     # malus proporzionale al prezzo alto
        #     reward -= E_charge * price_high

        # if P <= self.threshold:
        #     energy_used = E_charge + E_discharge
        #     reward -= energy_used * price_base  # scoraggia attività inutile

        # # 4) Bonus finale per batteria piena
        # if self.t == self.N - 1:   # episodio finito
        #     reward += (self.battery / self.capacity) * 50.0

        # # costo se uso la batteria, piutoto che non usarla.
        # cost_scale = 10_000   # ✅ valore corretto per il tuo problema
        # action_cost = 0.01 * (abs(a) * self.max_charge_rate * dt) / cost_scale
        # reward -= action_cost

        # # penalty per picco istantaneo
        # peak_penalty = 10.0 * max(P_grid - self.threshold, 0) * dt / cost_scale
        # reward -= peak_penalty


        # # 4) costo totale a fine episodio
        if self.t == self.N - 1:   # episodio finito
            reward -= sum(self.cost_history)

        # -----------------------------------
        # 7. TEMPO
        # -----------------------------------s<
        self.t += 1
        terminated = self.t > self.N - 1

        ## LOG ##
        self.battery_history.append(self.battery)
        self.cost_history.append(cost)
        self.time_history.append(self.df.loc[t, "time"])
        ## LOG ##


        return self._get_obs(), reward, terminated, False, {}                  



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


    vec_env = DummyVecEnv([lambda: HPCBatteryEnv(df)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_reward=10.0)
    # if os.path.exists(VEC_PATH):
    #     print("[Main] Carico VecNormalize esistente...")
    #     env = VecNormalize.load(VEC_PATH, env)
    # else:
    #     print("[Main] Crete new VecNormalize...")
    #     env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)

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