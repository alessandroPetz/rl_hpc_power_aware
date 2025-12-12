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
import shutil

from utils.renewable_real import RenewableModels
from utils.co2 import CarbonIntensityModels

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_powercap")
VEC_PATH = os.path.join(MODEL_DIR, "vecnormalize.pkl")


class HPCBatteryEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self, 
        df, 
        threshold=400000,
        battery_capacity=3200000,
        max_charge_rate=3200000,      # Wh/h
        max_discharge_rate=3200000
        # fattore medio (gCO2 per kWh) — scegli il valore adatto (es. 270 gCO2/kWh per ITA come esempio)
        # CO2_G_PER_KWH_STATIC = 270.0

    ):
        super().__init__()

        ## LOG ##
        self.episode_idx = -1
        self.battery_history = []
        self.time_history = []
        self.cost_history = []
        self.co2_history = []
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
        # Observation: [P_ratio, P_peak, battery_norm, time_left, price_base_norm, co2_intensity_norm
        # hour_sin, hour_cos, prev_a, P_ren_norm, forecast_ren_norm1h, forecast_ren_norm6h, 
        # co2_forecast_1h_norm, co2_forecast_6h_norm]
        # -------------------------------
        self.observation_space = spaces.Box(
            low=np.array([0., 0., 0., 0., 0., 0., 0. ,0., 0., 0., 0., 0., 0., 0. ], dtype=np.float32),
            high=np.array([3., 3., 1., 1., 1., 1., 1. ,1., 1., 1., 1., 1., 1., 1.], dtype=np.float32)
        )

        # -------------------------------
        # Action = singola variabile [-1, 1]
        # a < 0 → scarica
        # a > 0 → carica
        # -------------------------------
        

        # ACTION DISCRETA
        # self.action_levels = np.array([-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0], dtype=np.float32)
        # self.action_space = spaces.Discrete(len(self.action_levels))
        
        # ACTION CONTINUA
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
 

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
        
        co2_int = float(self.df.loc[t, "co2_intensity"])
        co2_int_norm = price_base / (self.df["co2_intensity"].max() + 1e-9)

        time_left = 1.0 - (t / (self.N - 1))
        prev_a = getattr(self, "prev_action", 0.0)

        P_ren = float(self.df.loc[t, "P_ren"])
        P_ren_norm = P_ren / (self.threshold + 1e-9)   # può superare 1

        E_forecast_1h = float(self.df.loc[t, "forecast_P_ren_1h"])  # qui è energia Wh (sum)
        E_forecast_6h = float(self.df.loc[t, "forecast_P_ren_6h"])

        # normalizzo forecast su capacità (Wh)
        E_forecast_1h_norm = E_forecast_1h / (self.capacity + 1e-9)
        E_forecast_6h_norm = E_forecast_6h / (6.0 * self.capacity + 1e-9)  # opzionale scaling

        co2_forecast_1h = float(self.df.loc[t, "forecast_P_ren_1h"])  # qui è energia Wh (sum)
        co2_forecast_6h = float(self.df.loc[t, "forecast_P_ren_6h"])

        # normalizzo minmax
        co2_forecast_1h_norm = (
            co2_forecast_1h - self.df["co2_intensity"].min()
        ) / (self.df["co2_intensity"].max() - self.df["co2_intensity"].min() + 1e-9)

        co2_forecast_6h_norm = (
            co2_forecast_6h - self.df["co2_intensity"].min()
        ) / (self.df["co2_intensity"].max() - self.df["co2_intensity"].min() + 1e-9)

        obs = np.array([
            P_ratio, P_peak, battery_norm, time_left,
            price_base_norm, co2_int_norm, hour_sin, hour_cos, prev_a,
            P_ren_norm, E_forecast_1h_norm, E_forecast_6h_norm, co2_forecast_1h_norm, co2_forecast_6h_norm
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

            #print(f"[Episode {self.episode_idx}] total cost = {sum(self.cost_history) :.4f}")
            #print(f"[Episode {self.episode_idx}] CO2 totale (kg): {sum(self.co2_history)/1000:.3f}")

            csv_path = os.path.join(".", "results.csv")

            # Aggiunge la riga
            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(f"{self.episode_idx};{sum(self.cost_history):.4f};{sum(self.co2_history)/1000:.3f}\n")

            


        self.episode_idx += 1
        self.battery_history = [self.capacity / 2]
        self.time_history =  [self.df.loc[0, "time"]]
        self.cost_history = [0]
        self.co2_history = [0]
        self.curtailment_history= [0]
        ## END PLOT LOG ##

        self.t = 0
        self.battery = self.capacity / 2  # inizialmente piena
        return self._get_obs(), {}


    def step(self, action):

        # --- ACTION ---
        # a in [-1,1], a<0 scarica; a=0 neutro; a>0 carica da rete
        # action DISCRETA
        #  a = float(self.action_levels[action])
        # action CONTINUA
        a = float(action[0])

        self.prev_action = a

        t = self.t
        dt = float(self.df.loc[t, "dt_hours"])
        time = self.df.loc[t, "time"]

        P_load = float(self.df.loc[t, "power"])      # W
        P_ren  = float(self.df.loc[t, "P_ren"])      # W

        price_base = float(self.df.loc[t, "price_base"])
        price_high = float(self.df.loc[t, "price_high"])

        CO2_G_PER_KWH = float(self.df.loc[t, "co2_intensity"])

        # se dt=0: no avanzamento fisico
        if dt == 0:
            self.t += 1
            terminated = self.t > self.N - 1
            return self._get_obs(), 0.0, terminated, False, {}

        # ---------------------------------------------------------
        # 1. USA LA RINNOVABILE PER COPRIRE IL CARICO
        # ---------------------------------------------------------
        P_from_ren = min(P_load, P_ren)
        P_surplus  = max(P_ren - P_from_ren, 0.0)

        # ---------------------------------------------------------
        # 1A. SE C'È SURPLUS: CARICA LA BATTERIA E FINE STEP
        # ---------------------------------------------------------
        if P_surplus > 0:
            E_surplus = P_surplus * dt                     # Wh
            max_charge_wh = self.max_charge_rate * dt      # Wh
            room = self.capacity - self.battery            # Wh

            E_charge = min(E_surplus, max_charge_wh, room)
            self.battery += E_charge

            E_curtail = E_surplus - E_charge

            # reward: gratis → positivo se usi rinnovabile
            reward = 0

            # logs
            self.battery_history.append(self.battery)
            self.cost_history.append(0.0)
            self.co2_history.append(0.0)
            self.curtailment_history.append(E_curtail)
            self.time_history.append(time)

            self.t += 1
            terminated = self.t > self.N - 1
            return self._get_obs(), float(reward), terminated, False, {}

        # ---------------------------------------------------------
        # 2. NON C'È SURPLUS → RESTO DA COPRIRE
        # ---------------------------------------------------------
        P_remaining = P_load - P_from_ren       # W
        E_needed = P_remaining * dt             # Wh

        E_curtail = 0.0
        E_charge = 0.0
        E_discharge = 0.0
        
        #a=-1 if unconmmented = sim standard
        
        # ---------------------------------------------------------
        # 2A. SE a < 0 → USA LA BATTERIA
        # ---------------------------------------------------------
        if a < 0:
            max_discharge_wh = -a* self.max_discharge_rate * dt
            E_discharge = min(E_needed, self.battery, max_discharge_wh)
            self.battery -= E_discharge
            E_needed -= E_discharge

        # ---------------------------------------------------------
        # 2B. SE a > 0 → CARICA DA RETE
        # ---------------------------------------------------------
        if a > 0:
            max_charge_wh = a* self.max_charge_rate * dt
            room = self.capacity - self.battery
            E_charge = min(max_charge_wh, room)
            self.battery += E_charge

            # caricare da rete aumenta la richiesta
            E_needed += E_charge

        # ---------------------------------------------------------
        # 3. ENERGIA FINALE DALLA RETE (Wh)
        # ---------------------------------------------------------
        P_grid = E_needed / dt if dt > 0 else 0

        E_base = min(P_grid, self.threshold) * dt
        E_peak = max(P_grid - self.threshold, 0) * dt

        cost = E_base * price_base + E_peak * price_high
        

        # CALCOLO co2
        co2_g = (E_base + E_peak) / 1000 * CO2_G_PER_KWH


        # ---------------------------------------------------------
        # 4. REWARD (stesso stile del tuo)
        # --------------------------------------------------------
        reward = 0
        # costo
        reward -= co2_g

        # reward -= cost 
        # Ridurre il picco energetico
        # reward -= 4.0 * (E_peak )
        # Favorire arbitraggio prezzo (carica quando costa poco, scarica quando costa molto)
        # if a > 0:
        #     reward -= (price_base * E_charge)
        # if a < 0:
        #     reward += (price_high * E_discharge)

        # ---------------------------------------------------------
        # LOGS
        # ---------------------------------------------------------
        self.battery_history.append(self.battery)
        self.cost_history.append(cost)
        self.co2_history.append(co2_g)
        self.curtailment_history.append(E_curtail)
        self.time_history.append(time)

        self.t += 1
        terminated = self.t > self.N - 1

        # shape reward
        if terminated:
            reward -= sum(self.cost_history)
            # reward -= sum(self.co2_history)

        return self._get_obs(), float(reward), terminated, False, {}



def dynamic_low_price(timestamp):
    hour = timestamp.hour
    if 0 <= hour < 6:
        return 0.0002
    elif 6 <= hour < 22:
        return 0.0012
    else:
        return 0.0005

def pulizia_progetto(base_path="."):
    plots_trovata = False
    results_trovato = False
    path_results = None

    for root, dirs, files in os.walk(base_path):
        
        # --- Gestione cartella plots ---
        if "plots" in dirs and not plots_trovata:
            plots_trovata = True
            plots_path = os.path.join(root, "plots")
            print(f"Trovata cartella: {plots_path}")

            # Svuota la cartella
            for item in os.listdir(plots_path):
                item_path = os.path.join(plots_path, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

            print("Cartella 'plots' svuotata.")

        # --- Gestione file results.csv ---
        if "results.csv" in files and not results_trovato:
            results_trovato = True
            path_results = os.path.join(root, "results.csv")
            print(f"Trovato file: {path_results}")

    # --- Scrittura o creazione results.csv ---
    intestazioni = "episodio;total_cost;total_co2\n"

    if results_trovato:
        with open(path_results, "w", encoding="utf-8") as f:
            f.write(intestazioni)
        print("File 'results.csv' svuotato e intestazioni riscritte.")
    else:
        # Se non trovato, lo crea nella base_path
        path_results = os.path.join(base_path, "results.csv")
        with open(path_results, "w", encoding="utf-8") as f:
            f.write(intestazioni)
        print(f"File 'results.csv' non trovato. Creato nuovo file in: {path_results}")

    if not plots_trovata:
        print("Nessuna cartella 'plots' trovata.")



# Implementato:
# forecasting meteo, forecasting c02 
# dati meteo reali, co2 intensity reale,  



if __name__ == "__main__":

    # elimino vecchi plots
    # svuota resluts.csv
    pulizia_progetto(".")
    print("Start to training.... results in resuts.csv")

    # -------------------------------------------------------
    # RUN TRAINING
    # -------------------------------------------------------
    df = pd.read_csv("csvs/cluster_power_only_nodes_10days.csv")
    # df = pd.read_csv("cluster_power_only_nodes.csv")

    df["time"] = pd.to_datetime(df["time"])
    df["dt_hours"] = df["time"].diff().dt.total_seconds() / 3600
    df["dt_hours"] = df["dt_hours"].fillna(0)
    
    # INSERISCO PREZZI
    df["price_base"] = df["time"].apply(dynamic_low_price)
    df["price_high"] = df["price_base"] * 3

    # INSERISCO VALORI RINNOVABILI
    rm = RenewableModels(seed=42)
    #df["P_solar"] = rm.solar_cloudy2(df)
    #df["P_solar"] = rm.solar_simple(df)
    #df["P_wind"] = rm.wind_stochastic(df)
    #df["P_wind"] = rm.wind_uniform(df)
    df["P_wind"] = rm.wind_from_openmeteo(df)
    df["P_solar"] = rm.solar_from_openmeteo(df)
    df["P_ren"]   = df["P_solar"] + df["P_wind"]

    # INSERISCO VALORI C02
    cm = CarbonIntensityModels(csv_file="csvs/carbon_intensity_IT-NORTH-2020.csv")
    df["co2_intensity"] = cm.co2_from_csv(df)

    # previsioni del tempo e co2 intensity
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
    df["forecast_co2_intensity_1h"] = (
        df["co2_intensity"]
        .rolling(window=steps_per_hour, min_periods=1)
        .sum()
        .shift(-steps_per_hour)
        .fillna(0)
    )
    df["forecast_co2_intensity_6h"] = (
        df["co2_intensity"]
        .rolling(window=steps_per_hour*6, min_periods=1)
        .sum()
        .shift(-steps_per_hour)
        .fillna(0)
    )  


    vec_env = DummyVecEnv([lambda: HPCBatteryEnv(df)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_reward=10.0)    # Option Reward = False


    policy_kwargs = dict(
        net_arch=[256, 256, 128],
        activation_fn=th.nn.ReLU
    )

    # questa rete sembra fuznionare, più lenta ma fa meglio..
    
    # model = PPO(
    #     "MlpPolicy",
    #     vec_env,
    #     learning_rate=1e-4,
    #     n_steps=4096,
    #     batch_size=512,
    #     gae_lambda=0.92,
    #     ent_coef=0.01,
    #     clip_range=0.15,
    #     max_grad_norm=0.5,
    #     verbose=0,
    #     device="cpu"
    # )   
    model = PPO("MlpPolicy", vec_env,verbose=0,device="cpu")

    #model.learn(total_timesteps=250_000)
    model.learn(total_timesteps=10_000_000)

    # Save final model trained
    model.save(MODEL_PATH)
    vec_env.save(VEC_PATH)