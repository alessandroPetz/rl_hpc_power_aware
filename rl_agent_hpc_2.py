####

# IN QUESTA SIMULAZIONE, LA BATTERIA DIVENTA UN ACCUMULATORE DI ENERGIA RINNOVABILE
# AD OGNI STEP, SI CERCA DI COPRIRE LA RICHISTA DEL CLUSTER CON LE RINNOVABILI
# SE AVANZA ENERGIA RINNOVABILE, SI CARICA LA BATTERIA * a_charge
# SE NON BASTA, SI USA LA BATTTERIA * a_discharge
# SE NON BASTA NEANCHE LA BATTERIA, SI USA LA RETE ELETTRICA (APPLICANDO COSTI E THRESHOLD COME IN PRECEDENZA) 

# RL agent per cercare di migliorare rispetto alla regola base
# qui si cerca di minimizzare sia il costo che il co2

# dato che non si ricava tanto, provaiamo a puntare sullo soh della batteria. 
# riusciamo a spendere e enettere uguale, ma mantenendo uno soh migliore?

####

# differenze col v1

# | Prima                  | Adesso                       |
# | ---------------------- | ---------------------------- |
# | carica sempre          | carica solo se conviene      |
# | curtailment forzato    | curtailment decisionale      |
# | batteria “gratis”      | batteria con costo implicito |
# | azione spesso ignorata | azione sempre rilevante      |
# | PPO confuso            | PPO stabile                  |



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
from utils.battery_model import Battery
from utils.price_model import PriceModel

OUTPUT_DIR = "outputRL"
MODEL_DIR = OUTPUT_DIR + "/models"
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
            high=np.array([3., 3., 1., 1., 1., 1., 1. ,1., 1., 4., 1., 1., 1., 1.], dtype=np.float32)
        )
        # senza c02 forcasting
        # self.observation_space = spaces.Box(
        #     low=np.array([0., 0., 0., 0., 0., 0., 0. ,0., 0., 0., 0., 0. ], dtype=np.float32),
        #     high=np.array([3., 3., 1., 1., 1., 1., 1. ,1., 1., 1., 1., 1.], dtype=np.float32)
        # )

        # -------------------------------
        # Action = singola variabile [-1, 1]
        # a < 0 → scarica
        # a > 0 → carica
        # -------------------------------
        

        # ACTION DISCRETA
        # self.action_levels = np.array([
        #     -1.0, -0.9, -0.8, -0.7, -0.6 -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 
        #     0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
        #     ], dtype=np.float32)
        # self.action_space = spaces.Discrete(len(self.action_levels))

        # non carico la batteria da rete. a DECIDE  quanto BATTERIA UTILIZZARE, IN OGNI CASISTICA)
        # provo a separare 
        #   a_charge= quanto ricarico la batterie con rinovabili? nel caso ren > richiesta (forse conviene semmpre a=1????)
        #   a_discharge, quanta batteria uso? nel caso ren non basti 
        self.action_levels = np.array([
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
            ], dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([
            len(self.action_levels),   # a_charge
            len(self.action_levels)    # a_discharge
        ])
        
        # ACTION CONTINUA
        #self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
 

        self.reset()

    def _get_obs(self):
        t = min(self.t, self.N - 1)
        P = float(self.df.loc[t, "power"])          # W
        P_ratio = np.clip(P / self.threshold, 0, 3)
        P_peak  = np.clip(max(P - self.threshold, 0) / self.threshold, 0, 3)
        battery_norm = float(self.battery.energy / self.battery.capacity)

        ts = self.df.loc[t, "time"]
        hour = ts.hour + ts.minute / 60.0
        hour_sin = np.sin(2 * np.pi * hour / 24.)
        hour_cos = np.cos(2 * np.pi * hour / 24.)

        price_base = float(self.df.loc[t, "price_base"])
        price_base_norm = price_base / (self.df["price_base"].max() + 1e-9)
        
        co2_int = float(self.df.loc[t, "co2_intensity"])
        co2_int_norm = co2_int / (self.df["co2_intensity"].max() + 1e-9)

        time_left = 1.0 - (t / (self.N - 1))
        prev_a = getattr(self, "prev_action", 0.0)

        P_ren = float(self.df.loc[t, "P_ren"])
        P_ren_norm = P_ren / (self.threshold + 1e-9)   # può superare 1

        E_forecast_1h = float(self.df.loc[t, "forecast_P_ren_1h"])  # qui è energia Wh (sum)
        E_forecast_6h = float(self.df.loc[t, "forecast_P_ren_6h"])

        # normalizzo forecast su capacità (Wh)
        E_forecast_1h_norm = E_forecast_1h / (self.capacity + 1e-9)
        E_forecast_6h_norm = E_forecast_6h / (6.0 * self.capacity + 1e-9)  # opzionale scaling

        co2_forecast_1h = float(self.df.loc[t, "forecast_co2_intensity_1h"])
        co2_forecast_6h = float(self.df.loc[t, "forecast_co2_intensity_6h"])

        # normalizzo minmax # TODO se non funziona, togliere
        co2_min = self.df["co2_intensity"].min()
        co2_max = self.df["co2_intensity"].max()

        co2_int_norm = (co2_int - co2_min) / (co2_max - co2_min + 1e-9)
        co2_forecast_1h_norm = (co2_forecast_1h - co2_min) / (co2_max - co2_min + 1e-9)
        co2_forecast_6h_norm = (co2_forecast_6h - co2_min) / (co2_max - co2_min + 1e-9)

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
            filename = f"{OUTPUT_DIR}/plots/battery_plot_ep{self.episode_idx}.png"
            plt.savefig(filename)
            plt.close()

            #print(f"[Episode {self.episode_idx}] total cost = {sum(self.cost_history) :.4f}")
            #print(f"[Episode {self.episode_idx}] CO2 totale (kg): {sum(self.co2_history)/1000:.3f}")

            csv_path = os.path.join(".", OUTPUT_DIR ,"results.csv")

            # Aggiunge la riga
            end_SOC = self.battery.info()['SOC']
            end_capacity =  self.battery.info["capacity_Wh"] / BATTERY_CAPACITY
            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(f"{self.episode_idx};{sum(self.cost_history):.4f};{sum(self.co2_history)/1000:.3f};{end_SOC};{end_capacity},\n")


        self.episode_idx += 1
        self.battery_history = [self.capacity / 2]
        self.time_history =  [self.df.loc[0, "time"]]
        self.cost_history = [0]
        self.co2_history = [0]
        self.curtailment_history= [0]
        ## END PLOT LOG ##

        self.t = 0
        self.battery = Battery(
            capacity_wh=self.capacity,
            initial_charge_wh=self.capacity / 2,
            max_charge_rate_w=self.max_charge_rate,
            max_discharge_rate_w=self.max_discharge_rate,
        )
        return self._get_obs(), {}

    def step(self, action):

        # -------------------
        # ACTION
        # -------------------
        idx_charge, idx_discharge = action

        a_charge = float(self.action_levels[idx_charge])
        a_discharge = float(self.action_levels[idx_discharge])

        # # a=1 # come simulazione..
        # a_charge = 1     # carico sempre con tutto i lsurplus da rinnovabile
        # a_discharge = 1  # uso sempre tutta la batteria utilizzabile.
        
        t = self.t
        dt = float(self.df.loc[t, "dt_hours"])
        time = self.df.loc[t, "time"]

        P_load = float(self.df.loc[t, "power"])      # W
        P_ren  = float(self.df.loc[t, "P_ren"])      # W

        price_base = float(self.df.loc[t, "price_base"])
        price_high = float(self.df.loc[t, "price_high"])
        CO2_G_PER_KWH = float(self.df.loc[t, "co2_intensity"])

        # dt nullo → skip fisico
        #non dovrebbe esistere
        if dt <= 0:
            self.t += 1
            terminated = self.t >= self.N - 1
            return self._get_obs(), 0.0, terminated, False, {}

        # ---------------------------------------------------------
        # 1. RINNOVABILI → CARICO (SEMPRE)
        # ---------------------------------------------------------
        P_from_ren = min(P_load, P_ren)
        P_surplus  = max(P_ren - P_from_ren, 0.0)

        E_surplus = P_surplus * dt
        E_deficit = (P_load - P_from_ren) * dt

        # ---------------------------------------------------------
        # 2. DECISIONE RL: CARICA DA RINNOVABILI
        # ---------------------------------------------------------
        E_to_batt = a_charge * E_surplus
        # E_to_batt = E_surplus   # nel caso c'è suplus lo prendo sempre tutto... perchè non dovrebbe?
        E_charged = 0.0
        E_curtail = 0.0

        #
        if E_to_batt > 0:

            #  E_surplus > 0 e a > 0 e E_deficit < 0
            E_charged = self.battery.charge(E_to_batt / dt, dt)
            E_curtail = E_surplus - E_charged
        else:
            # a = 0 o E_surplus <= 0
            E_curtail = E_surplus

        # ---------------------------------------------------------
        # 3. DECISIONE RL: USA BATTERIA PER COPRIRE DEFICIT
        # ---------------------------------------------------------
        E_from_batt = 0.0
        if E_deficit > 0 and a_discharge > 0:
            # E_surplus <=0
            E_from_batt = self.battery.discharge(
                (a_discharge * E_deficit) / dt,
                dt
            )
            E_from_batt = min(E_from_batt, E_deficit)

        E_grid = E_deficit - E_from_batt

        # ---------------------------------------------------------
        # 4. COSTI E CO2
        # ---------------------------------------------------------
        P_grid = E_grid / dt

        E_base = min(P_grid, self.threshold) * dt
        E_peak = max(P_grid - self.threshold, 0) * dt

        cost = E_base * price_base + E_peak * price_high
        co2_g = (E_base + E_peak) / 1000 * CO2_G_PER_KWH

        # ---------------------------------------------------------
        # 5. REWARD (semplice ma corretto)
        # ---------------------------------------------------------
        battery_throughput_kwh = self.battery.throughput_wh / 1000

        # print("cost = ", cost, " co2_g = ", co2_g, ", battery_throughput_kwh = ", battery_throughput_kwh)

        reward = (
            - 0.1 * cost
            - 0.1 / 500 * co2_g 
            - 0.02 / 1000 * battery_throughput_kwh
        )


        # ---------------------------------------------------------
        # LOGS
        # ---------------------------------------------------------
        self.battery_history.append(self.battery.energy)
        self.cost_history.append(cost)
        self.co2_history.append(co2_g)
        self.curtailment_history.append(E_curtail)
        self.time_history.append(time)

        self.battery.step(dt)

        self.t += 1
        terminated = self.t >= self.N - 1

        # shape reward
        if terminated:
            reward -= sum(self.cost_history)
            reward -= sum(self.co2_history) /500

        return self._get_obs(), float(reward), terminated, False, {}


def pulizia_progetto(base_path="."):
    plots_trovata = False
    results_trovato = False
    path_results = None

    for root, dirs, files in os.walk(base_path):

        if os.path.basename(root) != OUTPUT_DIR:
            continue

        # --- Gestione cartella plots ---
        if "plots" in dirs and not plots_trovata:
            plots_trovata = True
            plots_path = os.path.join(root, "plots")
            print(f"Trovata cartella: {plots_path}")

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
    intestazioni = "episodio;total_cost;total_co2;SOC_finale\n"

    if results_trovato:
        with open(path_results, "w", encoding="utf-8") as f:
            f.write(intestazioni)
        print(f"File {OUTPUT_DIR}/results.csv svuotato e intestazioni riscritte.")
    else:
        path_results = os.path.join(base_path, OUTPUT_DIR, "results.csv")
        os.makedirs(os.path.dirname(path_results), exist_ok=True)
        with open(path_results, "w", encoding="utf-8") as f:
            f.write(intestazioni)
        print(f"Creato nuovo file: {path_results}")

    if not plots_trovata:
        print("Nessuna cartella 'plots' trovata.")

class StopAfterNEpisodes(BaseCallback):
    def __init__(self, max_episodes, verbose=1):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.episode_count = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is not None:
            self.episode_count += int(sum(dones))

            if self.episode_count >= self.max_episodes:
                if self.verbose:
                    print(f"Stopping training after {self.episode_count} episodes")
                return False  # STOP TRAINING

        return True


if __name__ == "__main__":

    # THRESHOLD = 400000          # W
    # BATTERY_CAPACITY = 3_200_000      # Wh
    # P_MAX_RENS = (400000,450000)


    # THRESHOLD = 300000          # W
    # BATTERY_CAPACITY = 800_000      # Wh
    # P_MAX_RENS = (200000,250000)

    THRESHOLD = 500000          # W
    BATTERY_CAPACITY = 6400000      # Wh
    P_MAX_RENS = (400000,450000)


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
    price_model = PriceModel()
    df["price_base"], df["price_high"] = price_model.prices_from_df(df)

    # INSERISCO VALORI RINNOVABILI
    rm = RenewableModels(seed=42)
    #df["P_solar"] = rm.solar_cloudy2(df)
    #df["P_solar"] = rm.solar_simple(df)
    #df["P_wind"] = rm.wind_stochastic(df)
    #df["P_wind"] = rm.wind_uniform(df)
    df["P_wind"] = rm.wind_from_openmeteo(df, P_MAX_RENS[0])
    df["P_solar"] = rm.solar_from_openmeteo(df, P_MAX_RENS[1])
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
        .shift(-steps_per_hour*6)
        .fillna(0)
    )
    df["forecast_co2_intensity_1h"] = (
        df["co2_intensity"]
        .rolling(window=steps_per_hour, min_periods=1)
        .mean()
        .shift(-steps_per_hour)
        .fillna(0)
    )
    df["forecast_co2_intensity_6h"] = (
        df["co2_intensity"]
        .rolling(window=steps_per_hour*6, min_periods=1)
        .mean()
        .shift(-steps_per_hour * 6)
        .fillna(0)
    )  

    env= HPCBatteryEnv(
        df, 
        threshold=THRESHOLD,
        battery_capacity=BATTERY_CAPACITY,
        max_charge_rate=BATTERY_CAPACITY,      # Wh/h
        max_discharge_rate=BATTERY_CAPACITY
    )

    

    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_reward=10.0)    # Option Reward = False


    policy_kwargs = dict(
        net_arch=[256, 256, 128],
        activation_fn=th.nn.ReLU
    )

    # questa rete sembra fuznionare, più lenta ma fa meglio..
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=512,
        gae_lambda=0.92,
        ent_coef=0.01,
        clip_range=0.15,
        max_grad_norm=0.5,
        verbose=0,
        device="cpu"
    )   
    # model = PPO("MlpPolicy", vec_env,verbose=0,device="cpu")

    #model.learn(total_timesteps=250_000)
    stop_callback = StopAfterNEpisodes(max_episodes=100)

    model.learn(
        total_timesteps=10_000_000,  # limite alto
        callback=stop_callback
    )

    # Save final model trained
    model.save(MODEL_PATH)
    vec_env.save(VEC_PATH)