# ============================================================
# RL AGENT - SCENARIO 3
# Obiettivo:
# imparare una policy che usi batteria e generatore per
# avvicinarsi ai risultati economici/ambientali della Sim 2,
# ma mantenendo il vincolo duro:
#           P_grid <= threshold
#
# Logica fisica dell'ambiente:
# 1) uso tutte le rinnovabili per coprire il carico
# 2) l'eventuale surplus rinnovabile carica sempre la batteria
# 3) se le rinnovabili non bastano, l'agente decide quanto usare la batteria
# 4) la rete può coprire solo la quota sotto threshold
# 5) il peack si copre con batteria o gener
# 6) se resta peak non coperto => violazione hard constraint (forte penalità)
# ============================================================


import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import gymnasium as gym
from gymnasium import spaces

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from utils.renewable_real import RenewableModels
from utils.co2 import CarbonIntensityModels
from utils.battery_model import Battery
from utils.generator_model import Generator
from utils.price_model import PriceModel


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = "outputRL"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_powercap")
VEC_PATH = os.path.join(MODEL_DIR, "vecnormalize.pkl")

# costi / CO2 green, coerenti con le simulazioni greedy
SOLAR_COST_EUR_PER_KWH = 0.05
WIND_COST_EUR_PER_KWH = 0.04

SOLAR_CO2_G_PER_KWH = 50.0
WIND_CO2_G_PER_KWH = 34.0

# pesi reward
ALPHA_COST = 1.0
ALPHA_CO2 = 0.001          # riduce la scala della CO2 rispetto al costo
ALPHA_VIOLATION = 50.0     # penalità forte per kWh non servito sopra threshold
ALPHA_CURTAIL = 0.0        # opzionale
ALPHA_SOC_FINAL = 0.0      # opzionale


# ============================================================
# ENV
# ============================================================

class HPCBatteryEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        threshold=400000,
        battery_capacity=3200000,
        max_charge_rate=3200000,
        max_discharge_rate=3200000,
        generator_max_power_w=500000,
        generator_min_power_w=50000,
        generator_efficiency=0.4,
        generator_fuel_cost_per_wh=0.00025,
        generator_co2_g_per_kwh=450.0,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True).copy()
        self.N = len(self.df)

        self.threshold = float(threshold)
        self.capacity = float(battery_capacity)
        self.max_charge_rate = float(max_charge_rate)
        self.max_discharge_rate = float(max_discharge_rate)

        self.generator_max_power_w = float(generator_max_power_w)
        self.generator_min_power_w = float(generator_min_power_w)
        self.generator_efficiency = float(generator_efficiency)
        self.generator_fuel_cost_per_wh = float(generator_fuel_cost_per_wh)
        self.generator_co2_g_per_kwh = float(generator_co2_g_per_kwh)

        self.action_levels = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            dtype=np.float32
        )

        # azioni:
        # 0) quota deficit coperta da batteria
        # 1) quota del peak sopra threshold coperta da generatore
        self.action_space = spaces.MultiDiscrete([
            len(self.action_levels),
            len(self.action_levels),
        ])

        # osservazioni:
        # [P_ratio, deficit_after_ren_ratio, battery_norm, time_left,
        #  price_base_norm, price_high_norm, co2_norm,
        #  hour_sin, hour_cos,
        #  P_ren_norm,
        #  E_forecast_1h_norm, E_forecast_6h_norm,
        #  co2_forecast_1h_norm, co2_forecast_6h_norm]
        self.observation_space = spaces.Box(
            low=np.array([0., 0., 0., 0., 0., 0., 0., -1., -1., 0., 0., 0., 0., 0.], dtype=np.float32),
            high=np.array([3., 3., 1., 1., 1., 1., 1.,  1.,  1., 3., 3., 3., 1., 1.], dtype=np.float32),
            dtype=np.float32
        )

        # logging
        self.episode_idx = -1
        self.reset_logs()

        # normalizzazioni statiche
        self.price_base_max = max(float(self.df["price_base"].max()), 1e-9)
        self.price_high_max = max(float(self.df["price_high"].max()), 1e-9)
        self.co2_min = float(self.df["co2_intensity"].min())
        self.co2_max = float(self.df["co2_intensity"].max())

        self.reset()

    def reset_logs(self):
        self.battery_history = []
        self.battery_use_history = []
        self.gen_use_history = []
        self.time_history = []
        self.cost_history = []
        self.co2_history = []
        self.curtailment_history = []
        self.violation_history = []

        self.grid_cost_history = []
        self.grid_co2_history = []
        self.gen_cost_history = []
        self.gen_co2_history = []
        self.green_cost_history = []
        self.green_co2_history = []
        self.batt_cost_history = []
        self.batt_co2_history = []

    def _get_obs(self):
        t = min(self.t, self.N - 1)

        P_load = float(self.df.loc[t, "power"])
        P_ren = float(self.df.loc[t, "P_ren"])
        P_deficit_after_ren = max(P_load - P_ren, 0.0)

        P_ratio = np.clip(P_load / (self.threshold + 1e-9), 0.0, 3.0)
        deficit_after_ren_ratio = np.clip(P_deficit_after_ren / (self.threshold + 1e-9), 0.0, 3.0)
        battery_norm = float(self.battery.energy / max(self.battery.capacity, 1e-9))

        ts = self.df.loc[t, "time"]
        hour = ts.hour + ts.minute / 60.0
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)

        price_base = float(self.df.loc[t, "price_base"])
        price_high = float(self.df.loc[t, "price_high"])
        price_base_norm = price_base / self.price_base_max
        price_high_norm = price_high / self.price_high_max

        co2_int = float(self.df.loc[t, "co2_intensity"])
        co2_norm = (co2_int - self.co2_min) / (self.co2_max - self.co2_min + 1e-9)

        time_left = 1.0 - (t / max(self.N - 1, 1))

        P_ren_norm = np.clip(P_ren / (self.threshold + 1e-9), 0.0, 3.0)

        E_forecast_1h = float(self.df.loc[t, "forecast_E_ren_1h"])
        E_forecast_6h = float(self.df.loc[t, "forecast_E_ren_6h"])
        E_forecast_1h_norm = np.clip(E_forecast_1h / (self.capacity + 1e-9), 0.0, 3.0)
        E_forecast_6h_norm = np.clip(E_forecast_6h / (6.0 * self.capacity + 1e-9), 0.0, 3.0)

        co2_forecast_1h = float(self.df.loc[t, "forecast_co2_intensity_1h"])
        co2_forecast_6h = float(self.df.loc[t, "forecast_co2_intensity_6h"])
        co2_forecast_1h_norm = (co2_forecast_1h - self.co2_min) / (self.co2_max - self.co2_min + 1e-9)
        co2_forecast_6h_norm = (co2_forecast_6h - self.co2_min) / (self.co2_max - self.co2_min + 1e-9)

        obs = np.array([
            P_ratio,
            deficit_after_ren_ratio,
            battery_norm,
            time_left,
            price_base_norm,
            price_high_norm,
            co2_norm,
            hour_sin,
            hour_cos,
            P_ren_norm,
            E_forecast_1h_norm,
            E_forecast_6h_norm,
            co2_forecast_1h_norm,
            co2_forecast_6h_norm
        ], dtype=np.float32)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # salva episodio precedente
        if self.episode_idx >= 0:
            os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

            plt.figure(figsize=(14, 5))
            plt.plot(mdates.date2num(self.time_history), self.battery_history)
            plt.xlabel("Time")
            plt.ylabel("Battery Charge (Wh)")
            plt.title(f"Battery State of Charge - Episode {self.episode_idx}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "plots", f"battery_plot_ep{self.episode_idx}.png"))
            plt.close()

            csv_path = os.path.join(OUTPUT_DIR, "results_sim3_hardconstraint.csv")
            end_soc = self.battery.info()["SOC"]
            end_capacity_ratio = self.battery.info()["capacity_Wh"] / self.capacity

            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{self.episode_idx};"
                    f"{sum(self.cost_history):.4f};"
                    f"{sum(self.co2_history)/1000.0:.4f};"
                    f"{end_soc:.4f};"
                    f"{end_capacity_ratio:.4f};"
                    f"{sum(self.battery_use_history):.3f};"
                    f"{sum(self.gen_use_history):.3f};"
                    f"{sum(self.violation_history):.3f}\n"
                )

        self.episode_idx += 1
        self.reset_logs()

        self.t = 0

        self.battery = Battery(
            capacity_wh=self.capacity,
            initial_charge_wh=self.capacity / 2.0,
            max_charge_rate_w=self.max_charge_rate,
            max_discharge_rate_w=self.max_discharge_rate,
        )

        self.generator = Generator(
            max_power_w=self.generator_max_power_w,
            min_power_w=self.generator_min_power_w,
            efficiency=self.generator_efficiency,
            fuel_cost_per_wh=self.generator_fuel_cost_per_wh,
            co2_g_per_kwh=self.generator_co2_g_per_kwh
        )

        self.battery_history = [self.battery.energy]
        self.time_history = [self.df.loc[0, "time"]]
        self.cost_history = [0.0]
        self.co2_history = [0.0]
        self.gen_use_history = [0.0]
        self.battery_use_history = [0.0]
        self.curtailment_history = [0.0]
        self.violation_history = [0.0]

        self.grid_cost_history = [0.0]
        self.grid_co2_history = [0.0]
        self.gen_cost_history = [0.0]
        self.gen_co2_history = [0.0]
        self.green_cost_history = [0.0]
        self.green_co2_history = [0.0]
        self.batt_cost_history = [0.0]
        self.batt_co2_history = [0.0]

        return self._get_obs(), {}

    def step(self, action):
        idx_discharge, idx_gen_peak = action
        a_discharge = float(self.action_levels[idx_discharge])
        a_gen_peak = float(self.action_levels[idx_gen_peak])

        t = self.t
        dt = float(self.df.loc[t, "dt_hours"])

        if dt <= 0:
            self.t += 1
            terminated = self.t >= self.N
            obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, terminated, False, {}

        # --------------------------------------------------
        # INPUT
        # --------------------------------------------------
        P_load = float(self.df.loc[t, "power"])
        P_wind = float(self.df.loc[t, "P_wind"])
        P_solar = float(self.df.loc[t, "P_solar"])
        P_ren = float(self.df.loc[t, "P_ren"])

        price_base = float(self.df.loc[t, "price_base"])
        price_high = float(self.df.loc[t, "price_high"])
        co2_intensity = float(self.df.loc[t, "co2_intensity"])

        # --------------------------------------------------
        # GREEN COST / CO2
        # imputati sulla produzione totale
        # --------------------------------------------------
        E_wind_total = P_wind * dt
        E_solar_total = P_solar * dt

        green_cost = (
            (E_wind_total / 1000.0) * WIND_COST_EUR_PER_KWH +
            (E_solar_total / 1000.0) * SOLAR_COST_EUR_PER_KWH
        )
        green_co2_g = (
            (E_wind_total / 1000.0) * WIND_CO2_G_PER_KWH +
            (E_solar_total / 1000.0) * SOLAR_CO2_G_PER_KWH
        )

        # --------------------------------------------------
        # RENEWABLE FIRST
        # --------------------------------------------------
        P_from_ren = min(P_load, P_ren)
        P_surplus = max(P_ren - P_load, 0.0)
        P_deficit = max(P_load - P_ren, 0.0)

        E_surplus = P_surplus * dt
        E_deficit = P_deficit * dt

        # surplus -> batteria sempre
        E_charged = 0.0
        E_curtail = E_surplus
        if E_surplus > 0:
            E_charged = self.battery.charge(P_surplus, dt)
            E_curtail = max(E_surplus - E_charged, 0.0)

        # --------------------------------------------------
        # BATTERY ACTION
        # --------------------------------------------------
        E_from_batt = 0.0
        if E_deficit > 0 and a_discharge > 0:
            requested_batt = a_discharge * E_deficit
            E_from_batt = self.battery.discharge(requested_batt / dt, dt)
            E_from_batt = min(E_from_batt, E_deficit)

        E_after_batt = max(E_deficit - E_from_batt, 0.0)

        # --------------------------------------------------
        # GENERATOR ACTION
        # Ora può coprire qualsiasi parte del deficit residuo,
        # non solo il peak
        # --------------------------------------------------
        E_from_gen = 0.0
        if E_after_batt > 0 and a_gen_peak > 0:
            requested_gen = a_gen_peak * E_after_batt
            P_requested_gen = requested_gen / dt
            E_from_gen = self.generator.dispatch(P_requested_gen, dt)
            E_from_gen = min(E_from_gen, E_after_batt)

        E_after_gen = max(E_after_batt - E_from_gen, 0.0)

        # --------------------------------------------------
        # GRID LAST
        # La rete copre solo il resto, ma NON può superare threshold
        # --------------------------------------------------
        P_grid = E_after_gen / dt if dt > 0 else 0.0
        P_grid_allowed = min(P_grid, self.threshold)
        P_grid_violation = max(P_grid - self.threshold, 0.0)

        E_from_grid = P_grid_allowed * dt
        E_violation = P_grid_violation * dt

        # costo rete: solo quota entro threshold
        cost_grid = E_from_grid * price_base
        cost_gen = self.generator.get_cost(E_from_gen)

        E_batt_throughput = E_charged + E_from_batt
        batt_cost = self.battery.step_cost(dt, E_batt_throughput)

        step_cost = green_cost + batt_cost + cost_grid + cost_gen

        # co2 rete
        co2_grid = (E_from_grid / 1000.0) * co2_intensity
        co2_gen = self.generator.get_co2(E_from_gen / 1000.0)
        batt_co2_g = self.battery.step_co2_g(dt, E_batt_throughput)

        step_co2_g = green_co2_g + batt_co2_g + co2_grid + co2_gen

        # print("step_cost = ", step_cost)
        # print("step_co2_g ,", step_co2_g)


        # --------------------------------------------------
        # REWARD
        # --------------------------------------------------
        violation_kWh = E_violation / 1000.0
        reward = (
            - ALPHA_COST * step_cost
            - ALPHA_CO2 * step_co2_g
            - ALPHA_VIOLATION * violation_kWh
            #- ALPHA_CURTAIL * (E_curtail / 1000.0)
        )

        # opzionale: vincolo sul SOC finale
        last_step = (self.t == self.N - 1)
        if last_step and ALPHA_SOC_FINAL > 0:
            # final_soc = self.battery.energy / max(self.battery.capacity, 1e-9)
            # reward -= ALPHA_SOC_FINAL * abs(final_soc - 0.5)
            reward = (
                - sum(self.cost_history)
                - ALPHA_VIOLATION * sum(self.violation_history)
                #- ALPHA_CO2 * sum(self.co2_history) 
                
            )
            

        # --------------------------------------------------
        # LOG
        # --------------------------------------------------
        self.cost_history.append(step_cost)
        self.co2_history.append(step_co2_g)
        self.gen_use_history.append(E_from_gen)
        self.battery_use_history.append(E_from_batt)
        self.curtailment_history.append(E_curtail)
        self.violation_history.append(E_violation)

        self.grid_cost_history.append(cost_grid)
        self.grid_co2_history.append(co2_grid)
        self.gen_cost_history.append(cost_gen)
        self.gen_co2_history.append(co2_gen)
        self.green_cost_history.append(green_cost)
        self.green_co2_history.append(green_co2_g)
        self.batt_cost_history.append(batt_cost)
        self.batt_co2_history.append(batt_co2_g)

        self.battery.step(dt)

        self.battery_history.append(self.battery.energy)
        self.time_history.append(self.df.loc[t, "time"])

        # --------------------------------------------------
        # NEXT
        # --------------------------------------------------
        self.t += 1
        terminated = self.t >= self.N

        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_obs()

        info = {
            "step_cost": step_cost,
            "step_co2_g": step_co2_g,
            "grid_cost": cost_grid,
            "gen_cost": cost_gen,
            "batt_cost": batt_cost,
            "green_cost": green_cost,
            "grid_co2_g": co2_grid,
            "gen_co2_g": co2_gen,
            "batt_co2_g": batt_co2_g,
            "green_co2_g": green_co2_g,
            "E_from_grid": E_from_grid,
            "E_from_gen": E_from_gen,
            "E_from_batt": E_from_batt,
            "E_to_batt": E_charged,
            "E_violation": E_violation,
            "E_curtail": E_curtail,
        }

        return obs, float(reward), terminated, False, info


# ============================================================
# HELPERS
# ============================================================

def pulizia_progetto(base_path="."):
    plots_trovata = False
    results_trovato = False
    path_results = None

    for root, dirs, files in os.walk(base_path):
        if os.path.basename(root) != OUTPUT_DIR:
            continue

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

        if "results_sim3_hardconstraint.csv" in files and not results_trovato:
            results_trovato = True
            path_results = os.path.join(root, "results_sim3_hardconstraint.csv")
            print(f"Trovato file: {path_results}")

    intestazioni = "episodio;total_cost;total_co2_kg;SOC_finale;capacity_ratio;battery_use_Wh;gen_use_Wh;violation_Wh\n"

    if results_trovato:
        with open(path_results, "w", encoding="utf-8") as f:
            f.write(intestazioni)
        print(f"File {OUTPUT_DIR}/results_sim3_hardconstraint.csv svuotato e intestazioni riscritte.")
    else:
        path_results = os.path.join(base_path, OUTPUT_DIR, "results_sim3_hardconstraint.csv")
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
                return False
        return True


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    THRESHOLD = 400000
    BATTERY_CAPACITY = 3_200_000
    P_MAX_RENS = (400000, 450000)

    pulizia_progetto(".")
    print("Start training... results in outputRL/results_sim3_hardconstraint.csv")

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------
    df = pd.read_csv("csvs/cluster_power_only_nodes_30days.csv")
    # df = pd.read_csv("cluster_power_only_nodes.csv")

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["dt_hours"] = df["time"].diff().dt.total_seconds() / 3600.0
    df["dt_hours"] = df["dt_hours"].fillna(0.0)

    price_model = PriceModel()
    df["price_base"], df["price_high"] = price_model.prices_from_df(df)

    rm = RenewableModels(seed=42)
    df["P_wind"] = rm.wind_from_openmeteo(df, P_MAX_RENS[0])
    df["P_solar"] = rm.solar_from_openmeteo(df, P_MAX_RENS[1])
    df["P_ren"] = df["P_solar"] + df["P_wind"]

    cm = CarbonIntensityModels(csv_file="csvs/carbon_intensity_IT-NORTH-2020.csv")
    df["co2_intensity"] = cm.co2_from_csv(df)

    # --------------------------------------------------------
    # FORECASTS
    # --------------------------------------------------------
    valid_dt = df.loc[df["dt_hours"] > 0, "dt_hours"]
    dt_hours_nominal = float(valid_dt.iloc[0]) if len(valid_dt) > 0 else (20.0 / 3600.0)
    steps_per_hour = max(int(round(1.0 / dt_hours_nominal)), 1)

    # energia rinnovabile futura in Wh
    df["E_ren_step"] = df["P_ren"] * df["dt_hours"]

    df["forecast_E_ren_1h"] = (
        df["E_ren_step"]
        .rolling(window=steps_per_hour, min_periods=1)
        .sum()
        .shift(-steps_per_hour)
        .fillna(0.0)
    )

    df["forecast_E_ren_6h"] = (
        df["E_ren_step"]
        .rolling(window=steps_per_hour * 6, min_periods=1)
        .sum()
        .shift(-steps_per_hour * 6)
        .fillna(0.0)
    )

    df["forecast_co2_intensity_1h"] = (
        df["co2_intensity"]
        .rolling(window=steps_per_hour, min_periods=1)
        .mean()
        .shift(-steps_per_hour)
        .fillna(df["co2_intensity"].mean())
    )

    df["forecast_co2_intensity_6h"] = (
        df["co2_intensity"]
        .rolling(window=steps_per_hour * 6, min_periods=1)
        .mean()
        .shift(-steps_per_hour * 6)
        .fillna(df["co2_intensity"].mean())
    )

    # --------------------------------------------------------
    # ENV
    # --------------------------------------------------------
    env = HPCBatteryEnv(
        df=df,
        threshold=THRESHOLD,
        battery_capacity=BATTERY_CAPACITY,
        max_charge_rate=BATTERY_CAPACITY,
        max_discharge_rate=BATTERY_CAPACITY,
        generator_max_power_w=500000,
        generator_min_power_w=50000,
        generator_efficiency=0.4,
        generator_fuel_cost_per_wh=0.00025,
        generator_co2_g_per_kwh=450.0,
    )

    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_reward=10.0
    )

    policy_kwargs = dict(
        net_arch=[256, 256, 128],
        activation_fn=th.nn.ReLU
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
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

    stop_callback = StopAfterNEpisodes(max_episodes=200)

    model.learn(
        total_timesteps=10_000_000,
        callback=stop_callback
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    vec_env.save(VEC_PATH)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved VecNormalize stats to: {VEC_PATH}")