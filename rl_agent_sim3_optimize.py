####
# RL optimization of the "All" simulation scenario
# (grid + battery + generator + renewable generation)
#
# Objective:
# use renewable generation, battery storage, backup generation, and grid
# import to minimize the combined cost and CO2 impact of the dispatch.
#
# Physical logic of the environment:
# 1) renewable generation is used first to cover the load
# 2) any renewable surplus is used to charge the battery
# 3) if renewables are not sufficient, the agent decides how much battery
#    energy to discharge
# 4) the agent also decides how much backup generation to use
# 5) any remaining demand is supplied by the grid
# 6) grid energy is split into base and peak components according to the
#    configured threshold
# 7) the total objective includes grid, renewable, battery, and generator
#    cost and CO2 contributions
####



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
from utils.generator_model import Generator
from utils.price_model import PriceModel

OUTPUT_DIR = "outputRL"
MODEL_DIR = OUTPUT_DIR + "/models"
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_powercap")
VEC_PATH = os.path.join(MODEL_DIR, "vecnormalize.pkl")

# -------------------------------------------------------
# COSTI / CO2 GREEN
# -------------------------------------------------------
SOLAR_COST_EUR_PER_KWH = 0.05
WIND_COST_EUR_PER_KWH = 0.04

SOLAR_CO2_G_PER_KWH = 50.0
WIND_CO2_G_PER_KWH = 34.0


class HPCBatteryEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        threshold=400000,
        battery_capacity=3200000,
        max_charge_rate=3200000,
        max_discharge_rate=3200000
    ):
        super().__init__()

        # LOG
        self.episode_idx = -1
        self.battery_history = []
        self.battery_use_history = []
        self.gen_use_history = []
        self.time_history = []
        self.cost_history = []
        self.co2_history = []
        self.curtailment_history = []

        self.green_cost_history = []
        self.green_co2_history = []
        self.batt_cost_history = []
        self.batt_co2_history = []
        self.grid_cost_history = []
        self.grid_co2_history = []
        self.gen_cost_history = []
        self.gen_co2_history = []

        self.df = df.reset_index(drop=True)
        self.N = len(self.df)

        self.threshold = threshold
        self.capacity = battery_capacity
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate

        # Observation:
        # [P_ratio, P_peak, battery_norm, time_left, price_base_norm, co2_intensity_norm,
        #  hour_sin, hour_cos, P_ren_norm, forecast_ren_norm1h, forecast_ren_norm6h,
        #  co2_forecast_1h_norm, co2_forecast_6h_norm]
        self.observation_space = spaces.Box(
            low=np.array([0., 0., 0., 0., 0., 0., -1., -1., 0., 0., 0., 0., 0.], dtype=np.float32),
            high=np.array([3., 3., 1., 1., 1., 1., 1., 1., 4., 1., 1., 1., 1.], dtype=np.float32),
            dtype=np.float32
        )

        # ACTION
        self.action_levels = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([
            len(self.action_levels),  # a_discharge
            len(self.action_levels)   # a_gen
        ])

        self.reset()

    def _get_obs(self):
        t = min(self.t, self.N - 1)

        P = float(self.df.loc[t, "power"])
        P_ratio = np.clip(P / (self.threshold + 1e-9), 0, 3)
        P_peak = np.clip(max(P - self.threshold, 0) / (self.threshold + 1e-9), 0, 3)
        battery_norm = float(self.battery.energy / max(self.battery.capacity, 1e-9))

        ts = self.df.loc[t, "time"]
        hour = ts.hour + ts.minute / 60.0
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)

        price_base = float(self.df.loc[t, "price_base"])
        price_base_norm = price_base / (self.df["price_base"].max() + 1e-9)

        co2_int = float(self.df.loc[t, "co2_intensity"])

        co2_min = self.df["co2_intensity"].min()
        co2_max = self.df["co2_intensity"].max()
        co2_int_norm = (co2_int - co2_min) / (co2_max - co2_min + 1e-9)

        time_left = 1.0 - (t / max(self.N - 1, 1))

        P_ren = float(self.df.loc[t, "P_ren"])
        P_ren_norm = P_ren / (self.threshold + 1e-9)

        E_forecast_1h = float(self.df.loc[t, "forecast_E_ren_1h"])
        E_forecast_6h = float(self.df.loc[t, "forecast_E_ren_6h"])

        E_forecast_1h_norm = E_forecast_1h / (self.capacity + 1e-9)
        E_forecast_6h_norm = E_forecast_6h / (6.0 * self.capacity + 1e-9)

        co2_forecast_1h = float(self.df.loc[t, "forecast_co2_intensity_1h"])
        co2_forecast_6h = float(self.df.loc[t, "forecast_co2_intensity_6h"])

        co2_forecast_1h_norm = (co2_forecast_1h - co2_min) / (co2_max - co2_min + 1e-9)
        co2_forecast_6h_norm = (co2_forecast_6h - co2_min) / (co2_max - co2_min + 1e-9)

        obs = np.array([
            P_ratio,
            P_peak,
            battery_norm,
            time_left,
            price_base_norm,
            co2_int_norm,
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

        if self.episode_idx > 0:
            os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)

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

            csv_path = os.path.join(".", OUTPUT_DIR, "results_sim3_optimize.csv")

            end_soc = self.battery.info()["SOC"]
            end_capacity = self.battery.info()["capacity_Wh"] / self.capacity

            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{self.episode_idx};"
                    f"{sum(self.cost_history):.4f};"
                    f"{sum(self.co2_history)/1000:.3f};"
                    f"{end_soc:.4f};"
                    f"{end_capacity:.4f};"
                    f"{sum(self.battery_use_history):.3f};"
                    f"{sum(self.gen_use_history):.3f}\n"
                )

        self.episode_idx += 1

        self.battery_history = [self.capacity / 2]
        self.time_history = [self.df.loc[0, "time"]]
        self.cost_history = [0.0]
        self.co2_history = [0.0]
        self.gen_use_history = [0.0]
        self.battery_use_history = [0.0]
        self.curtailment_history = [0.0]

        self.green_cost_history = [0.0]
        self.green_co2_history = [0.0]
        self.batt_cost_history = [0.0]
        self.batt_co2_history = [0.0]
        self.grid_cost_history = [0.0]
        self.grid_co2_history = [0.0]
        self.gen_cost_history = [0.0]
        self.gen_co2_history = [0.0]

        self.t = 0

        self.battery = Battery(
            capacity_wh=self.capacity,
            initial_charge_wh=self.capacity / 2,
            max_charge_rate_w=self.max_charge_rate,
            max_discharge_rate_w=self.max_discharge_rate,
        )

        self.generator = Generator(
            max_power_w=500000,
            min_power_w=50000,
            efficiency=0.4,
            fuel_cost_per_wh=0.00025,
            co2_g_per_kwh=450
        )

        return self._get_obs(), {}

    def step(self, action):
        # =========================
        # 0. ACTIONS
        # =========================
        idx_discharge, idx_gen = action

        a_discharge = float(self.action_levels[idx_discharge])
        a_gen = float(self.action_levels[idx_gen])

        t = self.t
        dt = float(self.df.loc[t, "dt_hours"])

        if dt <= 0:
            self.t += 1
            terminated = self.t >= self.N
            if terminated:
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            else:
                obs = self._get_obs()
            return obs, 0.0, terminated, False, {}

        # =========================
        # 1. INPUT DATA
        # =========================
        P_load = float(self.df.loc[t, "power"])
        P_wind = float(self.df.loc[t, "P_wind"])
        P_solar = float(self.df.loc[t, "P_solar"])
        P_ren = float(self.df.loc[t, "P_ren"])

        price_base = float(self.df.loc[t, "price_base"])
        price_high = float(self.df.loc[t, "price_high"])
        co2_intensity = float(self.df.loc[t, "co2_intensity"])

        # =========================
        # 2. GREEN COST / CO2
        # =========================
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

        # =========================
        # 3. ENERGY BALANCE
        # =========================
        P_from_ren = min(P_load, P_ren)
        P_surplus = max(P_ren - P_load, 0.0)
        P_deficit = max(P_load - P_ren, 0.0)

        E_surplus = P_surplus * dt
        E_deficit = P_deficit * dt

        # =========================
        # 4. BATTERY CHARGE (from surplus)
        # =========================
        E_charged = 0.0
        E_curtail = E_surplus

        if E_surplus > 0:
            E_charged = self.battery.charge(P_surplus, dt)
            E_curtail = max(E_surplus - E_charged, 0.0)

        # =========================
        # 5. BATTERY DISCHARGE
        # =========================
        E_from_batt = 0.0

        if E_deficit > 0 and a_discharge > 0:
            request = a_discharge * E_deficit
            E_from_batt = self.battery.discharge(request / dt, dt)
            E_from_batt = min(E_from_batt, E_deficit)

        E_remaining_after_batt = max(E_deficit - E_from_batt, 0.0)

        # =========================
        # 6. GENERATOR
        # =========================
        E_from_gen = 0.0

        if E_remaining_after_batt > 0 and a_gen > 0:
            request_gen = a_gen * E_remaining_after_batt
            P_gen = request_gen / dt
            E_from_gen = self.generator.dispatch(P_gen, dt)
            E_from_gen = min(E_from_gen, E_remaining_after_batt)

        E_remaining_after_gen = max(E_remaining_after_batt - E_from_gen, 0.0)

        # =========================
        # 7. GRID
        # =========================
        E_grid = E_remaining_after_gen
        P_grid = E_grid / dt if dt > 0 else 0.0

        E_base = min(P_grid, self.threshold) * dt
        E_peak = max(P_grid - self.threshold, 0.0) * dt

        # =========================
        # 8. COSTS
        # =========================
        cost_grid = E_base * price_base + E_peak * price_high
        cost_gen = self.generator.get_cost(E_from_gen)

        E_batt_throughput = E_charged + E_from_batt
        batt_cost = self.battery.step_cost(dt, E_batt_throughput)

        cost = green_cost + batt_cost + cost_grid + cost_gen

        # =========================
        # 9. CO2
        # =========================
        co2_grid = (E_base + E_peak) / 1000.0 * co2_intensity
        co2_gen = self.generator.get_co2(E_from_gen / 1000.0)
        batt_co2_g = self.battery.step_co2_g(dt, E_batt_throughput)

        co2 = green_co2_g + batt_co2_g + co2_grid + co2_gen

        # print("cost = ", cost)
        # print("co2 ,", co2)

        # =========================
        # 10. REWARD
        # =========================
        reward = (
            - 1 * cost
            - 0.001 * co2
            # - 0.001 * (self.battery.throughput_wh / 1000.0)
            # - 0.005 * (E_curtail / 1000.0)
        )

        # =========================
        # 11. TERMINAL BONUS / PENALTY
        # mantengo la logica tua
        # =========================
        if self.t == self.N - 1:
            reward = (
                - 0.001 * sum(self.co2_history)
                -  sum(self.cost_history)
            )

        # =========================
        # 12. APPLY STATE UPDATES
        # =========================
        self.battery_history.append(self.battery.energy)
        self.gen_use_history.append(E_from_gen)
        self.battery_use_history.append(E_from_batt)
        self.cost_history.append(cost)
        self.co2_history.append(co2)
        self.curtailment_history.append(E_curtail)
        self.time_history.append(self.df.loc[t, "time"])

        self.green_cost_history.append(green_cost)
        self.green_co2_history.append(green_co2_g)
        self.batt_cost_history.append(batt_cost)
        self.batt_co2_history.append(batt_co2_g)
        self.grid_cost_history.append(cost_grid)
        self.grid_co2_history.append(co2_grid)
        self.gen_cost_history.append(cost_gen)
        self.gen_co2_history.append(co2_gen)

        self.battery.step(dt)
        self.t += 1

        terminated = self.t >= self.N

        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_obs()

        return obs, float(reward), terminated, False, {}


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

        if "results_sim3_optimize.csv" in files and not results_trovato:
            results_trovato = True
            path_results = os.path.join(root, "results_sim3_optimize.csv")
            print(f"Trovato file: {path_results}")

    intestazioni = "episodio;total_cost;total_co2;SOC_finale;capacity_ratio;battery_use_history;gen_use_history\n"

    if results_trovato:
        with open(path_results, "w", encoding="utf-8") as f:
            f.write(intestazioni)
        print(f"File {OUTPUT_DIR}/results_sim3_optimize.csv svuotato e intestazioni riscritte.")
    else:
        path_results = os.path.join(base_path, OUTPUT_DIR, "results_sim3_optimize.csv")
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


if __name__ == "__main__":

    THRESHOLD = 400000
    BATTERY_CAPACITY = 3200000
    P_MAX_RENS = (400000, 450000)

    pulizia_progetto(".")
    print("Start to training.... results in results_sim3_optimize.csv")

    # -------------------------------------------------------
    # RUN TRAINING
    # -------------------------------------------------------
    df = pd.read_csv("csvs/cluster_power_only_nodes_30days.csv")
    # df = pd.read_csv("cluster_power_only_nodes.csv")

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["dt_hours"] = df["time"].diff().dt.total_seconds() / 3600
    df["dt_hours"] = df["dt_hours"].fillna(0)

    # INSERISCO PREZZI
    price_model = PriceModel()
    df["price_base"], df["price_high"] = price_model.prices_from_df(df)

    # INSERISCO VALORI RINNOVABILI
    rm = RenewableModels(seed=42)
    df["P_wind"] = rm.wind_from_openmeteo(df, P_MAX_RENS[0])
    df["P_solar"] = rm.solar_from_openmeteo(df, P_MAX_RENS[1])
    df["P_ren"] = df["P_solar"] + df["P_wind"]

    # INSERISCO VALORI CO2
    cm = CarbonIntensityModels(csv_file="csvs/carbon_intensity_IT-NORTH-2020.csv")
    df["co2_intensity"] = cm.co2_from_csv(df)

    # -------------------------------------------------------
    # FORECASTS
    # uso energia futura, non somma di potenze
    # -------------------------------------------------------
    valid_dt = df.loc[df["dt_hours"] > 0, "dt_hours"]
    dt_hours_nominal = float(valid_dt.iloc[0]) if len(valid_dt) > 0 else (20.0 / 3600.0)
    steps_per_hour = max(int(round(1.0 / dt_hours_nominal)), 1)

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
        .fillna(0.0)
    )

    df["forecast_co2_intensity_6h"] = (
        df["co2_intensity"]
        .rolling(window=steps_per_hour * 6, min_periods=1)
        .mean()
        .shift(-steps_per_hour * 6)
        .fillna(0.0)
    )

    env = HPCBatteryEnv(
        df,
        threshold=THRESHOLD,
        battery_capacity=BATTERY_CAPACITY,
        max_charge_rate=BATTERY_CAPACITY,
        max_discharge_rate=BATTERY_CAPACITY
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