import numpy as np
import gymnasium as gym
from gymnasium import spaces

class HPCBatteryEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self, 
        df, 
        threshold=400000,
        battery_capacity=1000,
        max_charge_rate=200,       # Wh per ora
        max_discharge_rate=200,    # Wh per ora
        cost_low=0.0001,
        cost_high=0.0003
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.N = len(self.df)

        self.threshold = threshold
        self.capacity = battery_capacity
        
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate

        self.cost_low = cost_low
        self.cost_high = cost_high

        # -------------------------------
        # Observation: power, battery, time
        # -------------------------------
        self.observation_space = spaces.Box(
            low=np.array([0., 0., 0.], dtype=np.float32),
            high=np.array([1., 1., 1.], dtype=np.float32)
        )

        # -------------------------------
        # Action = [charge_rate, discharge_rate]
        # -------------------------------
        self.action_space = spaces.Box(
            low=np.array([0., 0.], dtype=np.float32),
            high=np.array([1., 1.], dtype=np.float32)
        )

        self.reset()

    def _get_obs(self):
        t = min(self.t, self.N - 1)
        P = float(self.df.loc[t, "power"])
        return np.array([
            min(P / self.threshold, 1.0),
            self.battery / self.capacity,
            t / (self.N - 1)
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.battery = self.capacity * 0.5
        return self._get_obs(), {}

    def step(self, action):
        if self.t >= self.N - 1:
            return self._get_obs(), 0.0, True, False, {}

        t = min(self.t, self.N - 1)

        # ------------------------------------------------
        # Dati dal dataset
        # ------------------------------------------------
        P = float(self.df.loc[t, "power"])       # W
        dt = float(self.df.loc[t, "dt_hours"])   # ore

        energy_need = P * dt
        E_base = min(P, self.threshold) * dt
        E_peak = max(P - self.threshold, 0) * dt

        # ------------------------------------------------
        # Action parsing
        # ------------------------------------------------
        charge_rate = float(action[0])     # ∈ [0,1]
        discharge_rate = float(action[1])  # ∈ [0,1]

        # Richieste massime
        E_charge_req = charge_rate * self.max_charge_rate * dt
        E_discharge_req = discharge_rate * self.max_discharge_rate * dt

        # ------------------------------------------------
        # VINCOLI FISICI
        # ------------------------------------------------
        # Non carico più della capacità residua
        E_charge = min(E_charge_req, self.capacity - self.battery)

        # Non scarico più dell’energia in batteria
        E_discharge = min(E_discharge_req, self.battery)

        # ------------------------------------------------
        # COPERTURA DEL FABISOGNO
        # ------------------------------------------------
        # 1) Prima uso la batteria per coprire E_peak (se richiesto)
        discharge_used_for_peak = min(E_peak, E_discharge)
        E_peak_remaining = E_peak - discharge_used_for_peak

        # 2) Eventuale energia ancora richiesta dalla rete
        #    (include base + parte di picco non coperta)
        E_from_grid = E_base + E_peak_remaining

        # ------------------------------------------------
        # COSTO ENERGETICO
        # ------------------------------------------------
        if P > self.threshold:
            # sopra soglia → tutta l’energia pagata high tranne la parte coperta da batteria
            cost = (E_base * self.cost_low) + (E_peak_remaining * self.cost_high)
        else:
            # sotto soglia → tutto low
            cost = energy_need * self.cost_low

        # Caricare costa sempre low
        cost += E_charge * self.cost_low

        # ------------------------------------------------
        # AGGIORNA BATTERIA
        # ------------------------------------------------
        self.battery += E_charge
        self.battery -= discharge_used_for_peak
        # (lo scarico extra non viene usato se non serve al cluster)

        # Clamp sicurezza
        self.battery = np.clip(self.battery, 0, self.capacity)

        # Reward
        reward = -cost

        self.t += 1
        terminated = self.t >= self.N - 1

        return self._get_obs(), reward, terminated, False, {}
