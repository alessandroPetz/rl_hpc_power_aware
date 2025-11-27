import numpy as np
import pandas as pd
from typing import Optional, Callable, Dict, Any

class Battery:
    """
    Simple pack-level battery model with SOC, SOH and two degr. options:
      - empirical (fast, per-timestep): loss proportional to energy throughput and C-rate term
      - ML surrogate: call `apply_ml_predictor` with characterization traces (v_series, T_series)
    
    Units:
      - capacity_nominal_kwh: kWh (nameplate at SoH==1.0)
      - soc_kwh stored as absolute kWh
      - action_kw: positive = charge (grid -> battery), negative = discharge (battery -> grid)
      - dt_seconds: seconds duration of timestep
    """
    def __init__(
        self,
        capacity_nominal_kwh: float = 100.0,
        soc_init: float = 0.5,
        soh_init: float = 1.0,
        p_charge_max_kw: float = 50.0,
        p_discharge_max_kw: float = 50.0,
        roundtrip_efficiency: float = 0.95,
        empirical_alpha: float = 1e-4,   # kWh^-1 loss per kWh throughput (calibrate with Oxford)
        empirical_beta: float  = 5e-6,   # C-rate sensitivity
        soh_min: float = 0.5,
    ):
        # params
        self.capacity_nominal_kwh = float(capacity_nominal_kwh)
        self.p_charge_max_kw = float(p_charge_max_kw)
        self.p_discharge_max_kw = float(p_discharge_max_kw)
        self.eta_rt = float(roundtrip_efficiency)
        self.emp_alpha = float(empirical_alpha)
        self.emp_beta  = float(empirical_beta)
        self.soh_min = float(soh_min)

        # state
        self.soh = float(soh_init)                  # 0..1
        self.cap_kwh = self.capacity_nominal_kwh * self.soh
        self.soc_kwh = float(soc_init) * self.capacity_nominal_kwh  # stored absolute kWh (relative to nameplate)
        
        # accumulators
        self.lifetime_energy_throughput_kwh = 0.0
        self.history = []  # list of dicts (time step records)

    # ---------------------------
    # Utility methods
    # ---------------------------
    def get_state(self) -> Dict[str, Any]:
        return {
            'soc_kwh': self.soc_kwh,
            'soh': self.soh,
            'cap_kwh': self.cap_kwh
        }

    def reset(self, soc_init: float = 0.5, soh_init: float = 1.0):
        """Reset state and history"""
        self.soh = float(soh_init)
        self.cap_kwh = self.capacity_nominal_kwh * self.soh
        self.soc_kwh = float(soc_init) * self.capacity_nominal_kwh
        self.lifetime_energy_throughput_kwh = 0.0
        self.history = []

    @staticmethod
    def power_to_c_rate(power_kw: float, capacity_kwh: float) -> float:
        """Return instantaneous C-rate in 1/h units (abs)"""
        if capacity_kwh <= 0:
            return 0.0
        return abs(power_kw) / capacity_kwh

    # ---------------------------
    # Degradation models
    # ---------------------------
    def _degrade_empirical_step(self, energy_moved_kwh: float, avg_c_rate: float) -> float:
        """
        Simple empirical degradation applied per timestep.
        energy_moved_kwh: absolute energy moved in this step (kWh)
        avg_c_rate: average C-rate (1/h) in this step
        returns new soh
        """
        loss = self.emp_alpha * energy_moved_kwh + self.emp_beta * (avg_c_rate ** 1.2)
        new_soh = max(self.soh_min, self.soh - loss)
        return new_soh

    def apply_ml_predictor(self, voltage_series: pd.Series, temperature_series: pd.Series, model) -> float:
        """
        Use your trained ML surrogate to predict SoH from a characterization trace.
        Assumption: model.predict expects a nested DataFrame with one row containing the series columns
                    or a callable that accepts (voltage_series, temperature_series) directly.
        Returns predicted soh (0..1) and sets self.soh accordingly.
        """
        # Try common sktime-like input: DataFrame with series columns
        try:
            X = pd.DataFrame({'voltage': [pd.Series(voltage_series)], 'temperature': [pd.Series(temperature_series)]})
            soh_pred = model.predict(X)
            # sklearn-like or sktime classifiers might return array-like; take first element
            if hasattr(soh_pred, '__len__'):
                soh_val = float(soh_pred[0])
            else:
                soh_val = float(soh_pred)
            # If model returns percentage (0-100), normalize:
            if soh_val > 1.5:
                soh_val = soh_val / 100.0
        except Exception:
            # fallback: assume model is a callable f(v,t) -> soh
            soh_val = float(model(voltage_series, temperature_series))

        # clamp and apply
        soh_val = max(self.soh_min, min(1.0, soh_val))
        self.soh = soh_val
        self.cap_kwh = self.capacity_nominal_kwh * self.soh
        # ensure SOC does not exceed new capacity
        self.soc_kwh = min(self.soc_kwh, self.cap_kwh)
        return self.soh

    # ---------------------------
    # Step function: main simulator update
    # ---------------------------
    def step(self, action_kw: float, dt_seconds: float, timestamp=None, demand_kw: Optional[float]=None,
             degrade_method: str = "empirical") -> Dict[str, Any]:
        """
        Perform a single timestep update.
        action_kw: positive -> charging (kW), negative -> discharging (kW)
        dt_seconds: timestep seconds
        demand_kw: optional cluster demand at this time (for logging)
        degrade_method: "empirical" or "none" (ml-based must be applied by apply_ml_predictor)
        Returns the record dict appended to self.history.
        """
        # clip action to battery limits
        action_kw = float(action_kw)
        action_kw = np.clip(action_kw, -self.p_discharge_max_kw, self.p_charge_max_kw)

        # energy moved this step (absolute kWh)
        energy_moved_kwh = abs(action_kw) * (dt_seconds / 3600.0)

        # apply efficiency: split equally on charge/discharge sides as sqrt(eta)
        # if action positive (charging) we add energy, if negative (discharging) we remove energy
        if action_kw >= 0:
            # charging: energy grid -> battery
            energy_in = action_kw * (dt_seconds / 3600.0) * (self.eta_rt ** 0.5)
            self.soc_kwh += energy_in
        else:
            # discharging: battery -> grid
            energy_out = (-action_kw) * (dt_seconds / 3600.0) * (self.eta_rt ** 0.5)
            self.soc_kwh -= energy_out

        # ensure SOC bounds [0, cap_kwh]
        self.soc_kwh = float(np.clip(self.soc_kwh, 0.0, self.cap_kwh))

        # instantaneous C-rate relative to current capacity
        inst_c_rate = self.power_to_c_rate(action_kw, self.cap_kwh)  # 1/h

        # degradation update (empirical per-step)
        if degrade_method == "empirical":
            new_soh = self._degrade_empirical_step(energy_moved_kwh, inst_c_rate)
            # update lifetime counters
            self.lifetime_energy_throughput_kwh += energy_moved_kwh
            # apply soh update
            self.soh = new_soh
            self.cap_kwh = self.capacity_nominal_kwh * self.soh
            # ensure soc <= cap_kwh
            self.soc_kwh = min(self.soc_kwh, self.cap_kwh)

        # record timestep
        record = {
            'time': timestamp,
            'demand_kw': demand_kw,
            'action_kw': action_kw,
            'soc_kwh': self.soc_kwh,
            'soh': self.soh,
            'cap_kwh': self.cap_kwh,
            'energy_moved_kwh': energy_moved_kwh,
            'inst_c_rate_1ph': inst_c_rate,
            'lifetime_throughput_kwh': self.lifetime_energy_throughput_kwh
        }
        self.history.append(record)
        return record

    # ---------------------------
    # Helpers to export history and quick plotting
    # ---------------------------
    def history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history).set_index('time')

    def plot_history(self):
        df = self.history_df()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3,1, figsize=(10,8), sharex=True)
        ax[0].plot(df.index, df['demand_kw'], label='demand_kw'); ax[0].plot(df.index, df['action_kw'], label='action_kw'); ax[0].legend()
        ax[1].plot(df.index, df['soc_kwh'], label='SOC (kWh)'); ax[1].legend()
        ax[2].plot(df.index, df['soh'], label='SOH'); ax[2].legend()
        plt.tight_layout()
        plt.show()


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # minimal example: run battery on a small synthetic trace
    import pandas as pd
    import numpy as np

    # create a fake power trace (kW) 1-minute resolution for 4 hours
    rng = pd.date_range("2025-01-01 00:00", periods=4*60, freq="1min")
    # synthetic cluster demand (kW): base load + random peaks
    base = 150.0
    demand = pd.Series(base + 30*np.sin(np.linspace(0,10,len(rng))) + 50*(np.random.rand(len(rng))>0.98), index=rng)

    # simple greedy controller example
    def greedy_controller(state: Dict[str,Any], t, demand_kw):
        TH = 180.0
        Pmax = 50.0
        soc_kwh = state['soc_kwh']
        cap_kwh = state['cap_kwh']
        # discharge to shave peaks
        if demand_kw > TH and soc_kwh > 0.05*cap_kwh:
            return -min(Pmax, demand_kw - TH)
        # charge when low demand and capacity available
        if demand_kw < TH*0.8 and soc_kwh < 0.95*cap_kwh:
            return min(Pmax, TH - demand_kw)
        return 0.0

    # instantiate battery
    batt = Battery(capacity_nominal_kwh=100.0, soc_init=0.5, soh_init=1.0,
                   p_charge_max_kw=50.0, p_discharge_max_kw=50.0, roundtrip_efficiency=0.95)

    dt = 60000  # 1 minute steps (seconds)


    for t, d in demand.items():
        state = {'soc_kwh': batt.soc_kwh, 'soh': batt.soh, 'cap_kwh': batt.cap_kwh}
        action = greedy_controller(state, t, d)
        batt.step(action, dt_seconds=dt, timestamp=t, demand_kw=d, degrade_method="empirical")

    # results
    hist = batt.history_df()
    print(hist.tail())
    batt.plot_history()
