import numpy as np


class Battery:
    """
    Modello di batteria semplificato ma realistico.
    Tutte le energie sono in Wh, le potenze in W, il tempo in ore.

    qui viene implementato:
        - Efficienza di carica e scarica (non è gratis)
            caricare non è gratis (perdite Joule)
            scaricare restituisce meno energia di quella nominale
        - SOC range 
            non 0-100 ma 15-95
        - Potenza dipendente dallo stato di carica
            sopra 80% meno potenza
        -  self-discharge
            0.05–0.2 % / giorno
        - degrado batteria
            riduco capacità in base a quanta energia muovo 



    """

    def __init__(
        self,
        capacity_wh,
        initial_charge_wh,
        max_charge_rate_w,
        max_discharge_rate_w,
        eta_charge=0.95,
        eta_discharge=0.95,
        soc_min=0.15,
        soc_max=0.95,
        self_discharge_per_hour=0.001 / 24,
        #degradation_per_kwh=2e-5,
        REAL_degradation_per_kwh = 5e-4,        # reale
        SCALE_degradation_per_kwh = 50000,        # 10 gg ≈ 500 gg
    
    ):
        self.capacity_nominal = capacity_wh
        self.capacity = capacity_wh

        self.energy = np.clip(initial_charge_wh, 0, capacity_wh)

        self.max_charge_rate = max_charge_rate_w
        self.max_discharge_rate = max_discharge_rate_w

        self.eta_charge = eta_charge
        self.eta_discharge = eta_discharge

        self.soc_min = soc_min
        self.soc_max = soc_max

        self.self_discharge_per_hour = self_discharge_per_hour
        self.degradation_per_kwh = REAL_degradation_per_kwh*SCALE_degradation_per_kwh

        self.throughput_wh = 0.0

    # --------------------------------------------------
    # Proprietà utili
    # --------------------------------------------------

    @property
    def soc(self):
        return self.energy / self.capacity if self.capacity > 0 else 0

    @property
    def e_min(self):
        return self.capacity * self.soc_min

    @property
    def e_max(self):
        return self.capacity * self.soc_max

    # --------------------------------------------------
    # Limiti dinamici
    # --------------------------------------------------

    def _charge_power_limit(self):
        """
        Riduzione progressiva della potenza di carica sopra l'80% di SOC
        (approssimazione fase CV).
        """
        if self.soc < 0.8:
            return self.max_charge_rate
        else:
            factor = max(0.0, 1 - (self.soc - 0.8) / 0.2)
            return self.max_charge_rate * factor

    # --------------------------------------------------
    # Azioni
    # --------------------------------------------------

    def charge(self, power_w, dt_hours):
        """
        Carica la batteria.
        Ritorna l'energia ASSORBITA dalla fonte (Wh).
        """
        if power_w <= 0 or dt_hours <= 0:
            return 0.0

        power_limit = min(power_w, self._charge_power_limit())
        e_available = power_limit * dt_hours

        room = self.e_max - self.energy
        if room <= 0:
            return 0.0

        e_charge = min(e_available, room / self.eta_charge)

        self.energy += e_charge * self.eta_charge
        self._apply_degradation(e_charge)

        return e_charge

    def discharge(self, power_w, dt_hours):
        """
        Scarica la batteria.
        Ritorna l'energia FORNITA al carico (Wh).
        """
        if power_w <= 0 or dt_hours <= 0:
            return 0.0

        power_limit = min(power_w, self.max_discharge_rate)
        e_requested = power_limit * dt_hours

        usable = self.energy - self.e_min
        if usable <= 0:
            return 0.0

        e_discharge = min(e_requested / self.eta_discharge, usable)

        self.energy -= e_discharge
        self._apply_degradation(e_discharge)

        return e_discharge * self.eta_discharge

    # --------------------------------------------------
    # Dinamiche lente
    # --------------------------------------------------

    def step(self, dt_hours):
        """Self-discharge"""
        if dt_hours > 0:
            self.energy *= (1 - self.self_discharge_per_hour * dt_hours)
            self.energy = max(self.energy, 0.0)

    def _apply_degradation(self, energy_wh):
        """
        Degrado proporzionale al throughput energetico.
        """
        self.throughput_wh += energy_wh
        loss = (energy_wh / 1000) * self.degradation_per_kwh
        self.capacity = max(self.capacity - loss, 0.7 * self.capacity_nominal)

        self.energy = min(self.energy, self.capacity)

    # --------------------------------------------------
    # Debug
    # --------------------------------------------------

    def info(self):
        return {
            "energy_Wh": self.energy,
            "capacity_Wh": self.capacity,
            "SOC": self.soc,
            "throughput_kWh": self.throughput_wh / 1000,
        }
