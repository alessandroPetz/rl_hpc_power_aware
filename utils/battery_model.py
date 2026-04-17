import numpy as np


class Battery:
    """
    Modello di batteria semplificato ma realistico.
    Unità:
        - energia: Wh
        - potenza: W
        - tempo: h

    Include:
        - efficienza di carica/scarica
        - vincoli SOC [soc_min, soc_max]
        - riduzione potenza di carica sopra 80% SOC
        - self-discharge
        - degrado proporzionale al throughput
        - costo fisso ammortizzato
        - CO2 fissa ammortizzata
        - costo variabile di usura per throughput
        - CO2 variabile di usura per throughput
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
        REAL_degradation_per_kwh=5e-4,
        SCALE_degradation_per_kwh=50000,

        # --- costo/CO2 batteria ---
        battery_capex_eur=480000.0,
        battery_lifetime_hours=10 * 365 * 24,
        battery_embodied_co2_kg=240000.0,

        wear_cost_eur_per_kwh=0.02,
        wear_co2_g_per_kwh=20.0,
    ):
        # Stato fisico
        self.capacity_nominal = float(capacity_wh)
        self.capacity = float(capacity_wh)
        self.energy = float(np.clip(initial_charge_wh, 0, capacity_wh))

        self.max_charge_rate = float(max_charge_rate_w)
        self.max_discharge_rate = float(max_discharge_rate_w)

        self.eta_charge = float(eta_charge)
        self.eta_discharge = float(eta_discharge)

        self.soc_min = float(soc_min)
        self.soc_max = float(soc_max)

        self.self_discharge_per_hour = float(self_discharge_per_hour)

        # degrado fisico
        self.degradation_per_kwh = float(REAL_degradation_per_kwh * SCALE_degradation_per_kwh)
        self.throughput_wh = 0.0

        # costo / CO2 "ownership"
        self.battery_capex_eur = float(battery_capex_eur)
        self.battery_lifetime_hours = float(battery_lifetime_hours)
        self.battery_embodied_co2_kg = float(battery_embodied_co2_kg)

        # costo / CO2 "usage"
        self.wear_cost_eur_per_kwh = float(wear_cost_eur_per_kwh)
        self.wear_co2_g_per_kwh = float(wear_co2_g_per_kwh)

    # --------------------------------------------------
    # Proprietà utili
    # --------------------------------------------------

    @property
    def soc(self):
        return self.energy / self.capacity if self.capacity > 0 else 0.0

    @property
    def e_min(self):
        return self.capacity * self.soc_min

    @property
    def e_max(self):
        return self.capacity * self.soc_max

    @property
    def fixed_cost_eur_per_hour(self):
        if self.battery_lifetime_hours <= 0:
            return 0.0
        return self.battery_capex_eur / self.battery_lifetime_hours

    @property
    def fixed_co2_g_per_hour(self):
        if self.battery_lifetime_hours <= 0:
            return 0.0
        return (self.battery_embodied_co2_kg * 1000.0) / self.battery_lifetime_hours

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

        factor = max(0.0, 1.0 - (self.soc - 0.8) / 0.2)
        return self.max_charge_rate * factor

    # --------------------------------------------------
    # Azioni
    # --------------------------------------------------

    def charge(self, power_w, dt_hours):
        """
        Carica la batteria.

        Parametri:
            power_w: potenza richiesta dalla fonte verso batteria
            dt_hours: durata step

        Ritorna:
            energia assorbita dalla fonte (Wh)
        """
        if power_w <= 0 or dt_hours <= 0:
            return 0.0

        power_limit = min(power_w, self._charge_power_limit())
        e_available = power_limit * dt_hours

        room = self.e_max - self.energy
        if room <= 0:
            return 0.0

        # energia richiesta alla fonte, tenendo conto dell'efficienza
        e_charge_from_source = min(e_available, room / self.eta_charge)

        # energia realmente immagazzinata
        e_stored = e_charge_from_source * self.eta_charge
        self.energy += e_stored

        # degrado proporzionale all'energia movimentata internamente
        self._apply_degradation(e_charge_from_source)

        return e_charge_from_source

    def discharge(self, power_w, dt_hours):
        """
        Scarica la batteria.

        Parametri:
            power_w: potenza richiesta dal carico
            dt_hours: durata step

        Ritorna:
            energia fornita al carico (Wh)
        """
        if power_w <= 0 or dt_hours <= 0:
            return 0.0

        power_limit = min(power_w, self.max_discharge_rate)
        e_requested_to_load = power_limit * dt_hours

        usable = self.energy - self.e_min
        if usable <= 0:
            return 0.0

        # energia che devo togliere internamente per dare e_requested_to_load
        e_drawn_internal = min(e_requested_to_load / self.eta_discharge, usable)

        self.energy -= e_drawn_internal
        self._apply_degradation(e_drawn_internal)

        # energia effettivamente consegnata al carico
        e_to_load = e_drawn_internal * self.eta_discharge
        return e_to_load

    # --------------------------------------------------
    # Dinamiche lente
    # --------------------------------------------------

    def step(self, dt_hours):
        """Self-discharge."""
        if dt_hours > 0:
            self.energy *= (1.0 - self.self_discharge_per_hour * dt_hours)
            self.energy = max(self.energy, 0.0)

    def _apply_degradation(self, energy_wh):
        """
        Degrado proporzionale al throughput energetico.
        """
        self.throughput_wh += energy_wh

        loss_wh = (energy_wh / 1000.0) * self.degradation_per_kwh
        self.capacity = max(self.capacity - loss_wh, 0.7 * self.capacity_nominal)

        self.energy = min(self.energy, self.capacity)

    # --------------------------------------------------
    # Costo / CO2
    # --------------------------------------------------

    def fixed_cost(self, dt_hours):
        """
        Quota fissa ammortizzata del CAPEX per lo step.
        """
        if dt_hours <= 0:
            return 0.0
        return self.fixed_cost_eur_per_hour * dt_hours

    def fixed_co2_g(self, dt_hours):
        """
        Quota fissa ammortizzata della CO2 embodied per lo step.
        """
        if dt_hours <= 0:
            return 0.0
        return self.fixed_co2_g_per_hour * dt_hours

    def wear_cost(self, throughput_wh):
        """
        Costo variabile di usura in funzione del throughput.
        """
        if throughput_wh <= 0:
            return 0.0
        return (throughput_wh / 1000.0) * self.wear_cost_eur_per_kwh

    def wear_co2_g(self, throughput_wh):
        """
        CO2 variabile imputata all'uso della batteria in funzione del throughput.
        """
        if throughput_wh <= 0:
            return 0.0
        return (throughput_wh / 1000.0) * self.wear_co2_g_per_kwh

    def step_cost(self, dt_hours, throughput_wh):
        """
        Costo totale batteria nello step:
            quota fissa + usura variabile
        """
        return self.fixed_cost(dt_hours) + self.wear_cost(throughput_wh)

    def step_co2_g(self, dt_hours, throughput_wh):
        """
        CO2 totale batteria nello step:
            quota fissa + usura variabile
        """
        return self.fixed_co2_g(dt_hours) + self.wear_co2_g(throughput_wh)

    # --------------------------------------------------
    # Debug
    # --------------------------------------------------

    def info(self):
        return {
            "energy_Wh": self.energy,
            "capacity_Wh": self.capacity,
            "capacity_loss_Wh": self.capacity_nominal - self.capacity,
            "SOC": self.soc,
            "throughput_kWh": self.throughput_wh / 1000.0,
            "fixed_cost_eur_per_hour": self.fixed_cost_eur_per_hour,
            "fixed_co2_g_per_hour": self.fixed_co2_g_per_hour,
            "wear_cost_eur_per_kWh": self.wear_cost_eur_per_kwh,
            "wear_co2_g_per_kWh": self.wear_co2_g_per_kwh,
        }