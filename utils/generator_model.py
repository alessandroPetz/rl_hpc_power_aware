class Generator:
    def __init__(
        self,
        max_power_w,
        min_power_w=0,
        efficiency=0.35,
        fuel_cost_per_wh=0.00025,   # €/Wh termico
        co2_g_per_kwh=0.45,          # g/KWh elettrico
    ):
        self.max_power_w = max_power_w
        self.min_power_w = min_power_w
        self.efficiency = efficiency
        self.fuel_cost_per_wh = fuel_cost_per_wh
        self.co2_g_per_Kwh = co2_g_per_kwh

        self.current_power = 0

    def dispatch(self, requested_power_w, dt_hours):
        """
        Decide quanta potenza erogare (W) e restituisce energia (Wh)
        """
        P = min(requested_power_w, self.max_power_w)

        # opzionale: minimo tecnico
        if P > 0:
            P = max(P, self.min_power_w)

        # opzionale: ramp rate
        if self.ramp_rate is not None:
            max_delta = self.ramp_rate * dt_hours
            P = min(P, self.current_power + max_delta)

        self.current_power = P

        E_out = P * dt_hours  # Wh

        return E_out

    def get_cost(self, energy_wh):
        """
        costo in €
        """
        # energia elettrica -> energia termica
        energy_kwh = energy_wh / 1000
        fuel_energy = energy_kwh / self.efficiency
        fuel_cost_per_Kwh = self.fuel_cost_per_wh *1000
        return fuel_energy * fuel_cost_per_Kwh

    def get_co2(self, energy_wh):
        return energy_wh * self.co2_g_per_Kwh 

    def step(self, dt_hours):
        pass