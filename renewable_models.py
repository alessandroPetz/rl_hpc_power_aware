import numpy as np


class RenewableModels:
    """
    Raccolta di modelli semplici per generazione rinnovabile
    (vento + solare) con seed riproducibile.
    """

    def __init__(self, seed=42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # -----------------------------------------------------
    # MODELLO VENTO 0: uniforme semplice
    # -----------------------------------------------------
    def wind_uniform(self, df, p_max=150_000):
        return self.rng.uniform(0, p_max, size=len(df))

    # -----------------------------------------------------
    # MODELLO VENTO 1: variabilità forte (lognormale)
    # -----------------------------------------------------
    def wind_variable(self, df, p_max=150_000):
        P_wind_raw = self.rng.lognormal(mean=10.5, sigma=0.6, size=len(df))
        return np.clip(P_wind_raw, 0, p_max)

    # -----------------------------------------------------
    # MODELLO VENTO 2: ciclo giornaliero + autocorrelato
    # -----------------------------------------------------
    def wind_stochastic_daily(
        self,
        df,
        p0=60_000,
        noise_std=8_000,
        daily_amp=20_000,
        alpha=0.97
    ):
        hours = df["time"].dt.hour.values
        daily_cycle = 0.5 + 0.5 * np.sin(2*np.pi*(hours - 6) / 24)

        P_wind = np.zeros(len(df))
        P_wind[0] = p0

        for i in range(1, len(df)):
            noise = self.rng.normal(0, noise_std)
            P_wind[i] = alpha * P_wind[i-1] + daily_amp * daily_cycle[i] + noise

        return np.clip(P_wind, 0, 150_000)

    # -----------------------------------------------------
    # MODELLO VENTO 3: puro autocorrelato (raffiche)
    # -----------------------------------------------------
    def wind_stochastic(self, df, p0=50_000, noise_std=10_000, alpha=0.98):
        P_wind = np.zeros(len(df))
        P_wind[0] = p0

        for i in range(1, len(df)):
            noise = self.rng.normal(0, noise_std)
            P_wind[i] = alpha * P_wind[i-1] + noise

        return np.clip(P_wind, 0, 150_000)

    # -----------------------------------------------------
    # MODELLO SOLARE semplice
    # -----------------------------------------------------
    def solar_simple(self, df, p_max=200_000):
        solar_raw = p_max * np.sin(2*np.pi*df["time"].dt.hour / 24)
        return np.maximum(0, solar_raw)

    # -----------------------------------------------------
    # MODELLO SOLARE con nuvolosità
    # -----------------------------------------------------
    def solar_cloudy(
        self,
        df,
        p_max=200_000,
        cloud_mu=1.0,
        cloud_sigma=0.2
    ):
        cloud_factor = np.clip(
            self.rng.normal(cloud_mu, cloud_sigma, size=len(df)),
            0.3, 1.2
        )

        solar_raw = p_max * np.sin(2 * np.pi * df["time"].dt.hour / 24)
        P_solar = np.maximum(0, solar_raw * cloud_factor)

        return P_solar
    
    def solar_cloudy2(
        self,
        df,
        p_max=200_000,
        clear_day_prob=0.30,      # 30% giorni limpidi
        cloudy_day_prob=0.15,     # 15% giorni pessimi
        variability=0.15,         # variazione intra-giorno
        cloud_min=0.4,            # notte esclusa
        cloud_max=1.3
    ):
        """
        Modello solare più realistico:
        - maggioranza giorni 'normali'
        - 30% limpidi (alta produzione)
        - 15% molto nuvolosi (bassa produzione)
        - variazione intra-giornaliera leggera
        - nessun drift verso valori costantemente bassi
        """

        N = len(df)
        hours = df["time"].dt.hour + df["time"].dt.minute/60

        # ---------------------------------------------------------
        # 1) Forma base del sole
        # ---------------------------------------------------------
        solar_raw = np.maximum(0, p_max * np.sin(2 * np.pi * hours / 24))

        # ---------------------------------------------------------
        # 2) Classificazione giornaliera (limpido / normale / nuvoloso)
        # ---------------------------------------------------------
        df["date_only"] = df["time"].dt.date
        unique_days = df["date_only"].unique()
        day_factor = {}

        for d in unique_days:
            r = self.rng.random()

            if r < clear_day_prob:
                # giorno molto limpido → tanta produzione
                day_factor[d] = self.rng.uniform(1.0, 1.3)

            elif r < clear_day_prob + cloudy_day_prob:
                # giorno molto nuvoloso
                day_factor[d] = self.rng.uniform(0.2, 0.5)

            else:
                # giorno normale
                day_factor[d] = self.rng.uniform(0.7, 1.0)

        day_mult = np.array([day_factor[d] for d in df["date_only"]])

        # ---------------------------------------------------------
        # 3) Variazione intra-giornaliera smooth
        # ---------------------------------------------------------
        base_cloud = self.rng.normal(1.0, variability, size=N)

        # smoothed using moving window (evita oscillazioni assurde)
        window = 20  # dipende dal passo temporale; 20 ≈ ~6–10 minuti
        kernel = np.ones(window) / window
        smooth_cloud = np.convolve(base_cloud, kernel, mode="same")

        smooth_cloud = np.clip(smooth_cloud, cloud_min, cloud_max)

        # ---------------------------------------------------------
        # 4) Produzione finale
        # ---------------------------------------------------------
        P_solar = solar_raw * day_mult * smooth_cloud

        return P_solar

