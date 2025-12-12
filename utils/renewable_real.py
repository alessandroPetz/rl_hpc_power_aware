import os
import requests
import pandas as pd
import numpy as np


class RenewableModels:

    def __init__(self, seed=42, cache_file="csvs/openmeteo_cache_bologna.csv"):
        self.rng = np.random.default_rng(seed)
        self.cache_file = cache_file

    def _load_openmeteo(self, df):
        """Carica dati meteo da cache o API, interpolando sui timestamp di df"""
        # 1️⃣ Usa cache se esiste
        if os.path.exists(self.cache_file):
            meteo = pd.read_csv(self.cache_file, parse_dates=["time"])
            meteo = meteo.set_index("time")
        else:
            # 2️⃣ Scarica dati OpenMeteo
            lat, lon = 44.4938, 11.3388  # Bologna
            start = df["time"].min().strftime("%Y-%m-%d")
            end   = df["time"].max().strftime("%Y-%m-%d")

            url = (
                "https://archive-api.open-meteo.com/v1/archive?"
                f"latitude={lat}&longitude={lon}"
                f"&start_date={start}&end_date={end}"
                "&hourly=wind_speed_10m,shortwave_radiation"
                "&timezone=Europe/Rome"
            )

            j = requests.get(url, timeout=20).json()

            if "hourly" not in j:
                raise RuntimeError("OpenMeteo JSON missing 'hourly'")

            meteo = pd.DataFrame({
                "time": pd.to_datetime(j["hourly"]["time"]),
                "wind_speed": j["hourly"]["wind_speed_10m"],
                "radiation": j["hourly"]["shortwave_radiation"],
            }).set_index("time")

            meteo.to_csv(self.cache_file)

        # 3️⃣ Allinea timezone
        df_index = pd.DatetimeIndex(df["time"])
        if meteo.index.tz is None:
            # Se meteo è naive, impostiamo UTC o la stessa di df
            meteo = meteo.tz_localize(df_index.tz)
        elif df_index.tz is None:
            df_index = df_index.tz_localize(meteo.index.tz)

        # 4️⃣ Reindex + interpolate
        meteo_interp = meteo.reindex(meteo.index.union(df_index)).sort_index()
        meteo_interp = meteo_interp.interpolate(method="time").reindex(df_index)

        return meteo_interp     


    # ----------------------------------------------------------------------
    # 🌬 MODELLO VENTO
    # ----------------------------------------------------------------------
    def wind_from_openmeteo(self, df, p_max=400_000):
        meteo = self._load_openmeteo(df)
        v = meteo["wind_speed"].values

        # curva semplificata turbina
        v_cut_in = 3
        v_rated  = 12
        v_norm = np.clip((v - v_cut_in) / (v_rated - v_cut_in + 1e-9), 0, 1)
        P_wind = p_max * (v_norm ** 3)
        return np.maximum(P_wind, 0)

    # ----------------------------------------------------------------------
    # 🌞 MODELLO SOLARE
    # ----------------------------------------------------------------------
    def solar_from_openmeteo(self, df, p_max=450_000):
        meteo = self._load_openmeteo(df)
        G = meteo["radiation"].values
        G_max = max(G.max(), 1e-9)
        P_solar = p_max * (G / G_max)
        return np.maximum(P_solar, 0)
