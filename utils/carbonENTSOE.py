import pandas as pd
import requests
from io import StringIO
from datetime import timedelta

class CarbonIntensityENTSOE:
    def __init__(self, api_key, country="IT"):
        self.api_key = api_key
        self.country = country

    # ---------------------------------------------------------------
    # 1️⃣ Scarico i dati ENTSO-E per l’intervallo necessario
    # ---------------------------------------------------------------
    def _download_entsoe(self, start, end):
        """
        Scarica il mix di produzione reale per tipo (Actual Generation per Production Type)
        per l'intervallo start-end, in UTC.
        """
        # ENTSO-E richiede formato: YYYYMMDDHHMM
        fmt = "%Y%m%d%H%M"
        periodStart = start.strftime(fmt)
        periodEnd   = end.strftime(fmt)

        url = (
            "https://transparency.entsoe.eu/api?"
            f"securityToken={self.api_key}&"
            "documentType=A75&"          # Actual Generation per Production Type
            "processType=A16&"           # Real-time
            f"outBiddingZone_Domain={self.country}&"
            f"periodStart={periodStart}&"
            f"periodEnd={periodEnd}"
        )

        r = requests.get(url)
        if r.status_code != 200:
            raise Exception(f"ENTSO-E error {r.status_code}: {r.text}")

        df = pd.read_xml(StringIO(r.text))
        if df is None or df.empty:
            raise Exception("ENTSO-E returned no data for this period.")
        
        return df

    # ---------------------------------------------------------------
    # 2️⃣ Costruzione DataFrame intensità CO₂ da ENTSO-E
    # ---------------------------------------------------------------
    def _build_intensity_df(self, df_raw):
        df = df_raw.copy()
        df = df[["start", "quantity", "productionType"]]
        df["start"] = pd.to_datetime(df["start"], utc=True)

        # pivot: righe = timestamp, colonne = fonte
        df = df.pivot_table(
            index="start",
            columns="productionType",
            values="quantity",
            aggfunc="sum"
        ).sort_index()

        # Fattori IPCC in gCO₂/kWh
        ef = {
            "Fossil Brown coal/Lignite": 1060,
            "Fossil Hard coal": 820,
            "Fossil Coal-derived gas": 1000,
            "Fossil Gas": 490,
            "Fossil Oil": 750,
            "Fossil Oil shale": 1000,
            "Peat": 1060,
        }

        df["emissions"] = 0
        for col, factor in ef.items():
            if col in df.columns:
                df["emissions"] += df[col] * factor

        df["total_gen"] = df.sum(axis=1)

        df["co2"] = df["emissions"] / df["total_gen"]

        return df[["co2"]]

    # ---------------------------------------------------------------
    # 3️⃣ Funzione principale simile alla tua co2_from_csv
    # ---------------------------------------------------------------
    def co2_from_entsoe(self, df_consumo):
        df = df_consumo.copy()
        df["time"] = pd.to_datetime(df["time"], utc=True)

        # trovo automaticamente l'intervallo necessario
        start = df["time"].min() - timedelta(hours=1)
        end   = df["time"].max() + timedelta(hours=1)

        # scarico solo i dati necessari
        raw = self._download_entsoe(start, end)

        # costruisco df intensità CO2
        self.co2_df = self._build_intensity_df(raw)

        # -------------------------------
        #  mapping anni (come nel tuo CSV)
        # -------------------------------
        csv_years = set(self.co2_df.index.year.unique())
        df_years = set(df["time"].dt.year.unique())
        missing_years = df_years - csv_years

        if missing_years:
            target_year = min(csv_years)
            print(f"⚠️ ENTSO-E non ha gli anni {missing_years}, uso {target_year}")
            df["time_mapped"] = df["time"].apply(lambda t: t.replace(year=target_year))
        else:
            df["time_mapped"] = df["time"]

        df = df.set_index("time_mapped")

        # merge + interpolazione temporale
        merged = df.join(self.co2_df, how="left")
        merged["co2"] = merged["co2"].interpolate("time").ffill().bfill()

        return merged["co2"].values
