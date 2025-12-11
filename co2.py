import pandas as pd

class CarbonIntensityModels:
    def __init__(self, csv_file="electricitymaps_cache.csv"):
        """
        csv_file: il tuo CSV con colonne come
        'Datetime (UTC)', 'Carbon intensity gCO₂eq/kWh (Life cycle)' ecc.
        """
        self.csv_file = csv_file
        self.co2_df = self._load_csv()

    # ------------------------------------------
    # 1️⃣ Legge il CSV e costruisce DataFrame
    # ------------------------------------------
    def _load_csv(self):
        df = pd.read_csv(self.csv_file, parse_dates=["Datetime (UTC)"])
        df = df.set_index("Datetime (UTC)").sort_index()

        # rendiamo l'indice timezone-aware UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        # prendiamo la colonna Life cycle
        df = df.rename(columns={"Carbon intensity gCO₂eq/kWh (Life cycle)": "co2"})
        return df[["co2"]]

    # ------------------------------------------
    # 2️⃣ Mapping anni e interpolazione
    # ------------------------------------------
    def co2_from_csv(self, df):
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], utc=True)

        # Mapping anni se necessario
        csv_years = set(self.co2_df.index.year.unique())
        df_years = set(df["time"].dt.year.unique())
        missing_years = df_years - csv_years

        if missing_years:
            target_year = min(csv_years)
            print(f"⚠️ Gli anni {missing_years} non sono presenti nel CSV, mapping su anno {target_year}")
            df["time_mapped"] = df["time"].apply(lambda t: t.replace(year=target_year))
        else:
            df["time_mapped"] = df["time"]

        # Impostiamo time_mapped come indice per interpolazione
        df = df.set_index("time_mapped")

        # Merge con CSV
        df = df.merge(self.co2_df, how="left", left_index=True, right_index=True)

        # Interpolazione temporale
        df["co2"] = df["co2"].interpolate(method="time").ffill().bfill()

        # Ripristiniamo l’indice originale
        df = df.reset_index(drop=True)

        return df["co2"].values


        

        