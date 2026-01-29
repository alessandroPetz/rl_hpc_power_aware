class PriceModel:
    def __init__(self, low_night=0.0005, day=0.0012, evening=0.0007, high_multiplier=3):
        self.low_night = low_night
        self.day = day
        self.evening = evening
        self.high_multiplier = high_multiplier

    def base_price(self, timestamp):
        hour = timestamp.hour

        if 0 <= hour < 6:
            return self.low_night
        elif 6 <= hour < 22:
            return self.day
        else:
            return self.evening

    def high_price(self, timestamp):
        return self.base_price(timestamp) * self.high_multiplier

    def prices_from_df(self, df, time_col="time"):
        base = df[time_col].apply(self.base_price)
        high = base * self.high_multiplier
        return base, high