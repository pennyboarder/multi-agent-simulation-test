import numpy as np


class Generator:
    def __init__(
        self,
        gen_type,
        rated_output,
        min_rate=0.0,
        max_rate=1.0,
        min_up_time=0,
        ramp_rate=0.2,
    ):
        self.gen_type = gen_type
        self.rated_output = rated_output
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.min_up_time = min_up_time
        self.ramp_rate = ramp_rate
        self.prev_output = 0
        self.gen_on = 0
        self.up_time = 0

    def compute_output(self, action_val, n_prices, hour=None, current_day=None):
        # 出力率計算
        output = int(
            self.rated_output
            * (
                self.min_rate
                + (self.max_rate - self.min_rate) * action_val / (n_prices - 1)
            )
        )
        # solar特殊処理
        if self.gen_type == "solar":
            if hour is not None and (hour < 7 or hour > 18):
                output = 0
            elif current_day is not None and current_day in [1, 4]:
                output = 0
            else:
                peak = self.rated_output
                mu = 12
                sigma = 2.5
                output = int(peak * np.exp(-((hour - mu) ** 2) / (2 * sigma**2)))
        # --- thermal/nuclear運転成約 ---
        if self.gen_type in ["thermal", "nuclear"]:
            # 最低出力率
            if output > 0 and output < int(self.rated_output * self.min_rate):
                output = int(self.rated_output * self.min_rate)
            # 最大出力率
            if output > int(self.rated_output * self.max_rate):
                output = int(self.rated_output * self.max_rate)
            # ランプ率
            ramp_limit = int(self.rated_output * self.ramp_rate)
            if abs(output - self.prev_output) > ramp_limit:
                if output > self.prev_output:
                    output = self.prev_output + ramp_limit
                else:
                    output = self.prev_output - ramp_limit
            # 最低稼働時間
            if self.gen_on == 1:
                self.up_time += 1
                if output == 0 and self.up_time < self.min_up_time:
                    output = int(self.rated_output * self.min_rate)
            if self.gen_on == 0 and output > 0:
                self.up_time = 1
            if self.gen_on == 1 and output == 0:
                self.up_time = 0
            self.gen_on = 1 if output > 0 else 0
            self.prev_output = output
        return output
