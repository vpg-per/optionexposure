
import pandas as pd
import numpy as np
import gc
from datetime import datetime,timedelta
import config
from alaDataManager import fetch_stock_data, resample_bars

class ServiceManager:
    def __init__(self):
        pass

    def analyze_stockdata(self, symbol):
        data5m = fetch_stock_data(symbol, "5min")
        data15m = resample_bars(data5m, "15min")

        self.data5m = self.calculate_macd(data5m)
        self.data5m = self.calculate_rsi(self.data5m)
        self.data5m = self.calculate_overall_bias(self.data5m)
        self.data15m = self.calculate_macd(data15m)
        self.data15m = self.calculate_rsi(self.data15m)
        self.data15m = self.calculate_overall_bias(self.data15m)

        self.data5m = self.calculate_bollinger_bands(self.data5m)
        self.data5m = self.add_trade_levels(self.data5m)
        
        # start = pd.Timestamp("2026-09-10 22:50:00", tz="America/New_York")
        # end   = pd.Timestamp("2026-09-11 03:20:00", tz="America/New_York")
        # print(self.data5m.loc[start:end].head(30))
        # print(self.data15m.loc[start:end])        
        # print(self.data5m.tail(48))
        # print(self.data15m.tail(30))

        if not self.data5m.empty:
            print(f"[{symbol} 5m] Bias change alerts:")
            print(self.data5m.tail(20).to_string(columns=['close', 'OverallBias','BiasChanged', 'LevelIsValid'], index=True))
            true_rows = self.data5m[self.data5m['LevelIsValid'] == True]
            last_true_row = true_rows.iloc[-1]
            print(last_true_row)
        last_5mrow = self.data5m.iloc[-1]
        if last_5mrow['LevelIsValid']:
            return last_5mrow, self.data15m.iloc[-1]
        
        return None, None


    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """MACD in-place; uses float64 for ewm accuracy, stores float32."""
        close     = df['close'].astype('float64')
        ema_fast  = close.ewm(span=fast,   adjust=False).mean()
        ema_slow  = close.ewm(span=slow,   adjust=False).mean()
        macd_line = ema_fast - ema_slow
        sig_line  = macd_line.ewm(span=signal, adjust=False).mean()

        df['macd']      = macd_line.round(2).astype('float32')
        df['msignal']   = sig_line.round(2).astype('float32')
        df['histogram'] = (macd_line - sig_line).round(2).astype('float32')
        above_signal = macd_line > sig_line
        below_signal = macd_line < sig_line
        above_zero   = macd_line > 0
        below_zero   = macd_line < 0

        trend = pd.Series("Neutral", index=df.index)
        trend[above_signal & above_zero] = "Bullish"
        trend[below_signal & below_zero] = "Bearish"
        df['MACDTrend'] = trend        
        df = df.drop(columns=["macd","msignal","histogram"], errors="ignore")

        del close, ema_fast, ema_slow, macd_line, sig_line, above_signal, below_signal, above_zero, below_zero, trend
        return df

    def calculate_rsi(self, df, period=14, smooth=3, threshold=0.5):
        diff = df['close'].astype('float64').diff()
        gain = diff.clip(lower=0)
        loss = (-diff).clip(lower=0)

        alpha    = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        rs       = avg_gain / avg_loss

        rsi_raw = 100 - (100 / (1 + rs))
        rsi     = rsi_raw.round(2).astype('float32')
        rsignal = rsi_raw.ewm(span=period, adjust=False).mean().round(2).astype('float32')
        rsi_hist = rsi_raw - rsignal.astype('float64')
        hist_smooth = rsi_hist.rolling(window=smooth).mean()
        hist_slope  = hist_smooth.diff()

        # Direction: is RSI currently above or below its own signal line.
        above_rsignal = rsi_raw > rsignal.astype('float64')
        below_rsignal = rsi_raw < rsignal.astype('float64')
        rsi_dir = pd.Series("Neutral", index=df.index)
        rsi_dir[above_rsignal] = "Up"
        rsi_dir[below_rsignal] = "Down"

        # Accelerating: histogram moving further from zero, in the same
        #   direction it's already pointing -- momentum building.
        # Decelerating: histogram moving back toward zero -- momentum fading,
        #   regardless of which side of zero it's on.
        # None: slope too small to call either way (flat/dead zone).
        same_direction = np.sign(hist_slope) == np.sign(hist_smooth)
        meaningful     = hist_slope.abs() > threshold

        momentum = pd.Series("None", index=df.index)
        momentum[meaningful & same_direction]  = "Accelerating"
        momentum[meaningful & ~same_direction] = "Decelerating"

        # rsitrend: only call Bullish/Bearish when direction AND accelerating momentum agree.
        rsitrend = pd.Series("Neutral", index=df.index)
        rsitrend[(rsi_dir == "Up")   & (momentum == "Accelerating")] = "Bullish"
        rsitrend[(rsi_dir == "Down") & (momentum == "Accelerating")] = "Bearish"

        df['rsi']      = rsi
        df['rsignal']  = rsignal
        df['rhistogram'] = rsi_hist
        df['momentum'] = momentum
        df['RSITrend'] = rsitrend
        df = df.drop(columns=["rsi","rsignal","rhistogram","momentum"], errors="ignore")

        del diff, gain, loss, rs, rsi_raw, rsi_hist, hist_smooth, hist_slope, same_direction, meaningful
        return df

    
    def calculate_bollinger_bands(self, df, period=20, std_dev=2):
        mid    = df['close'].rolling(window=period).mean()
        stddev = df['close'].rolling(window=period).std()
        df['midbnd'] = mid.round(2)
        df['ubnd']   = (mid + std_dev * stddev).round(2)
        df['lbnd']   = (mid - std_dev * stddev).round(2)
        del mid, stddev
        return df

    def calculate_overall_bias(self, df):
        bullish = (df['MACDTrend'] == "Bullish") & df['RSITrend'].isin(["Bullish"])
        bearish = (df['MACDTrend'] == "Bearish") & df['RSITrend'].isin(["Bearish"])

        bias = pd.Series("Neutral", index=df.index)
        bias[bullish] = "Bullish"
        bias[bearish] = "Bearish"
        df['OverallBias'] = bias
        df = df.drop(columns=["bullish","bearish"], errors="ignore")

        non_neutral = df['OverallBias'].where(df['OverallBias'] != "Neutral")
        prev_bias = non_neutral.ffill().shift(1)

        # df["OverallBias_prev"] = df["OverallBias"].shift(1)
        # changed = df["OverallBias"] != df["OverallBias_prev"]
        # back = ["OverallBias_prev", "OverallBias"]
        # remaining = [c for c in df.columns if c not in back]
        # df = df[remaining + back]
        # df["IsChanged"] = changed

        is_directional = df['OverallBias'].isin(["Bullish", "Bearish"])
        df['PreviousBias'] = prev_bias.fillna("None")
        df['BiasChanged'] = is_directional & (df['OverallBias'] != prev_bias)

        del bearish, bullish
        return df

    def add_trade_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add OpenOrder, StopLoss, Target, LevelsValid columns.

        Only on rows where BiasChanged is True:
            Bullish: OpenOrder=midbnd, StopLoss=prev low,  Target=ubnd
                    accepted only if StopLoss < OpenOrder < Target
            Bearish: OpenOrder=midbnd, StopLoss=prev high, Target=lbnd
                    accepted only if StopLoss > OpenOrder > Target
        Rejected / non-signal rows (and rows with NaN bands or no previous
        candle) get 0 in all three levels; LevelsValid is True only for accepted
        signals, so `BiasChanged & ~LevelsValid` lists the rejected ones.
        """
        out = df.copy()

        f64 = lambda col: out[col].to_numpy(dtype="float64")
        changed = out["BiasChanged"].fillna(False).to_numpy(dtype=bool)
        bias = out["OverallBias"].to_numpy(dtype=object)
        is_long = changed & (bias == "Bullish")
        is_short = changed & (bias == "Bearish")

        entry = f64("midbnd").round(2)
        #stop = np.where(is_long, np.roll(f64("low"), 1), np.roll(f64("high"), 1))
        #stop[0] = np.nan  # roll wraps the last bar around; row 0 has no prev candle
        #stop = stop.round(2)
        stop = np.where(is_long, f64("lbnd"), f64("ubnd")).round(2)
        target = np.where(is_long, f64("ubnd"), f64("lbnd")).round(2)

        valid = (is_long & (stop < entry) & (entry < target)) | (
            is_short & (stop > entry) & (entry > target)            )
        out["OpenOrder"] = np.where(valid, entry, 0.0)
        out["StopLoss"] = np.where(valid, stop, 0.0)
        out["Target"] = np.where(valid, target, 0.0)
        out["LevelIsValid"] =valid

        out = out.drop(columns=["MACDTrend"," RSITrend","PreviousBias","midbnd","ubnd","lbnd"], errors="ignore")
        df= out.copy()
        del changed, bias, entry, stop, target

        return out



    