"""All tunable settings in one place. Change symbol, lookback window, feeds,
or resample rules here rather than hunting through the rest of the code."""

import pytz
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

SYMBOL = "SPY"
LOOKBACK_DAYS = 5
BASE_TIMEFRAME = TimeFrame(5, TimeFrameUnit.Minute)
EASTERN = pytz.timezone("America/New_York")

# Feeds to pull, in priority order. Earlier feeds win on overlapping
# timestamps when merged. "sip" covers regular + full extended hours (needs
# Algo Trader Plus); "boats" covers the overnight session (needs an
# overnight data subscription). Add "iex" as a fallback here if you don't
# have "sip" access -- "iex" only covers regular market hours on the free tier.
FEEDS = ["sip", "boats"]

# (display label, pandas resample rule) pairs used by resample_bars()
RESAMPLE_RULES = [("15-Minute", "15min"), ("30-Minute", "30min"), ("1-Hour", "1h")]
