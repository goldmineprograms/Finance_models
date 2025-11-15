import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

stock = "MU"
start = "2015-01-01"
end = "2025-10-31"
period = 252

# Financial Data
data = yf.download(stock, start, end, period)
df = pd.DataFrame(data=data).dropna()

# MACD Signals
fast_period = 12
slow_period = 26
signal_period = 9

# Calculating the EMAs
df["fast_ema"] = df["Close"].ewm(span=fast_period, adjust=False).mean()
df["slow_ema"] = df["Close"].ewm(span=slow_period, adjust=False).mean()
df["macd"] = df["fast_ema"] - df["slow_ema"]
df["signal"] = df["macd"].ewm(span=signal_period, adjust=False).mean()

# Calculate MACD histogram
df["histogram"] = df["macd"] - df["signal"]

# Generating trading signals
df["position"] = 0     #0 = neutral, 1 = long, -1 = short

# Buy Signal
df.loc[df["macd"] > df["signal"], "position"] = 1

#Sell signal
df.loc[df["macd"] < df["signal"], "position"] = -1

# Identify crossover points
df["signal_change"] = df["position"].diff()
buy_signal = df[df["signal_change"] == 2]
sell_signal = df[df["signal_change"] == -2]

# Calculate returns
df["returns"] = df["Close"].pct_change()
df["strategy_returns"] = df["position"].shift(1) * df["returns"]

# Calculate cumulative returns
df["cumulative_returns"] = (1 + df["returns"]).cumprod()
df["cumulative_strategy_returns"] = (1 + df["strategy_returns"]).cumprod()

# Plotting results
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,10), sharex=True)

# Plot Price and Signals
ax1.plot(df.index, df["Close"], label="Price", linewidth=2)
ax1.scatter(buy_signal.index, df.loc[buy_signal.index, "Close"], color="green", marker="^", s=100, label="Buy Signal", zorder=5)
ax1.scatter(sell_signal.index, df.loc[sell_signal.index, "Close"], color="red", marker="v", s=100, label="Buy Signal", zorder=5)
ax1.set_ylabel("Price")
ax1.set_title("MACD Trading Strategy")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot MACD and Signal Line
ax2.plot(df.index, df["macd"], label="MACD", linewidth=2)
ax2.plot(df.index, df["signal"], label="Signal Line", linewidth=2)
ax2.bar(df.index, df["histogram"], label="Histogram", alpha=0.3)
ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
ax2.set_ylabel("MACD")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot cumulative returns
ax3.plot(df.index, df["cumulative_returns"], label="Buy & Hold", linewidth=2)
ax3.plot(df.index, df["cumulative_strategy_returns"], label="MACD Strategy", linewidth=2)
ax3.set_ylabel("Cumulative Returns")
ax3.set_xlabel("Date")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()