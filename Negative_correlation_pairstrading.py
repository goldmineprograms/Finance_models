import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fetch data
tickers = ["GLD", "UUP"]
data = yf.download(tickers, start = "2008-01-01", end = "2025-10-31")["Close"]
returns = data.pct_change().dropna()

#Compute rolling correlation
rolling_corr = returns[tickers[0]].rolling(60).corr(returns[tickers[1]])

signals = pd.DataFrame(index=returns.index)
signals["corr"] = rolling_corr
signals["returns_A"] = returns[tickers[0]]
signals["returns_B"] = returns[tickers[1]]

#Generate signals
signals["direction_same"] = np.sign(signals["returns_A"]) == np.sign(signals["returns_B"])
signals["strong_corr"] = signals["corr"] < -0.5

def choose_trade(row):
    if row["strong_corr"] and row["direction_same"]:
        if row["returns_A"] > 0 and row["returns_B"] > 0:
            return "short_A_long_B" if row["returns_A"] > row["returns_B"] else "short_B_long_A"
        elif row["returns_A"] < 0 and row["returns_B"] < 0:
            return "long_A_short_B" if abs(row["returns_A"]) > abs(row["returns_B"]) else "long_B_short_A"
    else:
        return "no_trade"
signals["signal"] = signals.apply(choose_trade, axis=1)
signals.tail(10)

# simulate trading
signals["next_returns_A"] = signals["returns_A"].shift(-1)
signals["next_returns_B"] = signals["returns_B"].shift(-1)

def simulate_pnl(row):
    if row["signal"] == "short_A_long_B" or row["signal"] == "long_B_short_A":
        return -row["next_returns_A"] + row["next_returns_B"]
    elif row["signal"] == "short_B_long_A" or row["signal"] == "long_A_short_B":
        return -row["next_returns_B"] + row["next_returns_A"]
    else:
        return 0

signals["pnl"] = signals.apply(simulate_pnl, axis=1)
signals["cum_pnl"] = (1 + signals["pnl"]).cumprod()

#plotting
fig, ax = plt.subplots(2, 1, figsize=(10,8), sharex=True)

ax[0].plot(signals.index, signals["corr"], label="Rolling Correlation")
ax[0].axhline(-0.5, color="red", linestyle="--", label="Threshold (-0.5)")
ax[0].set_title("Rolling 60-day Correlation (GLD vs UUP)")

ax[1].plot(signals.index, signals["cum_pnl"], label="Cumulative Strategy Return")
ax[1].set_title("Negative Correlation Pairs Trading Performace")
ax[1].legend()

plt.tight_layout()
plt.show()

