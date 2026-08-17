
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from statsmodels.tsa.stattools import adfuller

SECURITIES_20 = [
    # energy companies
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX", "OXY", "HAL",
    # banks 
    "JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "COF",
]

START_DATE = "2014-09-15"
END_DATE = "2024-08-26"
N_SEGMENTS = 10  # ten equal 10% time series chunks 

MIN_OBSERVATIONS = 30  # minimum overlapping history to even attempt a PCA fit
MIN_HALFLIFE_WEEKS = 2  # too fast to trust out-of-sample (the paper's own caveat)
MAX_HALFLIFE_WEEKS = 5  # too slow to be practical within a ~52-week training window
KALMAN_Q_RATIO = 0.01  # equilibrium drift variance Q, as a fraction of observation variance R
dt = 1  # one time step per weekly observation

KELLY_FRACTION = 0.5  # half-Kelly: guards against noisy theta/mu/sigma estimates
MAX_POSITION_FRACTION = 0.5  # hard cap regardless of what (fractional) Kelly recommends
STARTING_PORTFOLIO = 100_000.0
STEPS_PER_YEAR = 52  # weekly observations

#download weekly prices for the 20 securities and save them to CSV files
#save a combined CSV file rows securities and columns dates with the closing prices
def download_weekly_prices(tickers, start, end):

    frames = {}
    for ticker in tickers:
        df = yf.Ticker(ticker).history(start=start, end=end, interval="1wk")
        df.to_csv(f"{ticker}.csv")
        frames[ticker] = df["Close"]
    combined = pd.DataFrame(frames).T
    combined.index.name = "ticker"
    combined.to_csv("securities20_weekly_combined.csv")

# creates segments which breaks the dates into n_segments equal parts
# overwrites the last segment to include any remaining dates if the division isn't exact
def segment_dates(dates, n_segments):

    segment_size = len(dates) // n_segments
    segments = [dates[i * segment_size:(i + 1) * segment_size] for i in range(n_segments)]
    segments[-1] = dates[(n_segments - 1) * segment_size:]
    return segments


# test for stationarity using the Augmented Dickey-Fuller 
# fit the AR(1) model f(x_{t}) = x_{t+1} 
# check b is in (0,1) to ensure mean-reversion
# calculate the implied continous time OU parameters 
# theta, mu, halflife can be found from the OU model equations
# it can be derived by ito isometry that the residual variance of the ar(1) model is related to the OU parameters 
# by resid_var = sigma^2 * (1 - b^2) / (2 * theta) (lengthy derivation in paper also multivariate case)

def fit_stationary_ar1(combo):

    _, p_value, *_ = adfuller(combo)
    if p_value >= 0.05:
        return None

    x = combo.to_numpy()
    x_t, x_tp1 = x[:-1], x[1:]
    b, a = np.polyfit(x_t, x_tp1, 1)
    resid = x_tp1 - (a + b * x_t)
    if not (0 < b < 1):
        return None

    theta = -np.log(b) / dt
    mu = a / (1 - b)
    sigma = np.sqrt(resid.var(ddof=0) * 2 * theta / (1 - b ** 2))
    halflife = np.log(2) / theta
    return {"theta": theta, "mu": mu, "sigma": sigma, "halflife": halflife, "p_value": p_value}


# take the covariance matrix of log prices, find the eigenvectors and eigenvalues (PCA), 
# each eigenvector could be a potential cointegrated basket
# test each eigenvector for stationarity and mean-reversion using fit_stationary_ar1
# combo is a linear combination of the log prices weighted by the elements of the eigenvector
# smallest eigenvalue corresponds to the least variance, which is the best cointegration candidate
# Ensure enough history is available for each security in the basket by 
# checking for enough shared non-NaN observations in the training window. 
# find the unconditional z-score of today's deviation, and
# its horizon-adjustment (deviation relative to the unceartanty associated with period dt)
def select_cointegrated_baskets(combined_train, tickers):

    available = [t for t in tickers if combined_train.loc[t].notna().sum() >= MIN_OBSERVATIONS]
    if len(available) < 2:
        return []

    log_prices = np.log(combined_train.loc[available]).T.dropna()
    if len(log_prices) < MIN_OBSERVATIONS:
        return []

    cov = np.cov(log_prices.to_numpy(), rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalue order

    baskets = []
    for k in range(len(eigvals)):
        w = eigvecs[:, k]
        combo = pd.Series(log_prices.to_numpy() @ w, index=log_prices.index)

        fit = fit_stationary_ar1(combo)
        if fit is None:
            continue
        theta, mu, sigma, halflife = fit["theta"], fit["mu"], fit["sigma"], fit["halflife"]
        if not (MIN_HALFLIFE_WEEKS <= halflife <= MAX_HALFLIFE_WEEKS):
            continue

        std = sigma / np.sqrt(2 * theta)
        alpha = abs(combo.to_numpy()[-1] - mu)
        z_inf = alpha / std
        z_tau = z_inf * (1 - np.exp(-theta * dt)) / np.sqrt(1 - np.exp(-2 * theta * dt))

        weights = dict(zip(available, w))
        label = "+".join(f"{wt:+.2f}*{tk}" for tk, wt in weights.items())
        baskets.append({"eigenvalue": eigvals[k], "weights": weights, "label": label,
                         "theta": theta, "mu": mu, "sigma": sigma, "halflife": halflife,
                         "alpha": alpha, "z_inf": z_inf, "z_tau": z_tau})

    return baskets

# simulate an Ornstein-Uhlenbeck process with given parameters theta, mu, sigma, and a random number generator rng
# Euler-Maruyama discretization of the OU SDE: dx_t = theta*(mu - x_t)*dt + sigma*dW_t (dt=1, matching the
# script's weekly convention)
def simulate_ou_spread(index, theta, mu, sigma, rng):

    x = np.empty(len(index))
    x[0] = mu
    for t in range(1, len(index)):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) + sigma * rng.standard_normal()
    return pd.Series(x, index=index)

# assuming 0 risk-free rate, the optimal Kelly fraction f*
# is the expected return (OU drift) divided by the variance of returns.
# maximises log returns.
def continuous_kelly(theta, mu, sigma, y_t):
    
    drift = theta * (mu - y_t)
    sharpe = drift / sigma
    f_star = drift / sigma ** 2
    return f_star, sharpe

# Scalar Kalman filter initial mu updated each time step
# assumed constant theta and sigma, with 
# observation variance r_var = sigma^2/(2*theta) and the drift variance q_var = q_ratio*r_var
# these are implied from the AR(1) fit on the basket's training window, and are assumed to remain constant over the test window.
# q_var is not directly observed we just assume 1% of r_var is added per unit time step to account for slow drift in the equilibrium level mu.
#https://filippomb.github.io/python-time-series-handbook/notebooks/07/kalman-filter.html

def kalman_filter_mu(observations, mu0, r_var, q_var):
   
    mu_hat = mu0
    p = r_var  # initial estimate uncertainty
    filtered = np.empty(len(observations))
    for i, y in enumerate(observations):
        p = p + q_var  # predict: uncertainty grows by the drift variance
        k = p / (p + r_var)  # Kalman gain
        mu_hat = mu_hat + k * (y - mu_hat)  # update toward the new observation
        #= (mu_hat/p + y/r_var) / (1/p + 1/r_var) gaussian estimates weighted by 1/variance
        p = (1 - k) * p
        filtered[i] = mu_hat
    return filtered

# Renormalise the weights to sum to 1 in absolute value, 
# so that the Kelly fraction f* can be applied to the total portfolio value 
def _gross_normalized_weights(weights, tickers):

    w = np.array([weights[t] for t in tickers])
    return w / np.abs(w).sum()

# Run the kalman filter to find evolving mu,
# normalise w and create an array of leg prices to calculte actual returns on the underlying securities,
# apply the continuous-time (capped) Kelly criterion to each step of the test period,
# step return is kelly fraction * w(normalised) dot (leg returns per step)
# 
def kalman_kelly_growth(combo_full, test_start_idx, theta, sigma, mu0,
                         kelly_fraction, max_position_fraction, q_ratio=KALMAN_Q_RATIO,
                         leg_prices=None, weights=None):
   
    spread_values = combo_full.to_numpy()
    r_var = sigma ** 2 / (2 * theta)
    mu_path = kalman_filter_mu(spread_values, mu0, r_var, q_ratio * r_var)

    if leg_prices is not None:
        tickers = list(weights.keys())
        normalized_w = _gross_normalized_weights(weights, tickers)
        leg_price_arr = leg_prices[tickers].to_numpy()

    growth = 1.0
    period_returns = []
    f_history = []  # kelly_fraction*f* actually applied each step, for order sizing
    for i in range(test_start_idx, len(spread_values) - 1):
        f_star, _ = continuous_kelly(theta, mu_path[i], sigma, spread_values[i])
        f_capped = np.clip(kelly_fraction * f_star, -max_position_fraction, max_position_fraction)
        f_history.append(f_capped)

        if leg_prices is not None:
            leg_simple_returns = leg_price_arr[i + 1] / leg_price_arr[i] - 1
            step_return = f_capped * np.dot(normalized_w, leg_simple_returns)
        else:# for the ou-simulated path
            step_return = f_capped * (spread_values[i + 1] - spread_values[i])

        growth *= max(0.0, 1 + step_return)
        period_returns.append(step_return)
    return growth, period_returns, f_history

# for each basket, calculate the period returns over the test window using the Kalman-Kelly growth model,
#  and simulate a synthetic OU path with the fitted parameters to compare against.
def _basket_period_returns(basket, train_dates, test_dates, combined_csv, rng):
    
    tickers = list(basket["weights"].keys())
    w = np.array([basket["weights"][t] for t in tickers])

    # Extend back into the tail of the training dates 
    extended_dates = train_dates.append(test_dates)
    prices = combined_csv.loc[tickers, extended_dates].T.dropna()
    if len(prices) < 2:
        return None
    log_prices = np.log(prices)

    combo = pd.Series(log_prices.to_numpy() @ w, index=log_prices.index)
    test_start_idx = combo.index.searchsorted(test_dates[0])
    _, period_returns, _ = kalman_kelly_growth(
        combo, test_start_idx, basket["theta"], basket["sigma"], basket["mu"],
        KELLY_FRACTION, MAX_POSITION_FRACTION,
        leg_prices=prices, weights=basket["weights"])

    simulated_combo = simulate_ou_spread(combo.index, basket["theta"], basket["mu"], basket["sigma"], rng)
    simulated_growth, simulated_period_returns, _ = kalman_kelly_growth(
        simulated_combo, test_start_idx, basket["theta"], basket["sigma"], basket["mu"],
        KELLY_FRACTION, MAX_POSITION_FRACTION)

    return period_returns, simulated_period_returns, simulated_growth

# split the portfolio into sleeves (available cointegrated baskets) according to each basket's theta, 
# compound each sleeve over the test period using the basket's period returns.
def _compound_sleeves(basket_returns, portfolio, test_dates):

    curve_points = []
    n_weeks = max((len(r) for _, r in basket_returns), default=0)
    if n_weeks == 0:
        curve_points.append((test_dates[-1], portfolio))
        return portfolio, curve_points

    total_theta = sum(theta for theta, _ in basket_returns)
    sleeves = [theta / total_theta * portfolio for theta, _ in basket_returns]
    for j in range(n_weeks):
        for k, (_, period_returns) in enumerate(basket_returns):
            if j < len(period_returns):
                sleeves[k] = max(0.0, sleeves[k] * (1 + period_returns[j]))
        portfolio = sum(sleeves)
        date = test_dates[j + 1] if j + 1 < len(test_dates) else test_dates[-1]
        curve_points.append((date, portfolio))
    return portfolio, curve_points

# annualised Sharpe ratio (0 risk-free rate) of a series of per-step simple returns
def _annualized_sharpe(returns):
    returns = np.asarray(returns)
    if len(returns) < 2 or returns.std() == 0:
        return float("nan")
    return returns.mean() / returns.std() * np.sqrt(STEPS_PER_YEAR)

# Run the walk-forward backtest,
# segmenting the dates into N_SEGMENTS, selecting cointegrated baskets from each training segment
# compounding the portfolio over the test segments, while also simulating an OU path for comparison.

def run_walk_forward_backtest(combined_csv, seed=None):

    dates = combined_csv.columns
    segments = segment_dates(dates, N_SEGMENTS)
    rng = np.random.default_rng(seed)

    portfolio = STARTING_PORTFOLIO
    sim_portfolio = STARTING_PORTFOLIO
    curve = [(segments[0][-1], portfolio)]
    sim_curve = [(segments[0][-1], sim_portfolio)]
    window_stats = []
    basket_sim_results = []
    for i in range(len(segments) - 1):
        train_dates, test_dates = segments[i], segments[i + 1]
        combined_train = combined_csv[train_dates]
        baskets = select_cointegrated_baskets(combined_train, SECURITIES_20)

        basket_returns = []
        sim_basket_returns = []
        for basket in baskets:
            result = _basket_period_returns(basket, train_dates, test_dates, combined_csv, rng)
            if result is not None:
                period_returns, simulated_period_returns, simulated_growth = result
                basket_returns.append((basket["theta"], period_returns))
                sim_basket_returns.append((basket["theta"], simulated_period_returns))
                basket_sim_results.append({
                    "profitable": simulated_growth > 1.0,
                    "sharpe": _annualized_sharpe(simulated_period_returns),
                })

        portfolio_before = portfolio
        portfolio, curve_points = _compound_sleeves(basket_returns, portfolio, test_dates)
        curve.extend(curve_points)
        sim_portfolio, sim_curve_points = _compound_sleeves(sim_basket_returns, sim_portfolio, test_dates)
        sim_curve.extend(sim_curve_points)

        window_values = [portfolio_before] + [v for _, v in curve_points]
        window_stats.append({
            "qualified": len(basket_returns) > 0,
            "total_return": portfolio / portfolio_before - 1,
            "weekly_returns": np.array(window_values[1:]) / np.array(window_values[:-1]) - 1,
        })

    return curve, sim_curve, window_stats, basket_sim_results

# per-window stats: how many of the N_SEGMENTS-1 periods had a qualifying basket,
# how many of those were profitable, and the average annualised Sharpe ratio across qualifying
# periods vs. across just the profitable ("successful") ones.
def summarize_windows(window_stats):
    n_periods = len(window_stats)
    qualifying = [w for w in window_stats if w["qualified"]]
    successful = [w for w in qualifying if w["total_return"] > 0]

    sharpes_qualifying = [_annualized_sharpe(w["weekly_returns"]) for w in qualifying]
    sharpes_successful = [_annualized_sharpe(w["weekly_returns"]) for w in successful]

    return {
        "n_periods": n_periods,
        "n_qualifying": len(qualifying),
        "n_successful": len(successful),
        "avg_sharpe_qualifying": float(np.nanmean(sharpes_qualifying)) if sharpes_qualifying else float("nan"),
        "avg_sharpe_successful": float(np.nanmean(sharpes_successful)) if sharpes_successful else float("nan"),
    }

# basket_sim_results: how many qualifying baskets were
# profitable on their own OU-simulated path, out of the total, and the average annualised Sharpe
# ratio of those simulated paths.
def summarize_basket_simulations(basket_sim_results):
    n_baskets = len(basket_sim_results)
    n_profitable = sum(r["profitable"] for r in basket_sim_results)
    sharpes = [r["sharpe"] for r in basket_sim_results]
    avg_sharpe = float(np.nanmean(sharpes)) if sharpes else float("nan")
    return {"n_baskets": n_baskets, "n_profitable": n_profitable, "avg_sharpe": avg_sharpe}

# plot simulated and real equity curves
def plot_equity_curve(curve, sim_curve=None, title="Walk-forward equity curve", path="equity_curve.png"):
    dates, portfolio_values = zip(*curve)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, portfolio_values, linewidth=1.5, label="Real")
    if sim_curve is not None:
        sim_dates, sim_portfolio_values = zip(*sim_curve)
        ax.plot(sim_dates, sim_portfolio_values, linewidth=1.5, alpha=0.8, label="Simulated OU")
        ax.legend()
    ax.set_title(title)
    ax.set_ylabel("Portfolio value")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    download_weekly_prices(SECURITIES_20, START_DATE, END_DATE)
    combined_csv = pd.read_csv("securities20_weekly_combined.csv", index_col=0)

    combined_csv.columns = pd.to_datetime(combined_csv.columns, utc=True)

    curve, sim_curve, window_stats, basket_sim_results = run_walk_forward_backtest(combined_csv, seed=0)
    real_summary = summarize_windows(window_stats)
    print(f"Qualifying periods: {real_summary['n_qualifying']}/{real_summary['n_periods']}")
    print(f"Positive performance: {real_summary['n_successful']}/{real_summary['n_qualifying']}")
    print(f"Avg annualised Sharpe (qualifying periods): {real_summary['avg_sharpe_qualifying']:.2f}")
    print(f"Avg annualised Sharpe (successful periods): {real_summary['avg_sharpe_successful']:.2f}")

    sim_summary = summarize_basket_simulations(basket_sim_results)
    print(f"OU-simulated validation: profitable in {sim_summary['n_profitable']}/{sim_summary['n_baskets']} "
          f"qualifying baskets, avg annualised Sharpe {sim_summary['avg_sharpe']:.2f}")

    plot_equity_curve(curve, sim_curve)
    plot_equity_curve(curve)
