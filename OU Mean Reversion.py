
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
STEPS_PER_YEAR = 52

KELLY_FRACTION = 0.5  # half-Kelly: guards against noisy theta/mu/sigma estimates
MAX_POSITION_FRACTION = 0.5  # hard cap regardless of what (fractional) Kelly recommends
STARTING_PORTFOLIO = 100_000.0

N_RANDOM_OU_SPREADS = 10
RANDOM_OU_EVAL_YEARS = 5


def download_weekly_prices(tickers, start, end):
    frames = {}
    for ticker in tickers:
        df = yf.Ticker(ticker).history(start=start, end=end, interval="1wk")
        df.to_csv(f"{ticker}.csv")
        frames[ticker] = df["Close"]
    combined = pd.DataFrame(frames).T
    combined.index.name = "ticker"
    combined.to_csv("securities20_weekly_combined.csv")


def segment_dates(dates, n_segments):
    # n_segments equal-sized chunks covering the whole date range; any
    # remainder from integer division is folded into the last chunk.
    segment_size = len(dates) // n_segments
    segments = [dates[i * segment_size:(i + 1) * segment_size] for i in range(n_segments)]
    segments[-1] = dates[(n_segments - 1) * segment_size:]
    return segments


def fit_stationary_ar1(combo):
    # ADF-test combo for stationarity; if stationary and mean-reverting
    # (AR(1) coefficient in (0,1)), fit theta/mu/sigma via the exact MLE
    # (Section 4.2, eq. 42): discrete AR(1) coefficients are the exact
    # MLE of the underlying continuous OU process at dt=1, theta=-ln(b),
    # mu=a/(1-b), and sigma recovers from the one-step residual variance
    # via resid_var=sigma^2*(1-b^2)/(2*theta) (eq. 31/76). Returns None
    # if combo doesn't qualify. theta/sigma/halflife are all in units of
    # "one bar" of combo's own sampling frequency, whatever that is.
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


def select_cointegrated_baskets(combined_train, tickers):
    # Section 4.1's model-independent routine: PCA the sample covariance
    # of log-prices across the WHOLE basket (eq. 39), then test each
    # eigenvector Y_e(n) = X'e(n), starting from the smallest eigenvalue
    # (least variance -> best cointegration candidate, eq. 37), for
    # stationarity and mean-reversion via fit_stationary_ar1. Securities
    # without enough history in a given training window are simply
    # excluded from this window's basket rather than forcing NaNs
    # through the PCA.
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

        # eq. 43/45: the unconditional z-score of today's deviation, and
        # its horizon-adjustment to the ACTUAL one-week trading step --
        # how much of the eventual reversion is realized in a single
        # period, used here purely as a diagnostic on signal strength.
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

def simulate_ou_spread(index, theta, mu, sigma, rng):
    # Euler-Maruyama simulation of dX = theta*(mu-X)dt + sigma*dW at dt=1,
    # matching the step size the AR(1) fit above was estimated at.
    x = np.empty(len(index))
    x[0] = mu
    for t in range(1, len(index)):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) + sigma * rng.standard_normal()
    return pd.Series(x, index=index)


def continuous_kelly(theta, mu, sigma, y_t):
    # f* = mu_effective / sigma^2 = Sharpe / sigma, assuming a 0 risk-free
    # rate. The OU drift theta*(mu-Y_t) stands in for the basket's
    # instantaneous expected return: the further Y_t sits from
    # equilibrium, the stronger the pull back and the bigger the edge,
    # so f* naturally scales with the deviation rather than being a
    # single fixed bet size.
    drift = theta * (mu - y_t)
    sharpe = drift / sigma
    f_star = drift / sigma ** 2
    return f_star, sharpe

def kalman_filter_mu(observations, mu0, r_var, q_var):
    # Scalar Kalman filter for a local-level model: mu_t = mu_{t-1} + w_t
    # (w_t ~ N(0, q_var), the equilibrium's own slow drift) observed
    # through y_t = mu_t + v_t (v_t ~ N(0, r_var), the OU process's own
    # stationary spread around its mean). Returns the filtered mu_hat at
    # EVERY step, where mu_hat[i] only ever depends on observations[0..i]
    # -- no look-ahead. This is the minimum-variance recursive estimate
    # of the current equilibrium level, updated in O(1) per new
    # observation rather than re-fitting a regression on a trailing
    # window each time.
    mu_hat = mu0
    p = r_var  # initial estimate uncertainty
    filtered = np.empty(len(observations))
    for i, y in enumerate(observations):
        p = p + q_var  # predict: uncertainty grows by the drift variance
        k = p / (p + r_var)  # Kalman gain
        mu_hat = mu_hat + k * (y - mu_hat)  # update toward the new observation
        p = (1 - k) * p
        filtered[i] = mu_hat
    return filtered


def kalman_kelly_growth(combo_full, test_start_idx, theta, sigma, mu0,
                         kelly_fraction, max_position_fraction, q_ratio=KALMAN_Q_RATIO):
    # mu is a Kalman-filtered running estimate of the equilibrium level
    # rather than a single frozen training-window value or a window-
    # based re-fit -- see kalman_filter_mu. theta and sigma stay fixed
    # at their training-window fit, since reversion speed and
    # volatility drift far more slowly than the equilibrium level
    # itself. The training-window ADF/AR(1) test that qualified this
    # basket is trusted for the whole test window once made.
    spread_values = combo_full.to_numpy()
    r_var = sigma ** 2 / (2 * theta)
    mu_path = kalman_filter_mu(spread_values, mu0, r_var, q_ratio * r_var)

    growth = 1.0
    period_returns = []
    f_history = []  # kelly_fraction*f* actually applied each step, for order sizing
    for i in range(test_start_idx, len(spread_values) - 1):
        f_star, _ = continuous_kelly(theta, mu_path[i], sigma, spread_values[i])
        f_capped = np.clip(kelly_fraction * f_star, -max_position_fraction, max_position_fraction)
        f_history.append(f_capped)
        step_return = f_capped * (spread_values[i + 1] - spread_values[i])
        growth *= max(0.0, 1 + step_return)
        period_returns.append(step_return)
    return growth, period_returns, f_history


def basket_dollar_positions(weights, prices, capital):
    # Convert the basket's log-price eigenvector weights into literal
    # executable orders. d(log P_i) ~= dP_i/P_i for small moves, so a
    # DOLLAR exposure of w_i in security i reproduces w_i*d(log P_i) of
    # the combo's return -- dollar exposure to leg i is proportional to
    # w_i directly, not to w_i/price_i (that would be a log-price-
    # weighted, not dollar-weighted, position). Weights are renormalized
    # so GROSS dollar exposure (sum of |dollar_i|) equals |capital|; NET
    # exposure (sum of dollar_i) is whatever the eigenvector implies and
    # may be nonzero -- forcing dollar-neutrality would mean trading a
    # different combination than the one actually found to cointegrate.
    # capital's sign flips every leg's side (long the combo vs short it).
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers])
    gross_w = np.abs(w).sum()
    dollar_exposure = capital * w / gross_w

    dollars, shares = {}, {}
    for i, t in enumerate(tickers):
        dollars[t] = dollar_exposure[i]
        shares[t] = dollar_exposure[i] / prices[t]
    return dollars, shares


def _basket_period_returns(basket, train_dates, test_dates, combined_csv):
    # Kalman-Kelly per-period returns for one basket over one test
    # window; returns None if there isn't enough overlapping price
    # history.
    tickers = list(basket["weights"].keys())
    w = np.array([basket["weights"][t] for t in tickers])

    # Extend back into the tail of the training dates so the Kalman
    # filter is already warmed up by the time the test period starts --
    # those leading bars are pure history relative to the test period,
    # not a peek into its future.
    extended_dates = train_dates.append(test_dates)
    log_prices = np.log(combined_csv.loc[tickers, extended_dates]).T.dropna()
    if len(log_prices) < 2:
        return None

    combo = pd.Series(log_prices.to_numpy() @ w, index=log_prices.index)
    test_start_idx = combo.index.searchsorted(test_dates[0])
    _, period_returns, _ = kalman_kelly_growth(
        combo, test_start_idx, basket["theta"], basket["sigma"], basket["mu"],
        KELLY_FRACTION, MAX_POSITION_FRACTION)
    return period_returns


def run_walk_forward_backtest(combined_csv):
    # Returns a weekly (date, portfolio_value) equity curve rather than
    # a single end-of-window number. Capital is split across a window's
    # qualifying baskets once, at the window's start, weighted by theta
    # (a more stable cointegration-quality signal than a single last-
    # observed z-score); each basket's slice of capital then compounds
    # independently through its own weekly Kalman-Kelly returns for the
    # rest of the window -- summing the sleeves at the final week gives
    # exactly the same end-of-window total as weighting each basket's
    # overall growth factor, just resolved to weekly points in between.
    dates = combined_csv.columns
    segments = segment_dates(dates, N_SEGMENTS)

    portfolio = STARTING_PORTFOLIO
    curve = [(segments[0][-1], portfolio)]
    for i in range(len(segments) - 1):
        train_dates, test_dates = segments[i], segments[i + 1]
        combined_train = combined_csv[train_dates]
        baskets = select_cointegrated_baskets(combined_train, SECURITIES_20)

        basket_returns = []
        for basket in baskets:
            period_returns = _basket_period_returns(basket, train_dates, test_dates, combined_csv)
            if period_returns is not None:
                basket_returns.append((basket["theta"], period_returns))

        n_weeks = max((len(r) for _, r in basket_returns), default=0)
        if n_weeks > 0:
            total_theta = sum(theta for theta, _ in basket_returns)
            sleeves = [theta / total_theta * portfolio for theta, _ in basket_returns]
            for j in range(n_weeks):
                for k, (_, period_returns) in enumerate(basket_returns):
                    if j < len(period_returns):
                        sleeves[k] = max(0.0, sleeves[k] * (1 + period_returns[j]))
                portfolio = sum(sleeves)
                date = test_dates[j + 1] if j + 1 < len(test_dates) else test_dates[-1]
                curve.append((date, portfolio))
        else:
            curve.append((test_dates[-1], portfolio))

    return curve


def plot_equity_curve(curve, title="Walk-forward equity curve", path="equity_curve.png"):
    dates, portfolio_values = zip(*curve)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, portfolio_values, linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel("Portfolio value")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)


def evaluate_on_random_ou_spreads(n, eval_years, seed):
    # Sanity check under idealised conditions: cointegration is true by
    # construction here, since each spread IS a simulated OU process.
    rng = np.random.default_rng(seed)
    results = []
    all_period_returns = []

    for i in range(n):
        theta = rng.uniform(0.05, 0.5)
        mu = rng.uniform(-0.3, 0.3)
        sigma = rng.uniform(0.03, 0.12)

        eval_spread = simulate_ou_spread(pd.RangeIndex(eval_years * STEPS_PER_YEAR), theta, mu, sigma, rng)
        growth, period_returns, f_history = kalman_kelly_growth(eval_spread, 0, theta, sigma, mu,
                                                                  KELLY_FRACTION, MAX_POSITION_FRACTION)
        final_portfolio = STARTING_PORTFOLIO * growth
        all_period_returns.extend(period_returns)

        results.append({"simulation": i + 1, "theta": theta, "final_portfolio": final_portfolio,
                         "profitable": final_portfolio > STARTING_PORTFOLIO})

    profitable_count = sum(r["profitable"] for r in results)
    returns = np.array(all_period_returns)
    sharpe = returns.mean() / returns.std() if len(returns) > 1 and returns.std() > 0 else float("nan")
    return results, profitable_count, sharpe


if __name__ == "__main__":
    download_weekly_prices(SECURITIES_20, START_DATE, END_DATE)
    combined_csv = pd.read_csv("securities20_weekly_combined.csv", index_col=0)
    # Stock exchange timestamps carry a DST-dependent UTC offset that
    # shifts across the year (unlike crypto's constant-UTC 24/7 trading),
    # so the saved CSV headers mix offsets and need normalizing to UTC.
    combined_csv.columns = pd.to_datetime(combined_csv.columns, utc=True)

    curve = run_walk_forward_backtest(combined_csv)
    plot_equity_curve(curve)
