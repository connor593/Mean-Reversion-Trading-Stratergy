# Mean-Reversion-Trading-Stratergy

A statistical arbitrage research project that finds cointegrated baskets of equities within a
sector-mixed universe, models each basket's spread as an Ornstein-Uhlenbeck (OU) process, and
trades it with a Kalman-filtered, Kelly-sized mean-reversion strategy — backtested walk-forward,
out-of-sample, and validated against synthetic data simulated from each basket's own fitted model.

## What it does

1. **Universe** — 20 large-cap US equities: 10 energy companies and 10 banks (`SECURITIES_20`).
2. **Basket construction (PCA)** — for each training window, takes the covariance matrix of log
   prices and extracts its eigenvectors. Each eigenvector defines a candidate "basket": a linear
   combination of log prices that may be stationary (cointegrated) even though the individual
   securities aren't.
3. **OU fitting** — each candidate basket's spread is tested for stationarity with the Augmented
   Dickey-Fuller test, then fit as an AR(1) process via OLS. The AR(1) coefficients are converted
   into continuous-time OU parameters (`theta` = reversion speed, `mu` = equilibrium level, `sigma`
   = volatility) via the standard AR(1)↔OU correspondence. Baskets are kept only if their implied
   half-life falls in a plausible out-of-sample range (2–5 weeks).
4. **Kalman-filtered equilibrium** — rather than trusting a single frozen training-window estimate
   of `mu`, a scalar Kalman filter tracks the equilibrium level week-to-week during the test window,
   letting it drift slowly (`KALMAN_Q_RATIO` controls how much) while `theta`/`sigma` stay fixed at
   their training-window fit.
5. **Fractional Kelly sizing** — position size each week is the continuous-time OU-implied Kelly
   fraction (`theta*(mu - y) / sigma²`), scaled by `KELLY_FRACTION` (half-Kelly, to guard against
   estimation noise) and capped at `MAX_POSITION_FRACTION` of capital.
6. **Real per-security execution** — a basket's eigenvector weights are gross-normalized into
   per-leg dollar exposures, and each period's return is computed from the actual weighted simple
   returns of the underlying securities — not from the raw delta of the log-price combination (which
   isn't itself a tradeable return).
7. **Walk-forward backtest** — the full date range is split into 10 equal segments; each segment is
   used to fit baskets that are then traded, unseen, in the following segment (9 walk-forward
   periods total). Capital is split across a period's qualifying baskets in proportion to `theta`,
   each sleeve compounds independently, and sleeves are summed back into a single equity curve.
8. **Simulated validation** — for every qualifying basket, a synthetic path is also simulated via
   Euler-Maruyama discretization of the OU SDE, using that basket's own fitted parameters, and run
   through the identical trading logic. Comparing real vs. simulated performance isolates how much
   of the real result is explained by the fitted model versus real-world deviations from it.

## Results

From a fixed-seed run (`seed=0`) over 2014-09-15 to 2024-08-26:

| Metric | Value |
|---|---|
| Qualifying periods | 5 / 9 |
| Positive performance (of qualifying) | 4 / 5 |
| Avg. annualised Sharpe, qualifying periods | 0.73 |
| Avg. annualised Sharpe, successful periods | 1.06 |
| OU-simulated basket validation | 8 / 8 profitable |
| Avg. annualised Sharpe, simulated baskets | 2.29 |

The simulated-validation numbers depend on the RNG seed (the real-data numbers do not); results here
use `seed=0` for reproducibility, as pinned in `run_walk_forward_backtest(combined_csv, seed=0)` in
`__main__`.

## Requirements


This will:
- Download weekly closing prices for the 20-security universe via `yfinance` and cache them to CSV.
- Run the walk-forward backtest and print period-qualification / Sharpe-ratio summary statistics.
- Run the per-basket OU-simulated validation and print its summary.
- Plot and save the real-vs-simulated equity curve (`equity_curve.png`).

## Configuration

All tunable parameters are constants near the top of the script:

| Constant | Meaning |
|---|---|
| `SECURITIES_20` | The traded universe |
| `START_DATE` / `END_DATE` | Historical data range |
| `N_SEGMENTS` | Number of equal walk-forward chunks (→ `N_SEGMENTS - 1` train/test periods) |
| `MIN_OBSERVATIONS` | Minimum overlapping history required to attempt a PCA fit |
| `MIN_HALFLIFE_WEEKS` / `MAX_HALFLIFE_WEEKS` | Acceptable OU half-life range for a basket to qualify |
| `KALMAN_Q_RATIO` | Drift-noise variance for the Kalman filter, as a fraction of observation-noise variance |
| `KELLY_FRACTION` | Fraction of full Kelly sizing to use (0.5 = half-Kelly) |
| `MAX_POSITION_FRACTION` | Hard cap on position size regardless of Kelly recommendation |
| `STARTING_PORTFOLIO` | Initial capital for the backtest |

## Known limitations

- No transaction costs, bid-ask spread, or slippage are modeled.
- No discrete share-count rounding — positions are continuous dollar/return fractions.
- Short legs don't accrue borrow/financing costs or require margin.
- `theta`/`sigma` are fit once per training window and held fixed through the whole test window
  (only the equilibrium level `mu` is updated online, via the Kalman filter).
- `q_var` (the Kalman filter's drift-noise variance) is not estimated from data — it's fixed as a
  hand-picked fraction (`KALMAN_Q_RATIO = 0.01`) of the OU-implied observation-noise variance, since
  the true equilibrium level is never directly observed to estimate its own variance from.
- Universe is small (20 securities) and sector-concentrated (energy + banks); results may not
  generalize to other sectors or a larger universe.

## Project structure

Single-file script (`OU Mean Reversion.py`):

- `download_weekly_prices`, `segment_dates` — data acquisition and walk-forward windowing.
- `fit_stationary_ar1`, `select_cointegrated_baskets` — ADF/AR(1)-based OU parameter fitting and
  basket selection via PCA.
- `simulate_ou_spread` — Euler-Maruyama OU path simulation.
- `continuous_kelly`, `kalman_filter_mu`, `kalman_kelly_growth` — signal generation, online mean
  estimation, and per-step position sizing / return calculation.
- `_basket_period_returns`, `_compound_sleeves`, `run_walk_forward_backtest` — walk-forward backtest
  orchestration (real and simulated).
- `_annualized_sharpe`, `summarize_windows`, `summarize_basket_simulations` — performance reporting.
- `plot_equity_curve` — visualization.
