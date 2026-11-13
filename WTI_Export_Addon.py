"""Append these exports to WTI_Independent_ML_Factors.ipynb.

Keep this file beside the Notebook, then run ONE new cell after sections10-11:
    %run -i "WTI_Export_Addon.py"

Existing kernel tables are reused. After a kernel restart, set RESULTS_DIR to
its completed WTI_Independent_ML_Results folder first. To explicitly ignore
any stale kernel tables and use saved files, set EXPORT_FROM_DISK = True.
No model fitting, forecasting or trading-position calculation is performed.

The calendar-cut forecast ends TEST_START +60 calendar days (2026-11-30).
The complete60-quote-step file preserves the producer's stored forecast dates
(2026-12-28 for this supplied calendar), anchored before2026-10-01.
Annual metrics reuse original net equity, fees and cash dates. CSV return,
volatility and drawdown values are decimal fractions; drawdown is positive.
Annualization convention: CAGR; daily excess-return Sharpe; sample volatility.
Reference: https://quantopian.github.io/empyrical/appendix.html
"""
# %% Five-model CSV exports: existing predictions only
"""Export existing final-five forecasts only; never fit or extend a model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def export_five_model_forecasts(ns, outdir):
    """Return output paths, reusing current-kernel tables or the saved run."""
    supplied_cfg = ns.get("cfg", {})
    default = "WTI_Independent_ML_Smoke_Results" if supplied_cfg.get("RUN_MODE") == "smoke" else "WTI_Independent_ML_Results"
    source = Path(ns.get("RESULTS_DIR", ns.get("OUTPUT_DIR", Path.cwd() / default)))
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    config_file = source / "run_config.json"
    saved_cfg = json.loads(config_file.read_text(encoding="utf-8-sig")) if config_file.exists() else {}
    config = {**saved_cfg, **supplied_cfg}
    validation_start = pd.Timestamp(config.get("VAL_START", "2024-07-31")).normalize()
    forecast_start = pd.Timestamp(config.get("TEST_START", "2026-10-01")).normalize()
    count = int(config.get("H", 60))
    calendar_cut = forecast_start + pd.Timedelta(days=60)
    frame = ns.get("all_predictions")
    if not isinstance(frame, pd.DataFrame):
        prediction_file = source / "all_predictions_with_combinations.csv"
        if not prediction_file.exists():
            raise FileNotFoundError("Keep all_predictions in the kernel, or set RESULTS_DIR to the completed run containing all_predictions_with_combinations.csv. No model is rerun.")
        frame = pd.read_csv(prediction_file)
    frame = frame.copy()
    required = {"model", "origin_date", "target_date", "horizon", "pred", "actual"}
    if not required.issubset(frame.columns):
        raise ValueError("Saved predictions lack columns: " + repr(sorted(required - set(frame.columns))))
    for column in ("origin_date", "target_date"):
        frame[column] = pd.to_datetime(frame[column], errors="raise").dt.normalize()
    frame["horizon"] = pd.to_numeric(frame.horizon, errors="raise").astype(int)
    for column in ("pred", "actual"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    models = ns.get("final_candidates")
    if models is None:
        selection_file = source / "selected_models.json"
        selection = ns.get("summary", {})
        selection = selection if isinstance(selection, dict) else {}
        if selection_file.exists():
            selection = {**json.loads(selection_file.read_text(encoding="utf-8-sig")), **selection}
        var_name = ns.get("var_best", selection.get("VAR_window"))
        hybrid_name = ns.get("hybrid_best", selection.get("Hybrid_window"))
        models = [var_name, hybrid_name, "Equal Weight", "DMSFE", "RW"]
    models = list(models)
    if len(models) != 5 or len(set(models)) != 5 or any(name is None for name in models):
        raise ValueError("Provide final_candidates with the five final model names, or keep selected_models.json in RESULTS_DIR.")
    missing = set(models) - set(frame.model)
    if missing:
        raise ValueError("Final-five predictions unavailable: " + repr(sorted(missing)) + ". Run/load sections 9–10 outputs; this exporter does not create missing forecasts.")
    frame = frame.loc[frame.model.isin(models)].copy()
    keys = ["model", "origin_date", "horizon"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate model/origin/horizon records; choose one completed result set.")
    first_step = frame.loc[frame.horizon.eq(1) & frame.target_date.eq(forecast_start) & frame.origin_date.lt(forecast_start)]
    eligible_origins = first_step.groupby("origin_date").model.nunique()
    eligible_origins = eligible_origins[eligible_origins.eq(5)]
    if eligible_origins.empty:
        raise ValueError("No common saved origin has h=1 on " + str(forecast_start.date()) + ". The October-1 anchored path cannot be replaced by the latest-origin path without changing the forecast.")
    origin = eligible_origins.index.max()
    anchored = frame.loc[frame.origin_date.eq(origin) & frame.horizon.between(1, count)].copy()
    expected = set(range(1, count + 1))
    for name in models:
        part = anchored.loc[anchored.model.eq(name)]
        if len(part) != count or set(part.horizon) != expected:
            raise ValueError(f"Incomplete saved {count}-step path for {name} at {origin.date()}; no refitting or synthetic forecast filling is performed.")
    if anchored.groupby("horizon").target_date.nunique().ne(1).any():
        raise ValueError("The five models have inconsistent target calendars.")
    path_dates = anchored.loc[anchored.model.eq(models[0])].sort_values("horizon").target_date
    if path_dates.duplicated().any() or not path_dates.is_monotonic_increasing or path_dates.iloc[0] != forecast_start:
        raise ValueError("Stored anchored dates are not the expected increasing forecast calendar.")
    end_date = path_dates.iloc[-1]
    actual = frame.groupby("target_date", sort=True).actual.first()
    panel = ns.get("panel")
    if not isinstance(panel, pd.DataFrame) and (source / "00_data" / "regimes.csv").exists():
        panel = pd.read_csv(source / "00_data" / "regimes.csv", usecols=["date", "price"], parse_dates=["date"]).set_index("date")
    if isinstance(panel, pd.DataFrame) and "price" in panel:
        observed = panel.price.copy()
        observed.index = pd.to_datetime(observed.index).normalize()
        common = actual.dropna().index.intersection(observed.dropna().index)
        if not np.allclose(actual.loc[common], observed.loc[common], rtol=1e-10, atol=1e-10):
            raise ValueError("Saved forecasts and supplied/saved observed prices differ; choose matching RESULTS_DIR and kernel results.")
        actual = observed.combine_first(actual).sort_index()
    actual = pd.to_numeric(actual, errors="coerce")
    files = {}

    def write(name, data):
        path = destination / name
        data.to_csv(path, index=False, encoding="utf-8-sig")
        files[name] = str(path.resolve())

    anchor_values = anchored.pivot(index="target_date", columns="model", values="pred").reindex(columns=models)
    anchored_wide = anchor_values.copy()
    anchored_wide.insert(0, "forecast_status", np.where(anchored_wide[models].notna().all(axis=1), "complete", "missing saved model prediction; not filled"))
    anchored_wide.insert(0, "Actual WTI", actual.reindex(anchored_wide.index))
    anchored_wide.insert(0, "origin_date", origin)
    anchored_wide.insert(0, "horizon", range(1, count + 1))
    anchored_wide.index.name = "target_date"
    write("five_models_oct01_origin_60_quote_steps.csv", anchored_wide.reset_index())
    history = frame.loc[frame.horizon.eq(1) & frame.target_date.ge(validation_start) & frame.target_date.lt(forecast_start)].copy()
    if history.duplicated(["model", "target_date"]).any():
        raise ValueError("Multiple h=1 forecasts per model/target; a unique walk-forward path is required.")
    hist_values = history.pivot(index="target_date", columns="model", values="pred").reindex(columns=models)
    historical_dates = actual.index[(actual.index >= validation_start) & (actual.index < forecast_start)]
    grid = pd.DatetimeIndex([validation_start]).union(historical_dates).union(hist_values.index).union(anchor_values.index).sort_values()
    combined = pd.concat([hist_values, anchor_values]).reindex(index=grid, columns=models)
    combined.insert(0, "Actual WTI", actual.reindex(grid))
    hist_origins = history.groupby("target_date").origin_date.first()
    combined.insert(0, "origin_date", pd.Series(origin, index=grid).where(grid >= forecast_start, hist_origins.reindex(grid)))
    combined.insert(0, "forecast_type", np.where(grid < forecast_start, "walk-forward h=1", "fixed pre-October origin"))
    combined.insert(0, "forecast_status", np.where(combined[models].notna().all(axis=1), "complete", "missing saved model prediction; not filled"))
    combined.index.name = "target_date"
    combined = combined.loc[validation_start:end_date].reset_index()
    write("five_models_validation_to_60_quote_steps.csv", combined)
    write("five_models_validation_to_oct01_plus_60_calendar_days.csv", combined.loc[combined.target_date.le(calendar_cut)])
    long_table = frame.loc[frame.target_date.between(validation_start, end_date) & frame.horizon.between(1, count)].sort_values(["target_date", "origin_date", "horizon", "model"])
    write("five_models_all_origins_all_horizons_long.csv", long_table)
    information = {"models": models, "validation_start": str(validation_start.date()), "fixed_origin": str(origin.date()),
                   "forecast_start": str(forecast_start.date()), "forecast_steps": count, "last_quote_step_date": str(end_date.date()),
                   "calendar_cut_date": str(calendar_cut.date()), "missing_wide_rows": int(combined.forecast_status.ne("complete").sum()),
                   "calendar_policy": "Dates copied from saved forecasts; future dates retain the original projected US federal business calendar, not an actual quote-calendar guarantee.",
                   "series_policy": "Before October: existing walk-forward h=1. From October: fixed pre-October-origin full path. Long CSV preserves every origin/horizon. Missing forecasts and unknown actuals stay NaN.",
                   "models_refit": False, "source_directory": str(source.resolve())}
    metadata_file = destination / "forecast_export_metadata.json"
    metadata_file.write_text(json.dumps(information, indent=2), encoding="utf-8")
    files[metadata_file.name] = str(metadata_file.resolve())
    print(f"Five-model export: origin {origin.date()}, h=1 {forecast_start.date()}, h={count} {end_date.date()}.")
    print(f"Calendar-cut copy ends {calendar_cut.date()}; no weekend rows or missing predictions are manufactured.")
    return {"files": files, "metadata": information, "forecast_series": combined, "anchored_path": anchored_wide.reset_index()}


# %% Four annual trading metrics: existing net equity only
"""Append-only annual trading metrics from existing net-equity results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def export_annual_trading_metrics(ns, outdir):
    """Reuse existing positions/fees; never fit models or rerun trades.

    Return/volatility/drawdown CSV values are decimal fractions. Drawdown is
    a positive loss magnitude. Annualization uses 252 quote-return observations
    by default, including the original first-day entry fee and cash dates.
    """
    previous = ns.get("trade_results", {}) or {}
    configured = ns.get("RESULTS_DIR") or ns.get("OUTPUT_DIR")
    if configured:
        base = Path(configured)
    elif previous.get("daily") is not None and previous.get("metrics") is not None:
        base = Path(outdir).parent
    else:
        candidates = [Path.cwd() / name for name in ("WTI_Independent_ML_Results", "WTI_Independent_ML_Smoke_Results")]
        existing = [path for path in candidates if path.is_dir()]
        if len(existing) != 1:
            raise FileNotFoundError("Set RESULTS_DIR to the existing results folder; no model rerun is needed.")
        base = existing[0]
    saved = json.loads((base / "run_config.json").read_text()) if (base / "run_config.json").exists() else {}
    cfg = {**saved, **ns.get("cfg", {})}
    daily = previous.get("daily")
    stats = previous.get("metrics")
    if ((daily is None and not (base / "07_trading" / "daily_mark_to_market.csv").exists())
            or (stats is None and not (base / "07_trading" / "strategy_metrics_all_horizons.csv").exists())):
        raise FileNotFoundError("Trading results are missing. Point RESULTS_DIR to the correct folder, or run Section 11 only using existing predictions.")
    if daily is None:
        daily = pd.read_csv(base / "07_trading" / "daily_mark_to_market.csv")
    if stats is None:
        stats = pd.read_csv(base / "07_trading" / "strategy_metrics_all_horizons.csv")
    daily, stats = daily.copy(), stats.copy()
    keys = ["horizon", "strategy", "regime"]
    if not set(keys + ["date", "equity"]).issubset(daily.columns):
        raise ValueError("Trading daily results require horizon/strategy/regime/date/equity columns.")
    if not set(keys + ["N_intervals", "N_active_trades"]).issubset(stats.columns):
        raise ValueError("Trading metrics require interval and active-trade counts to distinguish no data from cash.")
    if stats.duplicated(keys).any():
        raise ValueError("Duplicate trading groups: use strategy_metrics_all_horizons.csv, not cost sensitivity.")
    periods = float(ns.get("ANNUALIZATION_FACTOR", cfg.get("TRADING_PERIODS_PER_YEAR", 252)))
    rf = float(ns.get("RISK_FREE_ANNUAL", cfg.get("RISK_FREE_ANNUAL", 0.0)))
    if periods <= 0 or not np.isfinite(periods) or not np.isfinite(rf) or rf <= -1:
        raise ValueError("Annualization factor must be positive and annual risk-free return must exceed -1.")
    rf_daily = (1.0 + rf) ** (1.0 / periods) - 1.0
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    groups = {key: part.sort_values("date") for key, part in daily.groupby(keys, dropna=False)}
    columns = ["Annual Return", "Annual Sharpe Ratio", "Annual Vol", "Max Drawdown"]
    rows = []
    for source in stats.to_dict("records"):
        key = tuple(source[k] for k in keys)
        part = groups.get(key)
        row = {k: source[k] for k in keys + ["N_intervals", "N_active_trades"]}
        row.update({column: np.nan for column in columns})
        row.update(N_return_observations=0, period_start=pd.NaT, period_end=pd.NaT,
                   annualization_factor=periods, risk_free_annual=rf,
                   one_way_cost_bps=source.get("one_way_cost_bps", np.nan),
                   status=str(source.get("status", "")), annualization_note="")
        if part is not None and len(part):
            if part.date.duplicated().any():
                raise ValueError(f"Duplicate quote dates in trading group {key}.")
            equity = pd.to_numeric(part.equity, errors="raise").to_numpy(dtype=float)
            if not np.isfinite(equity).all() or (equity <= 0).any():
                raise ValueError(f"Nonpositive/nonfinite net equity in trading group {key}.")
            # The original book starts with capital 1 before the first entry
            # cost. Dividing final equity by its first stored value loses it.
            returns = np.r_[equity[0] - 1.0, equity[1:] / equity[:-1] - 1.0]
            if "daily_return" in part:
                recorded = pd.to_numeric(part.daily_return, errors="raise").to_numpy(dtype=float)
                if not np.isfinite(recorded).all() or not np.allclose(recorded, returns, rtol=1e-7, atol=1e-10):
                    raise ValueError(f"Daily returns disagree with net equity/initial capital in group {key}.")
            if not np.allclose(np.cumprod(1.0 + returns), equity, rtol=1e-7, atol=1e-10):
                raise ValueError(f"Equity compounding validation failed in group {key}.")
            n = len(returns)
            row.update(N_return_observations=n, period_start=part.date.iloc[0], period_end=part.date.iloc[-1])
            row["annualization_note"] = "Approximate quote-return annualization; short sample" if n < 252 else "Quote-return annualization"
            if int(source["N_intervals"]) > 0:
                sd = float(np.std(returns, ddof=1)) if n >= 2 else np.nan
                excess = returns - rf_daily
                excess_sd = float(np.std(excess, ddof=1)) if n >= 2 else np.nan
                with np.errstate(over="ignore"):
                    row["Annual Return"] = float(np.expm1(np.log1p(returns).sum() * periods / n))
                row["Annual Vol"] = sd * np.sqrt(periods)
                row["Annual Sharpe Ratio"] = float(np.sqrt(periods) * np.mean(excess) / excess_sd) if excess_sd > 1e-12 else np.nan
                peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
                row["Max Drawdown"] = float(np.max(1.0 - equity / peak))
        elif int(source["N_intervals"]) > 0:
            raise ValueError(f"Missing net-equity observations for eligible trading group {key}.")
        if int(source["N_intervals"]) == 0:
            row["annualization_note"] = "No complete eligible trading interval; no performance estimate"
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)
    primary_h = int(cfg.get("TRADE_H", 5))
    primary = table.loc[table.horizon.eq(primary_h)].set_index(["strategy", "regime"])[columns]
    overall = table.loc[table.horizon.eq(primary_h) & table.regime.eq("Overall")].set_index("strategy")[columns]
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    files = {"all_horizons":out / "trading_annual_metrics_all_horizons.csv",
             "primary_horizon":out / f"trading_annual_metrics_h{primary_h}.csv"}
    table.to_csv(files["all_horizons"], index=False, encoding="utf-8-sig")
    primary.to_csv(files["primary_horizon"], encoding="utf-8-sig")
    print(f"Trading metrics | h={primary_h} | net of original costs | {periods:g} quotes/year | annual rf={rf:.2%}")
    print("Return, volatility and drawdown are percentages below; CSVs store decimal fractions. Drawdown is a positive magnitude.")
    from IPython.display import display
    shown = overall.copy()
    for column in columns:
        template = "{:.3f}" if column == "Annual Sharpe Ratio" else "{:.2%}"
        shown[column] = shown[column].map(lambda value: template.format(value) if pd.notna(value) else "N/A")
    display(shown)
    return {"all":table, "primary":primary, "overall":overall, "files":{k:str(v) for k,v in files.items()}}


# %% Run the two append-only exports
if __name__ == "__main__":
    RESULTS_DIR = Path(globals().get("RESULTS_DIR") or globals().get("OUTPUT_DIR")
                       or (Path.cwd() / "WTI_Independent_ML_Results"))
    ADDON_OUTPUT_DIR = Path(globals().get("ADDON_OUTPUT_DIR")
                           or (RESULTS_DIR / "08_additional_exports"))
    if bool(globals().get("EXPORT_FROM_DISK", False)):
        _export_context = {"RESULTS_DIR":RESULTS_DIR}
        for _key in ("ANNUALIZATION_FACTOR", "RISK_FREE_ANNUAL"):
            if _key in globals():
                _export_context[_key] = globals()[_key]
    else:
        _export_context = dict(globals())
        _export_context["RESULTS_DIR"] = RESULTS_DIR
    additional_forecast_exports = export_five_model_forecasts(_export_context, ADDON_OUTPUT_DIR)
    additional_trading_metrics = export_annual_trading_metrics(_export_context, ADDON_OUTPUT_DIR)
    _manifest = {
        "forecast_input": "kernel all_predictions" if isinstance(_export_context.get("all_predictions"), pd.DataFrame) else "saved CSV",
        "trading_input": "kernel trade_results" if isinstance(_export_context.get("trade_results"), dict) else "saved CSV",
        "source_results_directory":str(RESULTS_DIR.resolve()),
        "forecast":additional_forecast_exports["metadata"],
        "trading_scope":"Existing realized trading calendar only; future forecast prices are never treated as realized returns.",
        "annualization":"Approximate252-quote convention by default; short-sample counts are exported.",
        "forecast_files":additional_forecast_exports["files"],
        "trading_files":additional_trading_metrics["files"],
        "models_rerun":False,
    }
    (ADDON_OUTPUT_DIR / "additional_export_manifest.json").write_text(json.dumps(_manifest,indent=2),encoding="utf-8")
    print("\nAdditional CSV files:", ADDON_OUTPUT_DIR.resolve())
    for _file in sorted(ADDON_OUTPUT_DIR.glob("*.csv")):
        print(" -", _file.name)
