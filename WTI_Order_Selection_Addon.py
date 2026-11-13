"""Insertable order selection for WTI_Independent_ML_Factors.ipynb.

Run the file once with %run -i to define the functions; it fits nothing on import.
Stage A: after Section 2, BEFORE make_factor_inputs in Section 3.
    var_order_selection = select_wti_var_order(globals())
Stage B: after Section 4, BEFORE fit_origins in Section 5.
    arimax_order_selection = select_wti_arimax_order(globals())

Selection uses training data only. Both windows share one selected order.
The declared score averages per-observation BIC across the two overlapping
windows. It is an aggregate selection rule, not their joint BIC or a guarantee
of the best forecast RMSE. AIC is reported as a reference, not used to select.
VAR p is selected on the no-factor backbone, and VARX shares it for ablation.
The original reference-VAR docstring says VAR(2); its executed code reads
cfg['VAR_LAG'], so Stage A's selected p takes precedence over that old text.
ARIMAX (p,0,q) is selected with the actual historical cross-fitted X_ARIMAX;
the no-X ARIMA shares that order. d=0 refers to already differenced WTI.
GARCH is not refitted during mean-order selection. Old forecasts are not updated.

Primary implementation references:
https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.var_model.VARResults.html
https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.info_criteria.html
"""

from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd


def _order_context(ns, output_dir=None):
    config = ns.get('cfg')
    panel = ns.get('panel')
    if not isinstance(config, dict) or not isinstance(panel, pd.DataFrame):
        raise ValueError('Run the existing Notebook through Section 2 first: cfg and panel are required.')
    if (not isinstance(panel.index, pd.DatetimeIndex) or not panel.index.is_unique
            or not panel.index.is_monotonic_increasing):
        raise ValueError('panel must have a unique, increasing DatetimeIndex.')
    if 'price' not in panel:
        raise ValueError('Missing WTI price column.')
    root = Path(output_dir) if output_dir is not None else Path(ns.get('OUTPUT_DIR', 'WTI_Independent_ML_Results')) / '00_order_selection'
    root.mkdir(parents=True, exist_ok=True)
    return config, panel, root


def _order_export(report, root, prefix):
    paths = {}
    for name in ('candidates', 'summary'):
        path = root / f'{prefix}_{name}.csv'
        report[name].to_csv(path, index=False, encoding='utf-8-sig')
        paths[name] = str(path)
    path = root / f'{prefix}_selection.json'
    path.write_text(json.dumps(report['metadata'], indent=2, allow_nan=False), encoding='utf-8')
    paths['selection'] = str(path)
    report['files'] = paths
    try:
        from IPython.display import display
        display(report['summary'])
    except ImportError:
        print(report['summary'].to_string(index=False))
    print('Order-selection tables:', root.resolve())


def select_wti_var_order(ns, output_dir=None, apply=True):
    """Stage A; changing p must precede residual labels and all ML factor fits."""
    config, panel, root = _order_context(ns, output_dir)
    if apply and (config.get('_FACTOR_STORE') is not None or
                  any(ns.get(key) is not None for key in ('factor_targets', 'factor_store', 'predictions'))):
        raise RuntimeError('Stage A must run in a fresh kernel after Section 2 and before Section 3. '
                           'Use apply=False only to inspect a recommendation in an already-run kernel. '
                           'A new VAR order requires regenerating reference labels, ML factors and forecasts.')
    try:
        report = score_var_orders(panel, config)
    except VarOrderSelectionError as error:
        error.report['metadata']['applied'] = False
        _order_export(error.report, root, 'VAR')
        raise
    chosen = int(report['chosen_p'])
    report['metadata']['applied'] = bool(apply)
    report['metadata']['previous_VAR_LAG'] = int(config.get('VAR_LAG', 2))
    if apply:
        config['VAR_LAG'] = chosen
        config['COMMON_BURN'] = max(int(config.get('COMMON_BURN', 2)), chosen)
        config['ORDER_SELECTION_VAR'] = dict(report['metadata'])
    _order_export(report, root, 'VAR')
    print(f'Selected VAR({chosen}); VARX and its causal reference-VAR labels use the same p.')
    print('Applied to cfg.' if apply else 'Recommendation only; cfg and existing forecasts are unchanged.')
    return report


def _order_nonnegative_grid(config, name, default):
    try:
        raw = list(config.get(name, default))
        if not raw or any(isinstance(value, (bool, np.bool_)) or not np.isfinite(float(value))
                          or int(value) != float(value) or int(value) < 0 for value in raw):
            raise ValueError
        values = sorted(set(int(value) for value in raw))
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must contain nonnegative integer orders.') from None
    return values


def _arimax_selection_sample(ns, config, panel):
    """Keep every selected training quote; never drop missing X rows silently."""
    horizon = int(config.get('H', 60))
    first_validation = int(panel.index.searchsorted(pd.Timestamp(config['VAL_START'])))
    safe_position = first_validation - horizon - 1
    if horizon < 1 or safe_position < 1:
        raise ValueError('Insufficient pre-validation history for the declared forecast horizon.')
    safe_cutoff = panel.index[safe_position]
    cutoff = pd.Timestamp(config.get('ORDER_ARIMAX_END', safe_cutoff))
    if cutoff > safe_cutoff:
        raise ValueError(f'ORDER_ARIMAX_END must be <= {safe_cutoff.date()}, before the earliest possible validation origin.')
    if not config.get('MEAN_START'):
        raise ValueError('Run Section 4 first so MEAN_START and cross-fitted X_ARIMAX exist.')
    history = panel.loc[panel.index <= cutoff]
    changes = history.price.diff().iloc[1:]
    changes = changes.iloc[int(config.get('COMMON_BURN', config.get('VAR_LAG', 2))):]
    changes = changes.loc[changes.index >= pd.Timestamp(config['MEAN_START'])]
    if not len(changes) or not np.isfinite(changes.to_numpy(float)).all():
        raise ValueError('Pre-validation WTI changes are empty or nonfinite.')
    store = config.get('_FACTOR_STORE')
    if not isinstance(store, dict):
        store = ns.get('factor_store')
    if not isinstance(store, dict) or 'oof' not in store:
        raise ValueError('Run Section 4 first; actual cross-fitted factor_store is required.')
    rolling = int(config.get('ROLLING_WINDOW', 500))
    if rolling < 100 or len(changes) < rolling:
        raise ValueError(f'Need at least {rolling} complete pre-validation mean targets.')
    samples = {}
    digest = hashlib.sha256()
    for label, window in ((f'Rolling{rolling}', rolling), ('Expanding', None)):
        # The production Notebook currently names its rolling factor Rolling500.
        factor_key = ('Hybrid', 'Rolling500' if window is not None else 'Expanding')
        frame = store['oof'].get(factor_key)
        if not isinstance(frame, pd.DataFrame) or not frame.index.is_unique:
            raise ValueError(f'Missing unique cross-fitted factor history: {factor_key}.')
        needed = {'factor', 'fit_last_target', 'last_feature_date'}
        if not needed.issubset(frame.columns):
            raise ValueError(f'Factor provenance lacks {sorted(needed-set(frame.columns))}.')
        y = changes.iloc[-window:] if window is not None else changes
        proof = frame.reindex(y.index)
        x = pd.to_numeric(proof.factor, errors='coerce').to_numpy(float)
        fit_dates = pd.to_datetime(proof.fit_last_target)
        feature_dates = pd.to_datetime(proof.last_feature_date)
        if (not np.isfinite(x).all() or fit_dates.isna().any() or feature_dates.isna().any()
                or not (fit_dates.to_numpy() < y.index.to_numpy()).all()
                or not (feature_dates.to_numpy() < y.index.to_numpy()).all()):
            raise ValueError(f'{label}: missing or noncausal historical X_ARIMAX rows; nothing is imputed or dropped.')
        yscale = float(y.std(ddof=1))
        xcenter, xscale = float(x.mean()), float(x.std(ddof=1))
        if min(yscale, xscale) < 1e-10 or not np.isfinite([yscale, xcenter, xscale]).all():
            raise ValueError(f'{label}: constant or invalid WTI changes/X_ARIMAX.')
        samples[label] = dict(y=y.to_numpy(float)/yscale, x=((x-xcenter)/xscale)[:, None],
                              dates=y.index, y_scale=yscale, X_center=xcenter, X_scale=xscale)
        hash_frame = pd.DataFrame({'dWTI': y, 'X_ARIMAX': x, 'fit_last_target': fit_dates,
                                   'last_feature_date': feature_dates}, index=y.index)
        digest.update(pd.util.hash_pandas_object(hash_frame, index=True).values.tobytes())
    return samples, cutoff, safe_cutoff, digest.hexdigest()


def select_wti_arimax_order(ns, output_dir=None, apply=True):
    """Stage B: actual generated-X model, matched dates, fixed d=0 on dWTI."""
    config, panel, root = _order_context(ns, output_dir)
    if apply and ns.get('predictions') is not None:
        raise RuntimeError('Run Stage B after Section 4 and before Section 5 in a fresh execution. '
                           'Use apply=False to inspect only; new orders cannot update already-generated forecasts.')
    fitter = ns.get('_fit_sarimax_checked')
    if not callable(fitter):
        raise ValueError('Run the original Section 2 implementation cells first.')
    samples, cutoff, safe_cutoff, signature = _arimax_selection_sample(ns, config, panel)
    p_grid = _order_nonnegative_grid(config, 'ORDER_AR_P', range(4))
    q_grid = _order_nonnegative_grid(config, 'ORDER_MA_Q', range(4))
    rows = []
    total = len(p_grid)*len(q_grid)*len(samples)
    done = 0
    for p in p_grid:
        for q in q_grid:
            for label, sample in samples.items():
                started = time.perf_counter()
                row = dict(p=p, d=0, q=q, window=label, n_train=len(sample['y']),
                           train_start=str(sample['dates'][0].date()), train_end=str(sample['dates'][-1].date()),
                           y_scale=sample['y_scale'], X_center=sample['X_center'], X_scale=sample['X_scale'],
                           status='failed', error='', BIC=np.nan, AIC=np.nan, BIC_per_observation=np.nan)
                try:
                    candidate_config = dict(config, ARIMAX_ORDER=[p, 0, q])
                    fitted, messages, attempts = fitter(sample['y'], sample['x'], candidate_config)
                    n_effective = float(fitted.nobs_effective)
                    if n_effective != len(sample['y']):
                        raise ValueError('Candidate changed the effective likelihood sample.')
                    if not bool(fitted.mle_retvals.get('converged', False)):
                        raise ValueError('Candidate optimizer did not converge.')
                    ar_roots = np.abs(np.asarray(fitted.arroots))
                    ma_roots = np.abs(np.asarray(fitted.maroots))
                    if not (np.isfinite(np.asarray(fitted.params)).all() and np.isfinite([fitted.bic, fitted.aic, fitted.llf]).all()):
                        raise ValueError('Candidate returned nonfinite parameters or information criteria.')
                    if not (np.all(ar_roots > 1) and np.all(ma_roots > 1)):
                        raise ValueError('Candidate fails the strict AR or MA root condition.')
                    row.update(status='ok', BIC=float(fitted.bic), AIC=float(fitted.aic),
                               BIC_per_observation=float(fitted.bic/n_effective),
                               AIC_per_observation=float(fitted.aic/n_effective),
                               loglikelihood=float(fitted.llf), n_effective=int(n_effective),
                               parameter_count=int(len(fitted.params)),
                               minimum_AR_root_modulus=float(ar_roots.min()) if ar_roots.size else np.nan,
                               minimum_MA_root_modulus=float(ma_roots.min()) if ma_roots.size else np.nan,
                               optimizer_attempts=json.dumps(attempts), warnings=json.dumps(messages))
                except Exception as error:
                    row['error'] = f'{type(error).__name__}: {error}'
                row['seconds'] = time.perf_counter()-started
                rows.append(row)
                done += 1
                print(f'ARIMAX order scan {done}/{total}: ({p},0,{q}) | {label} | {row["status"]}', flush=True)
    candidates = pd.DataFrame(rows)
    summary_rows = []
    for (p, q), group in candidates.groupby(['p', 'q'], sort=True):
        eligible = len(group) == len(samples) and group.status.eq('ok').all()
        summary_rows.append(dict(p=int(p), d=0, q=int(q), eligible=bool(eligible),
                                 mean_BIC_per_observation=float(group.BIC_per_observation.mean()) if eligible else np.nan,
                                 complexity=int(p+q), successful_windows=int(group.status.eq('ok').sum()),
                                 reason='eligible' if eligible else 'one or more window fits failed'))
    summary = pd.DataFrame(summary_rows).sort_values(
        ['eligible', 'mean_BIC_per_observation', 'complexity', 'p', 'q'], ascending=[False, True, True, True, True],
        na_position='last').reset_index(drop=True)
    summary['rank'] = pd.array([i+1 if eligible else None for i, eligible in enumerate(summary.eligible)], dtype='Int64')
    good = summary.loc[summary.eligible]
    metadata = dict(stage='ARIMAX order selection', cutoff=str(cutoff.date()), safe_latest_cutoff=str(safe_cutoff.date()),
                    p_candidates=p_grid, q_candidates=q_grid, d=0, y='WTI dollar changes', X='actual historical forward-cross-fitted X_ARIMAX',
                    criterion='equally weighted mean(BIC/n_effective) across Rolling and Expanding; lower is better',
                    criterion_scope='shared-order aggregate score, not a joint BIC or a forecast-accuracy claim',
                    sample_signature=signature, selection_uses_main_validation_or_test=False,
                    baseline='No-X ARIMA shares the selected order for matched ablation.',
                    applied=bool(apply), previous_ARIMAX_ORDER=list(config.get('ARIMAX_ORDER', [1,0,1])))
    report = dict(candidates=candidates, summary=summary, metadata=metadata)
    if good.empty:
        metadata['status'] = 'no eligible order; configuration unchanged'
        metadata['applied'] = False
        _order_export(report, root, 'ARIMAX')
        raise RuntimeError('No ARIMAX candidate passed both window fits; inspect ARIMAX_candidates.csv. No fallback order was applied.')
    order = [int(good.iloc[0].p), 0, int(good.iloc[0].q)]
    metadata.update(status='selected', selected_order=order)
    report['chosen_order'] = order
    if apply:
        config['ARIMAX_ORDER'] = order
        config['ORDER_SELECTION_ARIMAX'] = dict(metadata)
        if callable(ns.get('describe_mean_specifications')):
            ns['describe_mean_specifications'](config).to_csv(root.parent/'model_specifications.csv', index=False, encoding='utf-8-sig')
    _order_export(report, root, 'ARIMAX')
    print(f'Selected ARIMAX{tuple(order)} on WTI changes + X_ARIMAX; the no-X ARIMA uses the same order.')
    print('Applied to cfg.' if apply else 'Recommendation only; cfg and existing forecasts are unchanged.')
    return report


"""Matched-calendar VAR order comparison inside the initial development prefix.

The shared-order score averages the VAR BIC criterion across Rolling and
Expanding fits. It is an aggregate selection rule, not the BIC of one pooled
likelihood and not a demonstration of superior forecast accuracy.
"""

import numpy as np
import pandas as pd


class VarOrderSelectionError(RuntimeError):
    """No eligible order; retain the comparison tables for the caller to export."""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def score_var_orders(panel, cfg):
    """Score fixed candidate lags on identical training targets in each window.

    Only WTI and Brent levels dated on/before ML_SELECTION_END are read. Both
    endogenous series are differenced once before fitting intercept VARs.
    The candidate must fit successfully and have a companion spectral radius
    strictly below one in both windows. This function does not mutate cfg.
    """
    if not isinstance(panel, pd.DataFrame) or not {"price", "F001"}.issubset(panel.columns):
        raise ValueError("VAR order selection requires panel price and Brent F001 columns")
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise ValueError("VAR order selection requires a DatetimeIndex")
    if panel.index.hasnans or not panel.index.is_unique or not panel.index.is_monotonic_increasing:
        raise ValueError("Quote dates must be nonmissing, unique and increasing")
    cutoff = pd.Timestamp(cfg["ML_SELECTION_END"])
    if pd.isna(cutoff) or not cutoff < pd.Timestamp(cfg["VAL_START"]):
        raise ValueError("VAR order-selection cutoff must precede the main validation period")
    requested = cfg.get("ORDER_VAR_P", list(range(1, 11)))
    try:
        requested = list(requested)
        if not requested or any(isinstance(x, (bool, np.bool_)) or not np.isfinite(float(x))
                                or int(x) != float(x) or int(x) < 1 for x in requested):
            raise ValueError
        orders = sorted(set(int(x) for x in requested))
    except (TypeError, ValueError, OverflowError):
        raise ValueError("ORDER_VAR_P must contain positive integer lag orders") from None
    rolling = int(cfg.get("ROLLING_WINDOW", 500))
    minimum = int(cfg.get("ORDER_MIN_TRAIN", 100))
    burn = int(cfg.get("COMMON_BURN", 2))
    if rolling < minimum or minimum < 25 or burn < 0:
        raise ValueError("Invalid rolling length, minimum sample size or common burn")
    levels = panel.loc[panel.index <= cutoff, ["price", "F001"]].astype(float)
    if levels.empty or not np.isfinite(levels.to_numpy()).all():
        raise ValueError("Development WTI/Brent levels must all be finite; no dates are dropped")
    delta = levels.diff().to_numpy()
    first = max(max(orders), burn) + 1
    positions = np.arange(first, len(levels), dtype=int)
    if len(positions) < rolling:
        raise ValueError("Not enough common development targets for the declared rolling window")
    metadata = {
        "selection_cutoff": str(cutoff.date()),
        "last_observed_level": str(levels.index[-1].date()),
        "candidate_orders": orders,
        "endogenous_series": ["WTI one-quote dollar change", "Brent one-quote dollar change"],
        "common_target_start": str(levels.index[first].date()),
        "common_target_end": str(levels.index[-1].date()),
        "common_expanding_targets": int(len(positions)),
        "rolling_targets": rolling,
        "criterion": "mean of per-window VAR BIC: logdet(SSE/n) + log(n)*k*(1+k*p)/n",
        "criterion_scope": "Equal-window aggregate, not a pooled-likelihood BIC or a forecast-error score",
        "eligibility": "Both declared windows fit and companion spectral radius is strictly below one",
        "parameter_count_scope": "Regression coefficients in all equations; covariance count is constant across orders",
        "future_information": "No level, label, validation or test outcome after the selection cutoff is read",
        "sample_policy": "Common equation-target dates for every candidate; lag warmup uses the maximum candidate lag",
    }
    rows = []
    windows = [(f"Rolling{rolling}", positions[-rolling:]), ("Expanding", positions)]
    k = 2
    for p in orders:
        for label, selected in windows:
            n = int(len(selected))
            row = dict(p=p, window=label, n_train=n,
                       train_start=str(levels.index[selected[0]].date()),
                       train_end=str(levels.index[selected[-1]].date()),
                       n_parameters=k*(1+k*p), status="failed", error="",
                       BIC=np.nan, AIC=np.nan, companion_radius=np.nan,
                       stable=False, near_unit_boundary=False)
            try:
                if n < max(minimum, k*p+5):
                    raise ValueError("Insufficient equation targets for this candidate")
                lagged = np.concatenate([delta[selected-j] for j in range(1, p+1)], axis=1)
                response = delta[selected]
                center, scale = lagged.mean(axis=0), lagged.std(axis=0, ddof=1)
                scale = np.where(scale < 1e-10, 1.0, scale)
                design = np.column_stack([np.ones(n), (lagged-center)/scale])
                if not np.isfinite(design).all() or not np.isfinite(response).all():
                    raise ValueError("Nonfinite common target or lag rows")
                beta, _, rank, singular = np.linalg.lstsq(design, response, rcond=1e-10)
                if rank != design.shape[1]:
                    raise ValueError("Rank-deficient VAR design")
                residual = response-design@beta
                covariance = residual.T@residual/n
                sign, logdet = np.linalg.slogdet(covariance)
                if sign <= 0 or not np.isfinite(logdet):
                    raise ValueError("Singular or nonpositive residual covariance")
                raw_beta = beta[1:]/scale[:, None]
                matrices = raw_beta.reshape(p, k, k).transpose(0, 2, 1)
                companion = np.zeros((k*p, k*p))
                companion[:k] = np.concatenate(matrices, axis=1)
                if p > 1:
                    companion[k:, :-k] = np.eye(k*(p-1))
                radius = float(np.max(np.abs(np.linalg.eigvals(companion))))
                if not np.isfinite(radius):
                    raise ValueError("Nonfinite companion eigenvalues")
                parameter_count = int(row["n_parameters"])
                row.update(status="ok", BIC=float(logdet+np.log(n)*parameter_count/n),
                           AIC=float(logdet+2*parameter_count/n), companion_radius=radius,
                           stable=bool(radius < 1.0), near_unit_boundary=bool(abs(radius-1.0) <= 1e-8),
                           design_rank=int(rank), condition_number=float(singular[0]/singular[-1]))
            except (ValueError, TypeError, np.linalg.LinAlgError, FloatingPointError) as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    candidates = pd.DataFrame(rows)
    summary_rows = []
    for p in orders:
        part = candidates.loc[candidates.p.eq(p)]
        success = bool(len(part) == 2 and part.status.eq("ok").all())
        stable = bool(success and part.stable.all())
        summary_rows.append(dict(p=p, eligible=stable,
                                 mean_window_BIC=float(part.BIC.mean()) if success else np.nan,
                                 mean_window_AIC=float(part.AIC.mean()) if success else np.nan,
                                 maximum_companion_radius=float(part.companion_radius.max()) if success else np.nan,
                                 successful_windows=int(part.status.eq("ok").sum()),
                                 reason="eligible" if stable else "unstable dynamics" if success else "fit failed"))
    summary = pd.DataFrame(summary_rows).sort_values(
        ["eligible", "mean_window_BIC", "p"], ascending=[False, True, True], na_position="last").reset_index(drop=True)
    summary["rank"] = pd.Series(pd.NA, index=summary.index, dtype="Int64")
    eligible = summary.index[summary.eligible]
    summary.loc[eligible, "rank"] = np.arange(1, len(eligible)+1)
    report = {"candidates": candidates, "summary": summary, "chosen_p": None, "metadata": metadata}
    if not len(eligible):
        raise VarOrderSelectionError("No VAR order fitted stably in both declared windows", report)
    report["chosen_p"] = int(summary.loc[eligible[0], "p"])
    metadata["chosen_p"] = report["chosen_p"]
    return report
