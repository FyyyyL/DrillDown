"""Append-only stationarity and fitted-dynamics diagnostics.

Old Notebook, new cell: %run -i "WTI_Stationarity_Addon.py"
Reuse fit_log/panel/factor_store or saved CSVs; no forecasting model is refit.
RESULTS_DIR may point to the existing results folder after a kernel restart.
Optional STATIONARITY_ORIGIN="2026-09-30" restricts the diagnostic origin.

VAR companion eigenvalues are checked INSIDE the unit circle. AR lag-
polynomial roots are checked OUTSIDE; these conventions are reciprocal.
MA roots assess invertibility separately. ADF/KPSS give sample evidence,
not proof of full oil-level/exogenous/GARCH-process stationarity.

Official references:
https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html
https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html
https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_process.ArmaProcess.html
"""
"""Post-fit conditional-dynamics diagnostics from already saved parameters.

These root checks assess the conditional VAR dynamics or ARMA error filter.
They do not prove marginal stationarity of oil levels, generated regressors,
the external processes or the full mean-plus-GARCH system. MA invertibility
is reported separately: it is not a condition for finite-order MA stationarity.
"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd


def _sm_polynomial_check(coefficients, tolerance):
    """Coefficients are increasing powers of z; roots must lie OUTSIDE one."""
    coefficients = np.asarray(coefficients, dtype=float)
    if not np.isfinite(coefficients).all() or coefficients.ndim != 1 or coefficients[0] != 1:
        raise ValueError('Invalid lag-polynomial coefficients')
    coefficients = np.trim_zeros(coefficients, trim='b')
    roots = np.polynomial.polynomial.polyroots(coefficients) if len(coefficients) > 1 else np.array([], dtype=complex)
    moduli = np.abs(roots)
    outside = bool((moduli > 1.0).all())
    near = bool((np.abs(moduli - 1.0) <= tolerance).any())
    return {
        'condition_met': outside,
        'near_unit_boundary': near,
        'minimum_root_modulus': float(moduli.min()) if len(moduli) else np.nan,
        'roots': json.dumps([{'real':float(z.real), 'imaginary':float(z.imag), 'modulus':float(abs(z))} for z in roots]),
        'nonzero_polynomial_degree': len(coefficients) - 1,
    }


def _sm_details(value):
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Missing saved model details')
    result = json.loads(value)
    if not isinstance(result, dict):
        raise ValueError('Saved model details are not an object')
    return result


def _sm_family(name, details):
    family = str(details.get('family', str(name).split('|')[0].strip()))
    return 'Hybrid' if family == 'ARIMAX' else family


def _sm_one_model(row, tolerance):
    name = str(row.get('model', ''))
    result = {
        'model':name, 'family':name.split('|')[0].strip(),
        'origin_date':pd.to_datetime(row.get('origin_date'), errors='coerce'),
        'fit_status':str(row.get('status', 'unknown')), 'check_status':'unavailable',
        'train_start':pd.NaT, 'train_end':pd.NaT, 'n_train':row.get('n_train', np.nan),
        'conditional_dynamics_pass':np.nan, 'conditional_dynamics_status':'unavailable',
        'AR_condition_met':np.nan, 'MA_invertibility_met':np.nan,
        'integration_d':np.nan, 'companion_radius':np.nan,
        'minimum_AR_root_modulus':np.nan, 'minimum_MA_root_modulus':np.nan,
        'near_unit_boundary':np.nan, 'check_source':'', 'notes':'',
    }
    if result['fit_status'] != 'ok':
        result.update(check_status='fit_failed', conditional_dynamics_status='fit_failed',
                      notes=str(row.get('error', 'Original fit did not succeed')))
        return result
    try:
        info = _sm_details(row.get('details'))
        family = _sm_family(name, info)
        result.update(family=family,
                      train_start=pd.to_datetime(info.get('train_start'), errors='coerce'),
                      train_end=pd.to_datetime(info.get('train_end'), errors='coerce'),
                      optimizer_converged=info.get('arimax_converged', np.nan))
        parameters = info.get('mean_parameters')
        if family in {'VAR', 'VARX'}:
            saved_radius = info.get('companion_radius')
            if isinstance(parameters, dict) and parameters and all(isinstance(value, dict) for value in parameters.values()):
                names = list(parameters)
                lag = int(info.get('lag', 0))
                if lag < 1:
                    raise ValueError('Saved VAR lag order is unavailable')
                k = len(names)
                matrices = np.empty((lag, k, k), dtype=float)
                for step in range(lag):
                    for i, target in enumerate(names):
                        for j, predictor in enumerate(names):
                            matrices[step, i, j] = float(parameters[target][f'{predictor}_L{step+1}'])
                if not np.isfinite(matrices).all():
                    raise ValueError('Nonfinite saved VAR autoregressive parameters')
                companion = np.zeros((lag*k, lag*k))
                companion[:k] = np.concatenate(matrices, axis=1)
                if lag > 1:
                    companion[k:, :-k] = np.eye((lag-1)*k)
                eigenvalues = np.linalg.eigvals(companion)
                radius = float(np.abs(eigenvalues).max())
                result.update(check_source='recomputed from saved autoregressive coefficients',
                              VAR_lag=lag, endogenous_dimension=k,
                              companion_eigenvalues=json.dumps([{'real':float(z.real), 'imaginary':float(z.imag)} for z in eigenvalues]),
                              saved_radius_mismatch=(bool(abs(radius-float(saved_radius)) > max(tolerance, 1e-10))
                                                     if saved_radius is not None and np.isfinite(float(saved_radius)) else np.nan))
                result['check_status'] = 'checked'
            elif saved_radius is not None and np.isfinite(float(saved_radius)):
                radius = float(saved_radius)
                result.update(check_status='saved_radius_only', check_source='saved companion radius; coefficients unavailable for independent recomputation',
                              saved_radius_mismatch=np.nan)
            else:
                raise ValueError('Saved VAR coefficients and companion radius are unavailable')
            if not np.isfinite(radius) or radius < 0:
                raise ValueError('Invalid companion spectral radius')
            stable = radius < 1.0
            boundary = abs(radius-1.0) <= tolerance
            result.update(companion_radius=radius, conditional_dynamics_pass=stable,
                          near_unit_boundary=boundary,
                          conditional_dynamics_status=('near_unit_boundary' if boundary else 'stable' if stable else 'unstable'),
                          stationarity_margin=1.0-radius,
                          notes='Companion eigenvalues must lie inside the unit circle; this is conditional mean dynamics only. Exogenous/GARCH stationarity is not certified.')
        elif family in {'Hybrid', 'ARIMA'}:
            if not isinstance(parameters, dict):
                raise ValueError('Saved ARMA parameter dictionary is unavailable')
            order = info.get('arimax_order')
            if not isinstance(order, (list, tuple)) or len(order) != 3:
                raise ValueError('Saved nonseasonal ARIMA order is unavailable')
            p, d, q = [int(value) for value in order]
            if min(p, d, q) < 0 or any(float(value) != int(value) for value in order):
                raise ValueError('Invalid saved ARIMA order')
            if any(re.match(r'^(ar|ma)\.S\.', str(key)) for key in parameters):
                raise ValueError('Seasonal polynomial parameters are not supported by this nonseasonal checker')
            ar = [float(parameters[f'ar.L{j}']) for j in range(1, p+1)]
            ma = [float(parameters[f'ma.L{j}']) for j in range(1, q+1)]
            ar_check = _sm_polynomial_check([1.0] + [-value for value in ar], tolerance)
            ma_check = _sm_polynomial_check([1.0] + ma, tolerance)
            conditional = bool(ar_check['condition_met'] and d == 0)
            status = ('integrated_d_positive' if d > 0 else
                      'near_unit_boundary' if ar_check['near_unit_boundary'] else
                      'stable' if ar_check['condition_met'] else 'unstable')
            result.update(check_status='checked', check_source='recomputed from saved nonseasonal AR/MA coefficients',
                          AR_order=p, MA_order=q, integration_d=d,
                          AR_condition_met=ar_check['condition_met'],
                          MA_invertibility_met=ma_check['condition_met'],
                          minimum_AR_root_modulus=ar_check['minimum_root_modulus'],
                          minimum_MA_root_modulus=ma_check['minimum_root_modulus'],
                          AR_nonzero_polynomial_degree=ar_check['nonzero_polynomial_degree'],
                          MA_nonzero_polynomial_degree=ma_check['nonzero_polynomial_degree'],
                          AR_roots=ar_check['roots'], MA_roots=ma_check['roots'],
                          near_unit_boundary=ar_check['near_unit_boundary'],
                          MA_near_unit_boundary=ma_check['near_unit_boundary'],
                          conditional_dynamics_pass=conditional, conditional_dynamics_status=status,
                          notes='AR roots outside the unit circle and d=0 assess the ARMA error process. MA roots assess invertibility separately; oil-level/generated-X/full-GARCH stationarity is not certified.')
        else:
            raise ValueError('Unsupported model family')
    except (ValueError, TypeError, KeyError, np.linalg.LinAlgError, OverflowError) as exc:
        result.update(check_status='unavailable', conditional_dynamics_status='unavailable',
                      conditional_dynamics_pass=np.nan, notes=f'{type(exc).__name__}: {exc}')
    return result


def check_saved_model_dynamics(ns, outdir):
    """Diagnose every existing fit, without fitting or forecasting anything.

    ``latest`` contains each model's latest successful saved fit with available
    training-date metadata, for use by the separate series-testing module.
    Optional STATIONARITY_ORIGIN excludes later saved fits from all diagnostics.
    """
    config = ns.get('cfg', {})
    default = 'WTI_Independent_ML_Smoke_Results' if config.get('RUN_MODE') == 'smoke' else 'WTI_Independent_ML_Results'
    source = Path(ns.get('RESULTS_DIR') or ns.get('OUTPUT_DIR') or Path.cwd()/default)
    log = ns.get('fit_log')
    if not isinstance(log, pd.DataFrame):
        path = source/'01_fit'/'fit_log.csv'
        if not path.exists():
            raise FileNotFoundError('Keep fit_log in the kernel or set RESULTS_DIR to the saved results folder containing 01_fit/fit_log.csv. No model is refitted.')
        log = pd.read_csv(path)
        input_mode = 'saved fit_log.csv'
    else:
        input_mode = 'kernel fit_log'
    required = {'model', 'origin_date', 'status', 'details'}
    if not required.issubset(log.columns):
        raise ValueError('Fit log lacks columns: ' + repr(sorted(required-set(log.columns))))
    log = log.copy()
    log['origin_date'] = pd.to_datetime(log.origin_date, errors='raise')
    allowed = {'VAR', 'VARX', 'Hybrid', 'ARIMAX'}
    if bool(ns.get('STATIONARITY_INCLUDE_ARIMA', True)):
        allowed.add('ARIMA')
    log = log.loc[log.model.astype(str).str.split('|', regex=False).str[0].str.strip().isin(allowed)]
    selected_origin = ns.get('STATIONARITY_ORIGIN')
    if selected_origin is not None:
        selected_origin = pd.Timestamp(selected_origin).normalize()
        log = log.loc[log.origin_date.le(selected_origin)]
    if log.empty:
        raise ValueError('No saved VAR/VARX/ARIMAX fits are available for the selected diagnostic scope')
    tolerance = float(ns.get('STATIONARITY_ROOT_TOL', 1e-8))
    if not np.isfinite(tolerance) or not 0 <= tolerance < 0.1:
        raise ValueError('STATIONARITY_ROOT_TOL must be a finite number in [0,0.1)')
    table = pd.DataFrame([_sm_one_model(row, tolerance) for row in log.to_dict('records')])
    table = table.sort_values(['model', 'origin_date']).reset_index(drop=True)
    summaries = []
    for model, part in table.groupby('model', sort=True):
        statuses = part.conditional_dynamics_status
        summaries.append({'model':model, 'family':part.family.iloc[0], 'N_saved_rows':len(part),
                          'N_successful_fits':int(part.fit_status.eq('ok').sum()),
                          'N_recomputed':int(part.check_status.eq('checked').sum()),
                          'N_saved_radius_only':int(part.check_status.eq('saved_radius_only').sum()),
                          'N_stable':int(statuses.eq('stable').sum()), 'N_unstable':int(statuses.eq('unstable').sum()),
                          'N_near_unit_boundary':int(statuses.eq('near_unit_boundary').sum()),
                          'N_integrated_d_positive':int(statuses.eq('integrated_d_positive').sum()),
                          'N_failed_fits':int(part.check_status.eq('fit_failed').sum()),
                          'N_unavailable':int(part.check_status.eq('unavailable').sum()),
                          'N_MA_noninvertible':int(part.MA_invertibility_met.eq(False).sum()),
                          'max_companion_radius':pd.to_numeric(part.companion_radius, errors='coerce').max(),
                          'minimum_AR_root_modulus':pd.to_numeric(part.minimum_AR_root_modulus, errors='coerce').min(),
                          'minimum_MA_root_modulus':pd.to_numeric(part.minimum_MA_root_modulus, errors='coerce').min()})
    summary = pd.DataFrame(summaries)
    usable = table.loc[table.fit_status.eq('ok') & table.train_start.notna() & table.train_end.notna()]
    latest = usable.sort_values('origin_date').groupby('model', sort=False).tail(1).sort_values('model').reset_index(drop=True)
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, frame in [('stationarity_model_dynamics_all_origins.csv', table),
                        ('stationarity_model_dynamics_summary.csv', summary),
                        ('stationarity_model_dynamics_latest.csv', latest)]:
        path = destination/name
        frame.to_csv(path, index=False, encoding='utf-8-sig')
        files[name] = str(path.resolve())
    metadata = {
        'input_mode':input_mode, 'source_directory':str(source.resolve()),
        'selected_max_origin':str(selected_origin.date()) if selected_origin is not None else None,
        'root_boundary_tolerance':tolerance,
        'VAR_condition':'All companion eigenvalue moduli <1. This differs from AR polynomial roots, whose moduli must be >1.',
        'AR_condition':'Roots of 1-phi1*z-...-phip*z**p outside the unit circle; d=0 for a stationary error process.',
        'MA_condition':'Roots of 1+theta1*z+...+thetaq*z**q outside the unit circle indicate invertibility, not MA stationarity.',
        'scope':'Saved conditional mean/error dynamics only. No proof of marginal stationarity of oil price levels, generated external signals, or a full mean-plus-GARCH process.',
        'models_refitted':False,
    }
    metadata_path = destination/'stationarity_model_dynamics_metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    files[metadata_path.name] = str(metadata_path.resolve())
    print('Saved-model root checks completed; no model was refitted.')
    return {'full':table, 'summary':summary, 'latest':latest, 'files':files, 'metadata':metadata}


"""ADF/KPSS evidence on the actual latest fitted training calendars only."""
from pathlib import Path
import json
import warnings
import zipfile
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def _st_frame_dates(frame, date_column=None):
    frame = frame.copy()
    if date_column is not None:
        frame = frame.set_index(date_column)
    frame.index = pd.to_datetime(frame.index, errors="raise").normalize()
    if frame.index.duplicated().any() or not frame.index.is_monotonic_increasing:
        raise ValueError("Series data need unique increasing quote dates")
    return frame


def _st_raw_source(source):
    """Read only supplied WTI and Brent columns, never execute model code."""
    source = Path(source).expanduser()
    if not source.exists():
        return None
    names = ["model_inputs/X_all65_values.csv", "target/WTI_target_daily.csv"]
    if source.is_file():
        if not zipfile.is_zipfile(source):
            return None
        with zipfile.ZipFile(source) as archive:
            members = set(archive.namelist())
            roots = [x[:-len(names[0])] for x in members if x.endswith(names[0])]
            roots = [x for x in roots if all(x + name in members for name in names)]
            if len(roots) != 1:
                raise ValueError("Expected one complete WTI/Brent source root in ZIP")
            with archive.open(roots[0] + names[0]) as stream:
                brent = pd.read_csv(stream, usecols=["date", "F001"])
            with archive.open(roots[0] + names[1]) as stream:
                wti = pd.read_csv(stream, usecols=["date", "wti_usd_per_bbl"])
    else:
        roots = [source, source / "Oil_Daily_65"]
        roots = [x for x in roots if all((x / name).exists() for name in names)]
        if len(roots) != 1:
            return None
        brent = pd.read_csv(roots[0] / names[0], usecols=["date", "F001"])
        wti = pd.read_csv(roots[0] / names[1], usecols=["date", "wti_usd_per_bbl"])
    brent = _st_frame_dates(brent, "date")
    wti = _st_frame_dates(wti, "date").rename(columns={"wti_usd_per_bbl":"price"})
    if not brent.index.equals(wti.index):
        raise ValueError("Source WTI and Brent calendars differ")
    return wti.join(brent)


def _st_load_panel(ns, base):
    panel = ns.get("panel")
    sources = {}
    if isinstance(panel, pd.DataFrame) and "price" in panel:
        panel = _st_frame_dates(panel[[x for x in ["price", "F001"] if x in panel]])
        sources.update({x:"kernel panel" for x in panel})
    elif (base / "00_data" / "regimes.csv").exists():
        panel = _st_frame_dates(pd.read_csv(base / "00_data" / "regimes.csv", usecols=["date", "price"]), "date")
        sources["price"] = str(base / "00_data" / "regimes.csv")
    else:
        panel = None
    if panel is None or "F001" not in panel:
        source = ns.get("STATIONARITY_DATA_SOURCE") or ns.get("DATA_SOURCE")
        if source is None:
            choices = [Path.cwd() / "Oil_Daily_65_20261023.zip", Path.cwd() / "Oil_Daily_65"]
            source = next((x for x in choices if x.exists()), None)
        raw = _st_raw_source(source) if source is not None else None
        if raw is not None:
            if panel is None:
                panel = raw
                sources = {x:str(source) for x in panel}
            else:
                common = panel.index.intersection(raw.index)
                if not np.allclose(pd.to_numeric(panel.loc[common, "price"]), pd.to_numeric(raw.loc[common, "price"]), equal_nan=True):
                    raise ValueError("Source ZIP and saved/model WTI prices do not match")
                panel["F001"] = raw.F001.reindex(panel.index)
                sources["F001"] = str(source)
    if panel is not None:
        panel = panel.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return panel, sources


def _st_one_test(values, expected_n, cfg):
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    row = {"N_expected":int(expected_n), "N_calendar":len(values), "N_finite":int(values.notna().sum()),
           "ADF_statistic":np.nan, "ADF_pvalue":np.nan, "ADF_used_lag":np.nan,
           "KPSS_statistic":np.nan, "KPSS_pvalue":np.nan, "KPSS_used_lag":np.nan,
           "ADF_reject_unit_root":None, "KPSS_reject_level_stationarity":None,
           "status":"not_tested", "stationarity_evidence":"unavailable", "warnings":""}
    if len(values) != int(expected_n):
        row.update(status="training_calendar_length_mismatch", stationarity_evidence="unavailable")
        return row
    if values.isna().any():
        row.update(status="missing_or_nonfinite_on_training_calendar", stationarity_evidence="unavailable")
        return row
    minimum = int(cfg.get("STATIONARITY_MIN_OBSERVATIONS", 50))
    if len(values) < minimum:
        row.update(status="insufficient_observations", stationarity_evidence="inconclusive")
        return row
    data = values.to_numpy(dtype=float)
    if np.ptp(data) <= 1e-12 * max(1.0, float(np.max(np.abs(data)))):
        row.update(status="constant_series_tests_undefined", stationarity_evidence="degenerate_constant_series")
        return row
    alpha = float(cfg.get("STATIONARITY_ALPHA", .05))
    if not 0 < alpha < 1:
        raise ValueError("STATIONARITY_ALPHA must lie strictly between zero and one")
    requested_lag = int(cfg.get("STATIONARITY_MAX_LAG", 12))
    if requested_lag < 0:
        raise ValueError("STATIONARITY_MAX_LAG must be nonnegative")
    maxlag = min(requested_lag, len(data) // 2 - 2)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            adf = adfuller(data, maxlag=maxlag, regression="c", autolag="AIC")
            kps = kpss(data, regression="c", nlags="auto")
        row.update(ADF_statistic=float(adf[0]), ADF_pvalue=float(adf[1]), ADF_used_lag=int(adf[2]),
                   KPSS_statistic=float(kps[0]), KPSS_pvalue=float(kps[1]), KPSS_used_lag=int(kps[2]),
                   warnings="; ".join(str(w.message) for w in caught))
        reject_adf, reject_kpss = bool(adf[1] < alpha), bool(kps[1] < alpha)
        evidence = ("consistent_with_level_stationarity" if reject_adf and not reject_kpss else
                    "consistent_with_nonstationarity" if not reject_adf and reject_kpss else
                    "mixed_test_evidence" if reject_adf and reject_kpss else "inconclusive")
        row.update(ADF_reject_unit_root=reject_adf, KPSS_reject_level_stationarity=reject_kpss,
                   status="ok", stationarity_evidence=evidence)
    except Exception as exc:
        row.update(status="test_failed", warnings=f"{type(exc).__name__}: {exc}")
    return row


def check_training_series_stationarity(ns, outdir, model_report):
    """Diagnose latest actual mean-training samples; never refit a forecast.

    ADF H0: a unit root. KPSS H0: level stationarity (constant mean).
    These tests provide evidence, not proof of full process stationarity.
    """
    base = Path(ns.get("RESULTS_DIR") or ns.get("OUTPUT_DIR") or Path.cwd() / "WTI_Independent_ML_Results")
    saved = json.loads((base / "run_config.json").read_text(encoding="utf-8-sig")) if (base / "run_config.json").exists() else {}
    cfg = {**saved, **ns.get("cfg", {})}
    for key in ["STATIONARITY_ALPHA", "STATIONARITY_MAX_LAG", "STATIONARITY_MIN_OBSERVATIONS"]:
        if key in ns:
            cfg[key] = ns[key]
    latest = model_report.get("latest", pd.DataFrame())
    panel, sources = _st_load_panel(ns, base)
    store = ns.get("factor_store") or cfg.get("_FACTOR_STORE") or {}
    factor_cache = {}
    requests = {}

    def add(series_name, values, record, window, source):
        start, end = pd.Timestamp(record["train_start"]), pd.Timestamp(record["train_end"])
        origin = pd.Timestamp(record["origin_date"])
        if end > origin:
            raise ValueError("A saved training window extends beyond its prediction origin")
        expected = int(record["n_train"])
        key = (series_name, window, start, end, expected)
        if key in requests:
            requests[key]["source_models"].append(str(record["model"]))
            return
        values = None if values is None else values.loc[start:end]
        requests[key] = {"series":series_name, "window":window, "train_start":start, "train_end":end,
                         "source_models":[str(record["model"])], "source":source,
                         "values":values, "expected":expected}

    for record in latest.to_dict("records"):
        if pd.isna(record.get("train_start")) or pd.isna(record.get("train_end")) or not record.get("n_train"):
            continue
        model = str(record["model"])
        family = str(record.get("family", model.split(" | ")[0]))
        window = str(record.get("window") or ("Rolling500" if "Rolling" in model else "Expanding"))
        price = panel.price if isinstance(panel, pd.DataFrame) and "price" in panel else None
        add("WTI price level (diagnostic, not differenced model input)", price, record, window, sources.get("price", "unavailable"))
        add("WTI one-quote change (mean-model dependent series)", None if price is None else price.diff(), record, window, sources.get("price", "unavailable"))
        if family in {"VAR", "VARX"}:
            brent = panel.F001 if isinstance(panel, pd.DataFrame) and "F001" in panel else None
            add("Brent one-quote change (VAR endogenous input)", None if brent is None else brent.diff(), record, window, sources.get("F001", "unavailable: provide kernel panel or STATIONARITY_DATA_SOURCE"))
        factor_family = "Hybrid" if family in {"Hybrid", "ARIMAX"} else "VARX" if family == "VARX" else None
        if factor_family is not None:
            cache_key = (factor_family, window)
            if cache_key not in factor_cache:
                frame = store.get("oof", {}).get(cache_key)
                factor_source = "kernel historical OOF factor"
                if frame is None:
                    path = base / "02_ml_factors" / f"oof_factor_{factor_family}_{window}.csv"
                    frame = _st_frame_dates(pd.read_csv(path), "target_date") if path.exists() else None
                    factor_source = str(path) if frame is not None else "unavailable: saved historical OOF factor missing"
                elif isinstance(frame, pd.DataFrame):
                    frame = _st_frame_dates(frame)
                factor_cache[cache_key] = (frame["factor"] if isinstance(frame, pd.DataFrame) and "factor" in frame else None, factor_source)
            factor, factor_source = factor_cache[cache_key]
            factor_name = "X_ARIMAX" if factor_family == "Hybrid" else "X_VARX"
            add(factor_name + " (historical OOF signal)", factor, record, window, factor_source)
    rows = []
    for item in requests.values():
        values, expected = item.pop("values"), item.pop("expected")
        item["source_models"] = " | ".join(dict.fromkeys(item["source_models"]))
        if values is None:
            test = _st_one_test(pd.Series(dtype=float), expected, cfg)
            test.update(status="source_unavailable", stationarity_evidence="unavailable")
        else:
            test = _st_one_test(values, expected, cfg)
        rows.append({**item, **test})
    table = pd.DataFrame(rows)
    metadata = {"alpha":float(cfg.get("STATIONARITY_ALPHA", .05)), "ADF_null":"unit root", "KPSS_null":"level stationarity",
                "regression":"c (constant mean; no deterministic trend)", "ADF_autolag":"AIC", "ADF_max_lag":int(cfg.get("STATIONARITY_MAX_LAG", 12)),
                "KPSS_lags":"auto", "minimum_observations":int(cfg.get("STATIONARITY_MIN_OBSERVATIONS", 50)),
                "scope":"Latest successful fitted mean-training calendars only; WTI levels are a diagnostic comparator. No future forecast path is tested; no model is refitted.",
                "interpretation":"Tests provide evidence, not proof. KPSS boundary warnings identify table-limited p-values. Missing observations are not filled or removed to compress the quote clock.",
                "data_sources":sources, "model_window_count":len(latest), "unique_series_tests":len(table)}
    output = Path(outdir); output.mkdir(parents=True, exist_ok=True)
    path = output / "stationarity_training_series_tests.csv"
    table.to_csv(path, index=False, encoding="utf-8-sig")
    meta = output / "stationarity_series_metadata.json"
    meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"tests":table, "metadata":metadata, "files":{"series_tests":str(path), "series_metadata":str(meta)}}


def run_wti_stationarity_checks(ns):
    """Check saved fits and historical inputs; retain all original forecasts."""
    config=ns.get("cfg", {})
    default="WTI_Independent_ML_Smoke_Results" if config.get("RUN_MODE")=="smoke" else "WTI_Independent_ML_Results"
    source=Path(ns.get("RESULTS_DIR") or ns.get("OUTPUT_DIR") or Path.cwd()/default)
    output=Path(ns.get("STATIONARITY_OUTPUT_DIR") or source/"09_stationarity_checks")
    if bool(ns.get("STATIONARITY_FROM_DISK", False)):
        context={"RESULTS_DIR":source}
        for key in ("STATIONARITY_ORIGIN","STATIONARITY_ROOT_TOL","STATIONARITY_INCLUDE_ARIMA",
                    "STATIONARITY_ALPHA","STATIONARITY_MAX_LAG","STATIONARITY_MIN_OBSERVATIONS",
                    "STATIONARITY_DATA_SOURCE","DATA_SOURCE"):
            if key in ns:context[key]=ns[key]
    else:
        context=dict(ns);context["RESULTS_DIR"]=source
    models=check_saved_model_dynamics(context,output)
    series=check_training_series_stationarity(context,output,models)
    from IPython.display import display
    print("Latest successful saved fits: conditional dynamics; MA invertibility is separate.")
    model_columns=["model","origin_date","n_train","companion_radius","minimum_AR_root_modulus",
                   "AR_condition_met","integration_d","minimum_MA_root_modulus","MA_invertibility_met",
                   "MA_near_unit_boundary","conditional_dynamics_status"]
    shown_models=models["latest"].reindex(columns=model_columns).copy()
    for column in ("companion_radius","minimum_AR_root_modulus","minimum_MA_root_modulus"):
        shown_models[column]=shown_models[column].map(lambda value: format(value,".12g") if pd.notna(value) else "N/A")
    display(shown_models)
    print("All-origin summary (failed/unavailable fits are retained):")
    display(models["summary"])
    print("Historical training-series evidence: ADF H0=unit root; KPSS H0=level stationarity.")
    test_columns=["series","window","train_end","N_finite","ADF_pvalue","KPSS_pvalue",
                  "stationarity_evidence","status"]
    shown=series["tests"].reindex(columns=test_columns).copy()
    for column in ("ADF_pvalue","KPSS_pvalue"):
        shown[column]=shown[column].map(lambda value: format(value,".4g") if pd.notna(value) else "N/A")
    display(shown)
    print("Root conditions concern conditional mean/error dynamics. They do not establish stationarity of oil levels, generated X or the full GARCH system.")
    print("KPSS may report table-bound p-values; see warnings in the CSV. No forecasts or model choices have been changed.")
    files={**models["files"],**series["files"]}
    manifest={"models":models["metadata"],"series":series["metadata"],"models_refitted":False,"output_directory":str(output.resolve()),"files":files}
    manifest_path=output/"stationarity_diagnostics_manifest.json"
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    files[manifest_path.name]=str(manifest_path.resolve())
    print("Stationarity reports:",output.resolve())
    return {"model":models,"series":series,"files":files,"output_directory":str(output.resolve())}


if __name__ == "__main__":
    stationarity_checks = run_wti_stationarity_checks(globals())
