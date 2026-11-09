"""Experiment code shared by notebooks 03 (synthetic), 04 (Parkinson's benchmark) and 05 (other data).

FittedMethods fits every method once on a training set; evaluate() then gives, for any test set and
density-ratio function, the coverage indicator and region size of every method. run() is the loop
over random splits and shift intensities used by notebooks 04 and 05.
"""
import time

import numpy as np
import pandas as pd

import cpshift as cp
import otgeo


METHODS = ['SCP-L2-LR', 'WCP-L2-LR', 'EU', 'EUc-LR', 'OTCP', 'GRCP',
           'Deep-LocalCov', 'Deep-LocalCov + WCP (ours)', 'Deep-LocalCov + EUc (ours)']

# [my choice] shift-corrected versions of the two exchangeable baselines: the same OTCP / GRCP
# scores, calibrated with the weighted quantile of [TBCR] instead of the plain split quantile.
# The OT map and the rank estimator are still fitted on unweighted hold-out data.
WEIGHTED_BASELINES = ['OTCP + WCP', 'GRCP + WCP']
METHODS_ALL = METHODS + WEIGHTED_BASELINES


class FittedMethods:
    def __init__(self, Xtr, Ytr, cfg, rng, deep_module, seed, groups=None):
        """groups: optional patient id per training record. If given, every split (SCP fit/cal,
        EU jackknife, EUc-LR resampling, network train/early-stop/cal, OTCP/GRCP hold-out/cal) keeps
        each patient on one side only (notebook 07). None = record-level splits (notebooks 01-06)."""
        a = self.alpha = cfg['ALPHA']
        self.cfg, self.Xtr, self.Ytr, self.groups = cfg, Xtr, Ytr, groups
        if groups is None:
            self.scp = cp.SplitL2LR(Xtr, Ytr, rng, cfg['SCP_FIT_FRAC'])
            self.eu = cp.EUJackknife(Xtr, Ytr, a)
            self.net = deep_module.DeepLocalCov(Xtr, Ytr, cfg, rng, seed)
        else:
            self.scp = cp.SplitL2LR(Xtr, Ytr, rng, cfg['SCP_FIT_FRAC'], groups=groups)
            self.eu = cp.EUJackknife(Xtr, Ytr, a, groups=groups)
            self.net = deep_module.DeepLocalCov(Xtr, Ytr, cfg, rng, seed, groups=groups)
        self.Xc = Xtr[self.net.cal_idx]
        self.cal = self.net.outputs(self.Xc, Ytr[self.net.cal_idx])

        # OTCP / GRCP: [my choice] the deep calibration half is split again into hold-out (map /
        # rank estimation) and calibration -- the three-way split of Klein et al. Sec. 3.1.
        Zc = self.cal['Y'] - self.cal['mu']
        if groups is None:
            nh = len(self.net.cal_idx) // 2
            Xh, Zh, Xc2, Zc2 = self.Xc[:nh], Zc[:nh], self.Xc[nh:], Zc[nh:]
        else:
            h, c2 = cp.split_indices_grouped(groups[self.net.cal_idx], 0.5, rng)
            Xh, Zh, Xc2, Zc2 = self.Xc[h], Zc[h], self.Xc[c2], Zc[c2]
        self.ot = otgeo.OTCP(Zh, rng, m=cfg['OT_M'], eps_rel=cfg['OT_EPS'])
        self.Xc2 = Xc2
        self.s_ot = self.ot.score(Zc2)
        self.q_ot = cp.split_quantile(self.s_ot, a)
        self.gr = otgeo.GRCP(Xh, Zh)
        self.s_gr = self.gr.score(Xc2, Zc2)
        self.q_gr = cp.split_quantile(self.s_gr, a)
        self.G, self.cell, self.edge = otgeo.residual_grid(Zh, cfg['GRID_SIDE'])
        self.ot_G = self.ot.score(self.G)                     # OT scores of the grid (do not depend on x)
        self.ot_G_sorted = np.sort(self.ot_G)
        self.ot_G_edge_min = self.ot_G[self.edge].min()
        self.ot_area, self.ot_edge = otgeo.area_from_mask(self.ot_G <= self.q_ot, self.cell, self.edge)
        self.yscale = np.prod(self.net.ys)

    def evaluate(self, Xte, Yte, r_fn, rng, vol_idx, oracle_r=None):
        """vol_idx: indices of test inputs whose GRCP region area is computed on the grid."""
        a, cfg, net = self.alpha, self.cfg, self.net
        test = net.outputs(Xte, Yte)
        Zt = test['Y'] - test['mu']
        gr_grid = [self.gr.score_grid(Xte[i], self.G) for i in vol_idx]
        gr = [otgeo.area_from_mask(g <= self.q_gr, self.cell, self.edge) for g in gr_grid]

        s_ot_te = self.ot.score(Zt)
        s_gr_te = self.gr.score(Xte, Zt)

        # weighted (shift-corrected) OTCP / GRCP thresholds, one per test point
        r_te = r_fn(Xte)
        r_c2 = r_fn(self.Xc2)
        q_wot = cp.weighted_quantile_inf(self.s_ot, r_c2, r_te, a)
        q_wgr = cp.weighted_quantile_inf(self.s_gr, r_c2, r_te, a)
        fin = np.isfinite(q_wot)
        wot_area = np.full(len(Zt), np.inf)
        wot_area[fin] = np.searchsorted(self.ot_G_sorted, q_wot[fin], side='right') * self.cell * self.yscale
        wot_edge = float(np.mean(self.ot_G_edge_min <= q_wot))
        wgr = [(np.inf, True) if not np.isfinite(q_wgr[i]) else
               otgeo.area_from_mask(g <= q_wgr[i], self.cell, self.edge) for i, g in zip(vol_idx, gr_grid)]
        res = {
            'SCP-L2-LR': self.scp.unweighted(Xte, Yte, a),
            'WCP-L2-LR': self.scp.weighted(Xte, Yte, a, r_fn),
            'EU': self.eu.predict(Xte, Yte),
            'EUc-LR': (cp.EUcLR(self.Xtr, self.Ytr, r_fn, a, cfg['B'], rng, 'text').predict(Xte, Yte)
                       if self.groups is None else
                       cp.EUcLR(self.Xtr, self.Ytr, r_fn, a, cfg['B'], rng, 'text', groups=self.groups).predict(Xte, Yte)),
            'OTCP': dict(covered=s_ot_te <= self.q_ot,
                         vol=np.full(len(Zt), self.ot_area * self.yscale), edge=float(self.ot_edge)),
            'GRCP': dict(covered=s_gr_te <= self.q_gr,
                         vol=np.array([g[0] for g in gr]) * self.yscale,
                         edge=float(np.mean([g[1] for g in gr]))),
            'Deep-LocalCov': cp.deep_split(self.cal, test, a, net.ys),
            'Deep-LocalCov + WCP (ours)': cp.deep_weighted(self.cal, test, a, net.ys, r_fn(self.Xc), r_fn(Xte)),
            'Deep-LocalCov + EUc (ours)': cp.deep_euc(self.cal, test, a, net.ys, r_fn(self.Xc), cfg['B'], rng, 'text'),
            'OTCP + WCP': dict(covered=s_ot_te <= q_wot, vol=wot_area, edge=wot_edge),
            'GRCP + WCP': dict(covered=s_gr_te <= q_wgr,
                               vol=np.array([g[0] for g in wgr]) * self.yscale,
                               edge=float(np.mean([g[1] for g in wgr]))),
        }
        if oracle_r is not None:
            res['Deep-LocalCov + WCP, oracle r'] = cp.deep_weighted(self.cal, test, a, net.ys,
                                                                    oracle_r(self.Xc), oracle_r(Xte))
            res['Deep-LocalCov + EUc, oracle r'] = cp.deep_euc(self.cal, test, a, net.ys, oracle_r(self.Xc),
                                                               cfg['B'], rng, 'text')
        return res


def run(X_all, Y_all, sampler, cfg, cc, deep_module, log=print, methods=METHODS):
    rows = []
    a, N = cfg['ALPHA'], len(Y_all)
    t0 = time.time()
    for rep in range(cfg['N_REPS']):
        rng = np.random.default_rng(cfg['SEED'] + rep)
        perm = rng.permutation(N)
        tr, pool = perm[:cfg['N_TRAIN']], perm[cfg['N_TRAIN']:]
        Xtr, Ytr = X_all[tr], Y_all[tr]
        fm = FittedMethods(Xtr, Ytr, cfg, rng, deep_module, cfg['SEED'] + rep)
        centers = cp.kmeanspp(cp.standardize_from(Xtr, Xtr), cc['CEC_J'], rng)

        for k in cfg['K_VALUES']:
            te = sampler.sample(X_all, pool, k, min(cfg['N_TEST'], len(pool)), rng, replace=False)
            Xte, Yte = X_all[te], Y_all[te]
            r_hat = cp.fit_density_ratio(Xtr, Xte)          # from the unlabeled test inputs
            if cfg.get('RATIO_CLIP'):                        # weight clipping (notebook 08), all weighted methods
                r_hat = cp.clip_ratio(r_hat, Xtr, cfg['RATIO_CLIP'])
            vol_idx = rng.choice(len(Xte), size=min(cfg['GRCP_VOL_POINTS'], len(Xte)), replace=False)
            res = fm.evaluate(Xte, Yte, r_hat, rng, vol_idx)

            Xs = cp.standardize_from(Xtr, Xte)
            w = cp.wsc(Xs, [res[m]['covered'].astype(float) for m in methods],
                       cp.random_directions(cc['WSC_N_DIR'], Xte.shape[1], rng), cc['WSC_DELTA'], cc['WSC_GRID'])
            lab = cp.assign(Xs, centers)
            for m, wv in zip(methods, w):
                v = res[m]['vol']
                fin = np.isfinite(v)
                rows.append(dict(rep=rep, k=k, method=m, n_test=len(Yte),
                                 coverage=100 * res[m]['covered'].mean(),
                                 volume_median=np.median(v),
                                 volume_mean_finite=v[fin].mean() if fin.any() else np.nan,
                                 pct_infinite=100 * (1 - fin.mean()),
                                 WSC=100 * wv,
                                 CEC_X=1e4 * cp.cec_x(lab, res[m]['covered'].astype(float), cc['CEC_J'], a),
                                 grid_edge_hit=res[m].get('edge', 0.0)))
        log(f"rep {rep + 1}/{cfg['N_REPS']} done | deep epochs={len(fm.net.hist)} | elapsed {time.time() - t0:.0f}s")
    return pd.DataFrame(rows)


def summarize(df, order=METHODS, keys=('method', 'k')):
    order = [m for m in order if m in set(df.method)] + [m for m in METHODS_ALL if m in set(df.method) and m not in order]
    """Volume: median over splits of the per-split median area (robust); the mean is kept too."""
    by_k = (df.groupby(list(keys))
            .agg(coverage=('coverage', 'mean'), coverage_sd=('coverage', 'std'),
                 volume=('volume_median', 'median'), volume_mean=('volume_median', 'mean'),
                 pct_infinite=('pct_infinite', 'mean'), WSC=('WSC', 'mean'), WSC_sd=('WSC', 'std'),
                 CEC_X=('CEC_X', 'mean'), grid_edge_hit=('grid_edge_hit', 'mean'))
            .round(2).reset_index())
    tables = {col: by_k.pivot(index='method', columns='k', values=col).reindex(order)
              for col in ['coverage', 'volume', 'WSC', 'CEC_X']}
    return by_k, tables


# ----------------------------------------------------------------------------------------------
# Notebook 07: real (patient-level) shifts. Training and test records come from DIFFERENT patients.
# ----------------------------------------------------------------------------------------------
def patient_groups(df, design, rng):
    """Return (source_patients, target_patients) for one split.

    design = 'random' : patients split at random into two halves (no systematic shift; measures
                        the effect of testing on new patients only)
             'age'    : source = patients younger than the median patient age, target = the others
             'sex'    : source = sex 0, target = sex 1 (UCI coding: 0 = male, 1 = female [my reading])
    """
    pat = df.groupby('subject#')[['age', 'sex']].first()
    ids = pat.index.to_numpy()
    if design == 'random':
        perm = rng.permutation(ids)
        return perm[:len(ids) // 2], perm[len(ids) // 2:]
    if design == 'age':
        med = np.median(pat['age'])
        return ids[pat['age'].to_numpy() < med], ids[pat['age'].to_numpy() >= med]
    if design == 'sex':
        return ids[pat['sex'].to_numpy() == 0], ids[pat['sex'].to_numpy() == 1]
    raise ValueError(design)


def run_patient_shift(df, X_all, Y_all, designs, cfg, cc, deep_module, log=print, methods=METHODS_ALL,
                      grouped_cal=False):
    """grouped_cal=True: all internal splits keep each patient on one side (FittedMethods(groups=...))."""
    """Like run(), but training records come from source patients and test records from target patients."""
    for c in ['subject#', 'age', 'sex']:
        if c not in df.columns:
            raise KeyError(f"column '{c}' not found -- use the original UCI file parkinsons_updrs.data")
    rows, t0 = [], time.time()
    a = cfg['ALPHA']
    subj = df['subject#'].to_numpy()
    for design in designs:
        for rep in range(cfg['N_REPS']):
            rng = np.random.default_rng(cfg['SEED'] + rep)
            src, tgt = patient_groups(df, design, rng)
            src_idx = np.flatnonzero(np.isin(subj, src))
            tgt_idx = np.flatnonzero(np.isin(subj, tgt))
            tr = rng.choice(src_idx, size=min(cfg['N_TRAIN'], len(src_idx)), replace=False)
            te = rng.choice(tgt_idx, size=min(cfg['N_TEST'], len(tgt_idx)), replace=False)
            Xtr, Ytr, Xte, Yte = X_all[tr], Y_all[tr], X_all[te], Y_all[te]
            fm = FittedMethods(Xtr, Ytr, cfg, rng, deep_module, cfg['SEED'] + rep,
                               groups=subj[tr] if grouped_cal else None)
            r_hat = cp.fit_density_ratio(Xtr, Xte)
            if cfg.get('RATIO_CLIP'):
                r_hat = cp.clip_ratio(r_hat, Xtr, cfg['RATIO_CLIP'])
            rc = r_hat(fm.Xc)
            ess = rc.sum() ** 2 / (rc ** 2).sum()          # effective sample size of calibration weights
            vol_idx = rng.choice(len(Xte), size=min(cfg['GRCP_VOL_POINTS'], len(Xte)), replace=False)
            res = fm.evaluate(Xte, Yte, r_hat, rng, vol_idx)

            Xs = cp.standardize_from(Xtr, Xte)
            centers = cp.kmeanspp(cp.standardize_from(Xtr, Xtr), cc['CEC_J'], rng)
            w = cp.wsc(Xs, [res[m]['covered'].astype(float) for m in methods],
                       cp.random_directions(cc['WSC_N_DIR'], Xte.shape[1], rng), cc['WSC_DELTA'], cc['WSC_GRID'])
            lab = cp.assign(Xs, centers)
            for m, wv in zip(methods, w):
                v = res[m]['vol']
                fin = np.isfinite(v)
                rows.append(dict(design=design, grouped_cal=grouped_cal, rep=rep, method=m,
                                 n_train=len(tr), n_test=len(te),
                                 n_src_patients=len(src), n_tgt_patients=len(tgt), ess_cal=ess,
                                 n_cal=len(rc), coverage=100 * res[m]['covered'].mean(),
                                 volume_median=np.median(v),
                                 volume_mean_finite=v[fin].mean() if fin.any() else np.nan,
                                 pct_infinite=100 * (1 - fin.mean()), WSC=100 * wv,
                                 CEC_X=1e4 * cp.cec_x(lab, res[m]['covered'].astype(float), cc['CEC_J'], a),
                                 grid_edge_hit=res[m].get('edge', 0.0)))
            log(f"{design} rep {rep + 1}/{cfg['N_REPS']} | patients {len(src)}->{len(tgt)} | "
                f"ESS {ess:.0f}/{len(rc)} | elapsed {time.time() - t0:.0f}s")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Notebook 08: density-ratio ablation (weight clipping, stronger ridge). Each split is fitted once and
# evaluated with every ratio variant, so differences between variants are paired.
# ----------------------------------------------------------------------------------------------
ABLATION_METHODS = ['WCP-L2-LR', 'EUc-LR', 'OTCP + WCP', 'GRCP + WCP', 'Deep-LocalCov',
                    'Deep-LocalCov + WCP (ours)', 'Deep-LocalCov + EUc (ours)']


def run_ratio_ablation(df, X_all, Y_all, settings, variants, cfg, cc, deep_module, log=print,
                       methods=ABLATION_METHODS):
    """settings: list of ('tilt', k) [Gan & Liu shift, record-level splits as notebook 04] or
    ('patient', design) [notebook 07 designs, patient-grouped calibration]."""
    rows, t0 = [], time.time()
    subj = df['subject#'].to_numpy() if 'subject#' in df.columns else None
    sampler = cp.ShiftSampler(X_all)
    for kind, val in settings:
        tag = f"{kind}:{val}"
        for rep in range(cfg['N_REPS']):
            rng = np.random.default_rng(cfg['SEED'] + rep)
            if kind == 'tilt':
                perm = rng.permutation(len(Y_all))
                tr, pool = perm[:cfg['N_TRAIN']], perm[cfg['N_TRAIN']:]
                te = sampler.sample(X_all, pool, val, cfg['N_TEST'], rng, replace=False)
                groups = None
            elif kind == 'patient':
                if subj is None:
                    raise KeyError("column 'subject#' not found -- use the original UCI file")
                src, tgt = patient_groups(df, val, rng)
                src_idx, tgt_idx = np.flatnonzero(np.isin(subj, src)), np.flatnonzero(np.isin(subj, tgt))
                tr = rng.choice(src_idx, size=min(cfg['N_TRAIN'], len(src_idx)), replace=False)
                te = rng.choice(tgt_idx, size=min(cfg['N_TEST'], len(tgt_idx)), replace=False)
                groups = subj[tr]
            else:
                raise ValueError(kind)
            Xtr, Ytr, Xte, Yte = X_all[tr], Y_all[tr], X_all[te], Y_all[te]
            fm = FittedMethods(Xtr, Ytr, cfg, rng, deep_module, cfg['SEED'] + rep, groups=groups)
            vol_idx = rng.choice(len(Xte), size=min(cfg['GRCP_VOL_POINTS'], len(Xte)), replace=False)
            dirs = cp.random_directions(cc['WSC_N_DIR'], Xte.shape[1], rng)
            Xs = cp.standardize_from(Xtr, Xte)
            for vname, r_fn in cp.ratio_variants(Xtr, Xte, variants).items():
                rc = r_fn(fm.Xc)
                ess = rc.sum() ** 2 / (rc ** 2).sum()
                res = fm.evaluate(Xte, Yte, r_fn, rng, vol_idx)
                w = cp.wsc(Xs, [res[m]['covered'].astype(float) for m in methods], dirs,
                           cc['WSC_DELTA'], cc['WSC_GRID'])
                for m, wv in zip(methods, w):
                    v = res[m]['vol']
                    rows.append(dict(setting=tag, variant=vname, rep=rep, method=m, ess_cal=ess,
                                     n_cal=len(rc), coverage=100 * res[m]['covered'].mean(),
                                     volume_median=np.median(v), pct_infinite=100 * np.mean(~np.isfinite(v)),
                                     WSC=100 * wv))
            log(f"{tag} rep {rep + 1}/{cfg['N_REPS']} | elapsed {time.time() - t0:.0f}s")
    return pd.DataFrame(rows)
