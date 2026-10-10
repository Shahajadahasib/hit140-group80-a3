# FIFA WORLD CUP 2026 - OBJECTIVE 2 - LINEAR REGRESSION
#   2.1 : Goal DIFFERENCE (104 rows)   2.2 : Goals SCORED (208 rows)
# Loads the two datasets from csv files (converted in 00_convert_to_csv.py)
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, GroupKFold, train_test_split, GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
SEED = 2026
np.random.seed(SEED)



# paths are relative to this file 
# after cloning the repo datasets are saved as csv
HERE = os.path.dirname(os.path.abspath(__file__))   # notebooks folder
BASE_DIR = os.path.join(HERE, "..", "data")
FILE_21 = os.path.join(BASE_DIR, "FIFA_WC2026_Dataset_2_1_104rows.csv")
FILE_22 = os.path.join(BASE_DIR, "FIFA_WC2026_Dataset_2_2_208rows.csv")
OUTPUT_DIR = os.path.join(HERE, "..", "figures")    # charts and tables are saved here
os.makedirs(OUTPUT_DIR, exist_ok=True)
SHOW_PLOTS = True

#expected columns 
Y21, Y22 = "Goal_Difference", "Goals_Scored"
X21_COLS = ["Rank_Diff", "Host_Diff", "Rest_Days_Diff", "Prior_GF_Diff",
            "Prior_GA_Diff", "Prior_PPG_Diff", "Age_Diff", "Log_Value_Diff"]
X22_COLS = ["Team_Rank", "Opp_Rank", "Is_Host", "Team_Prior_GF",
            "Opp_Prior_GA", "Knockout_Stage", "Team_Prior_Matches",
            "Opp_Prior_Clean_Sheet_Rate"]
NON_PREDICTORS = {"match_id", "team", "team_a", "team_b", "opp", "date", "stage"}


def load_dataset(path, y_col, x_cols, expected_rows):
    """Read the .csv dataset and work out predictors."""
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip().str.replace(" ", "_")
    print(f"\n{os.path.basename(path)} -> {df.shape}")
    print("Columns found:", list(df.columns))

    if y_col not in df.columns:
        raise SystemExit(f"Dependent variable '{y_col}' not found in {path}")

    if all(c in df.columns for c in x_cols):
        xs = x_cols
    else:
        xs = [c for c in df.select_dtypes("number").columns
              if c != y_col and c.lower() not in NON_PREDICTORS]
        print("Expected predictor names not all present; using:", xs)

    if len(xs) != 8:
        raise SystemExit(f"Need exactly 8 predictors, found {len(xs)}: {xs}")
    if len(df) != expected_rows:
        print(f"WARNING: expected {expected_rows} rows, found {len(df)}")
    df[[y_col] + xs] = df[[y_col] + xs].apply(pd.to_numeric, errors="coerce")
    if df[[y_col] + xs].isnull().any().any():
        print("WARNING: missing values found - dropping those rows")
        df = df.dropna(subset=[y_col] + xs)
    return df.reset_index(drop=True), xs


print("=" * 80)
print("FIFA WORLD CUP 2026 - OBJECTIVE 2 (LINEAR REGRESSION)")
print("=" * 80)

dataset_21, X21_COLS = load_dataset(FILE_21, Y21, X21_COLS, 104)
dataset_22, X22_COLS = load_dataset(FILE_22, Y22, X22_COLS, 208)

# max 4 shared rules
CONCEPT = {
    "Rank_Diff": "FIFA rank", "Host_Diff": "Host status",
    "Rest_Days_Diff": "Rest days", "Prior_GF_Diff": "Prior goals scored",
    "Prior_GA_Diff": "Prior goals conceded", "Prior_PPG_Diff": "Prior points/game",
    "Age_Diff": "Squad age", "Log_Value_Diff": "Squad value",
    "Team_Rank": "FIFA rank", "Opp_Rank": "FIFA rank", "Is_Host": "Host status",
    "Team_Prior_GF": "Prior goals scored", "Opp_Prior_GA": "Prior goals conceded",
    "Knockout_Stage": "Knockout stage", "Team_Prior_Matches": "Matches played",
    "Opp_Prior_Clean_Sheet_Rate": "Opp clean-sheet rate",
}
shared = sorted({CONCEPT.get(c, c) for c in X21_COLS} &
                {CONCEPT.get(c, c) for c in X22_COLS})
print("\nShared concepts (max 4):", len(shared), shared)
if len(shared) > 4:
    print("WARNING: more than 4 shared explanatory variables!")

# groups for model 2.2 both teams of one match stay together
groups22 = dataset_22["Match_ID"].values if "Match_ID" in dataset_22.columns else None


# modeling functions
def rmse(y, yhat):
    return float(np.sqrt(mean_squared_error(y, yhat)))


def evaluate_split(df, x_cols, y_col, groups=None):
    X, y = df[x_cols].values, df[y_col].values
    if groups is None:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SEED)
    else:
        tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED)
                      .split(X, y, groups=groups))
    pred = LinearRegression().fit(X[tr], y[tr]).predict(X[te])
    hold = {"Test_n": len(te), "Test_R2": r2_score(y[te], pred),
            "Test_RMSE": rmse(y[te], pred),
            "Test_MAE": mean_absolute_error(y[te], pred),
            "Baseline_RMSE": rmse(y[te], np.full(len(te), y[tr].mean()))}

    # repeated 5 fold cv: 20 different random splits, so the result does not depend on one lucky or unlucky split (in model 2.2).
    N_REPEATS = 20
    r2s, rm, ma, bs = [], [], [], []
    for rep in range(N_REPEATS):
        if groups is None:
            splitter = KFold(5, shuffle=True, random_state=SEED + rep).split(X)
        else:
            splitter = GroupKFold(5, shuffle=True, random_state=SEED + rep).split(X, y, groups)
        for a, b in splitter:
            p = LinearRegression().fit(X[a], y[a]).predict(X[b])
            r2s.append(r2_score(y[b], p)); rm.append(rmse(y[b], p))
            ma.append(mean_absolute_error(y[b], p))
            bs.append(rmse(y[b], np.full(len(b), y[a].mean())))
    cv = {"CV_R2": np.mean(r2s), "CV_R2_SD": np.std(r2s), "CV_RMSE": np.mean(rm),
          "CV_MAE": np.mean(ma), "CV_Baseline_RMSE": np.mean(bs)}
    return hold, cv


def run_regression(df, x_cols, y_col, title, tag, groups=None):
    print("\n\n" + "#" * 80); print(title); print("#" * 80)

    print("\nDESCRIPTIVE STATISTICS"); print("-" * 80)
    print(df[[y_col] + x_cols].describe().T.round(3))

    print("\nCORRELATION WITH DEPENDENT VARIABLE"); print("-" * 80)
    print(df[x_cols + [y_col]].corr()[y_col].drop(y_col)
          .sort_values(key=np.abs, ascending=False).round(3))

    # Eda check : confirm each variable chosen by logic with a scatterplot and Pearson correlation against the response
    print("\nPEARSON r AND p-VALUE FOR EACH VARIABLE (EDA check)"); print("-" * 80)
    fig_s, axs = plt.subplots(2, 4, figsize=(16, 7.5))
    for axx, col in zip(axs.ravel(), x_cols):
        r, p = stats.pearsonr(df[col], df[y_col])
        print(f"{col:28s} r = {r:6.3f}   p = {p:.4f}")
        axx.scatter(df[col], df[y_col], alpha=0.6)
        slope, intercept = np.polyfit(df[col], df[y_col], 1)
        line_x = np.linspace(df[col].min(), df[col].max(), 50)
        axx.plot(line_x, slope * line_x + intercept, "r--")
        axx.set(title=f"{col}\nr = {r:.2f}, p = {p:.3f}", xlabel=col, ylabel=y_col)
    fig_s.suptitle(f"{tag}: each explanatory variable vs {y_col}", fontsize=13)
    fig_s.tight_layout()
    fig_s.savefig(os.path.join(OUTPUT_DIR, f"{tag}_scatter_pearson.png"), dpi=150)

    Xc = sm.add_constant(df[x_cols], has_constant="add")
    vif = pd.DataFrame({"Variable": x_cols,
                        "VIF": [variance_inflation_factor(Xc.values, i + 1)
                                for i in range(len(x_cols))]})
    print("\nMULTICOLLINEARITY (VIF)"); print("-" * 80)
    print(vif.round(2).to_string(index=False))

    y = df[y_col]
    model = sm.OLS(y, Xc).fit()
    print("\nOLS REGRESSION RESULTS"); print("-" * 80)
    print(model.summary())

    if groups is not None:
        robust = sm.OLS(y, Xc).fit(cov_type="cluster", cov_kwds={"groups": groups})
        print("\nCLUSTER-ROBUST (by Match_ID) COEFFICIENTS"); print("-" * 80)
        print(pd.DataFrame({"coef": robust.params, "robust_SE": robust.bse,
                            "p_value": robust.pvalues}).round(4))

    std_beta = (model.params[x_cols] * df[x_cols].std() / y.std()).rename("Std_Beta")
    coef_table = pd.DataFrame({
        "Coef": model.params[x_cols], "Std_Beta": std_beta,
        "p_value": model.pvalues[x_cols],
        "CI_low": model.conf_int().loc[x_cols, 0],
        "CI_high": model.conf_int().loc[x_cols, 1]})
    coef_table["Significant_5%"] = np.where(coef_table["p_value"] < 0.05, "Yes", "No")
    print("\nCOEFFICIENT SUMMARY (sorted by |standardised beta|)"); print("-" * 80)
    print(coef_table.reindex(std_beta.abs().sort_values(ascending=False).index).round(4))

    resid, fitted = model.resid, model.fittedvalues
    bp_p = het_breuschpagan(resid, Xc)[1]
    sh_p = stats.shapiro(resid)[1]
    dw = durbin_watson(resid)
    n_infl = int((model.get_influence().cooks_distance[0] > 4 / len(df)).sum())
    print("\nASSUMPTION CHECKS"); print("-" * 80)
    print(f"Breusch-Pagan p = {bp_p:.4f} ->", "OK" if bp_p > 0.05 else "heteroscedasticity")
    print(f"Shapiro-Wilk  p = {sh_p:.4f} ->", "OK" if sh_p > 0.05 else "not perfectly normal")
    print(f"Durbin-Watson   = {dw:.3f} (about 2 is ideal)")
    print(f"Influential points (Cook's D > 4/n): {n_infl}")


    hold, cv = evaluate_split(df, x_cols, y_col, groups=groups)
    print("\nMODEL EVALUATION"); print("-" * 80)
    print(f"In-sample R2 = {model.rsquared:.4f} | Adj R2 = {model.rsquared_adj:.4f} | "
          f"RMSE = {rmse(y, fitted):.4f} | MAE = {mean_absolute_error(y, fitted):.4f}")
    print(f"F-test F = {model.fvalue:.3f}, p = {model.f_pvalue:.4g}")
    print(f"AIC = {model.aic:.2f} | BIC = {model.bic:.2f}")
    print(f"Hold-out 80/20 (n={hold['Test_n']}): R2 = {hold['Test_R2']:.4f}, "
          f"RMSE = {hold['Test_RMSE']:.4f}, MAE = {hold['Test_MAE']:.4f}, "
          f"baseline RMSE = {hold['Baseline_RMSE']:.4f}")
    print(f"5-fold CV x20: R2 = {cv['CV_R2']:.4f} (SD {cv['CV_R2_SD']:.4f}), RMSE = {cv['CV_RMSE']:.4f}, "
          f"MAE = {cv['CV_MAE']:.4f}, baseline RMSE = {cv['CV_Baseline_RMSE']:.4f}")

    # plottings
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    ax[0, 0].scatter(fitted, y, alpha=0.7)
    lo, hi = min(fitted.min(), y.min()), max(fitted.max(), y.max())
    ax[0, 0].plot([lo, hi], [lo, hi], "r--")
    ax[0, 0].set(title="Actual vs Predicted", xlabel="Predicted", ylabel="Actual")
    ax[0, 1].scatter(fitted, resid, alpha=0.7)
    ax[0, 1].axhline(0, color="r", ls="--")
    ax[0, 1].set(title="Residuals vs Fitted", xlabel="Fitted", ylabel="Residual")
    sm.qqplot(resid, line="45", fit=True, ax=ax[1, 0])
    ax[1, 0].set_title("Normal Q-Q Plot of Residuals")
    order = std_beta.sort_values().index
    ax[1, 1].barh(order, std_beta[order])
    ax[1, 1].axvline(0, color="k", lw=0.8)
    ax[1, 1].set(title="Standardised Coefficients", xlabel="Std. beta")
    fig.suptitle(title, fontsize=13); fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{tag}_diagnostics.png"), dpi=150)

    corr = df[x_cols + [y_col]].corr()
    f2, a2 = plt.subplots(figsize=(8, 6.5))
    im = a2.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    a2.set_xticks(range(len(corr))); a2.set_yticks(range(len(corr)))
    a2.set_xticklabels(corr.columns, rotation=60, ha="right")
    a2.set_yticklabels(corr.columns)
    f2.colorbar(im); a2.set_title(f"{tag}: Correlation Heatmap"); f2.tight_layout()
    f2.savefig(os.path.join(OUTPUT_DIR, f"{tag}_correlation.png"), dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close("all")


    return {"Model": tag, "n": len(df), "R2": model.rsquared,
            "Adj_R2": model.rsquared_adj, "RMSE": rmse(y, fitted),
            "MAE": mean_absolute_error(y, fitted), "F_p_value": model.f_pvalue,
            "Hold_R2": hold["Test_R2"], "Hold_RMSE": hold["Test_RMSE"],
            "CV_R2": cv["CV_R2"], "CV_RMSE": cv["CV_RMSE"],
            "Baseline_CV_RMSE": cv["CV_Baseline_RMSE"],
            "Significant_vars": ", ".join(coef_table.index[coef_table["p_value"] < 0.05])}



# running both models
res21 = run_regression(dataset_21, X21_COLS, Y21,
                       "LINEAR REGRESSION 2.1 - GOAL DIFFERENCE (104 matches)", "Model_2_1")
res22 = run_regression(dataset_22, X22_COLS, Y22,
                       "LINEAR REGRESSION 2.2 - GOALS SCORED (208 team-match rows)",
                       "Model_2_2", groups=groups22)



# completing algorithm comparison
# Linear regression is built first, then compared with other scikitlearn regressors using the same repeated 5 fold CV


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def compare_algorithms(df, x_cols, y_col, tag, groups=None, n_repeats=20):
    X, y = df[x_cols].values, df[y_col].values
    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=SEED),
        "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
    }
    rows = []
    for name, model in candidates.items():
        r2s, rmses, maes = [], [], []
        for rep in range(n_repeats):
            if groups is None:
                splitter = KFold(5, shuffle=True, random_state=SEED + rep).split(X)
            else:
                splitter = GroupKFold(5, shuffle=True, random_state=SEED + rep).split(X, y, groups)
            for a, b in splitter:
                pred = model.fit(X[a], y[a]).predict(X[b])
                r2s.append(r2_score(y[b], pred))
                rmses.append(rmse(y[b], pred))
                maes.append(mean_absolute_error(y[b], pred))
        rows.append({"Algorithm": name, "CV_R2": np.mean(r2s),
                     "CV_RMSE": np.mean(rmses), "CV_MAE": np.mean(maes)})
    table = pd.DataFrame(rows).sort_values("CV_RMSE").reset_index(drop=True)
    print(f"\nCOMPETING ALGORITHMS - {tag} (5-fold CV x{n_repeats})"); print("-" * 80)
    print(table.round(4).to_string(index=False))
    best = table.loc[0, "Algorithm"]
    print(f"Lowest CV RMSE: {best} ")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(table["Algorithm"], table["CV_RMSE"])
    ax.set(title=f"{tag}: competing algorithms (lower RMSE is better)", xlabel="Mean CV RMSE")
    ax.invert_yaxis(); fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{tag}_algorithm_comparison.png"), dpi=150)
    plt.close(fig)
    table.to_csv(os.path.join(OUTPUT_DIR, f"{tag}_algorithm_comparison.csv"), index=False)
    return table


cmp21 = compare_algorithms(dataset_21, X21_COLS, Y21, "Model_2_1")
cmp22 = compare_algorithms(dataset_22, X22_COLS, Y22, "Model_2_2", groups=groups22)

summary = pd.DataFrame([res21, res22])
print("\n\n" + "=" * 80); print("FINAL MODEL COMPARISON"); print("=" * 80)
print(summary.round(4).T.to_string(header=False))

dataset_21.to_csv(os.path.join(OUTPUT_DIR, "Dataset_2_1_used.csv"), index=False)
dataset_22.to_csv(os.path.join(OUTPUT_DIR, "Dataset_2_2_used.csv"), index=False)
summary.to_excel(os.path.join(OUTPUT_DIR, "Objective2_Results_Summary.xlsx"), index=False)
print("\nAll outputs saved in:", OUTPUT_DIR)
print("OBJECTIVE 2 ANALYSIS COMPLETE")
