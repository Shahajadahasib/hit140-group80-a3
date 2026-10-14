# hit140-group80-a3

HIT140 Assessment 3 – Darwin Group 80 – Linear regression on FIFA World Cup 2026 data (Objective 2)

## Group members

- Yash Bhaveshbhai Balar – S398712
- Md Shahajada Hasib – S397670
- Gurcharan Singh – S377948
- Vikram Sharma – S406159

## Models

- **Model 2.1** – goal difference per match (104 rows, 8 pre-match variables)
- **Model 2.2** – goals scored by a team in a match (208 rows, 8 pre-match variables)
- 4 variable concepts are shared between the models (FIFA rank, host status, prior goals scored, prior goals conceded); the other 8 are unique.

All explanatory variables are known before kick-off. Form variables (e.g. prior goals scored) use only matches played before the match being predicted.

## Folder structure

```
data/        FIFA_WC2026_Dataset_2_1_104rows.csv / .xlsx, FIFA_WC2026_Dataset_2_2_208rows.csv / .xlsx
notebooks/   00_convert_to_csv.py  – converts the raw .xlsx files to .csv
             Objective_2_Code.py   – EDA, OLS models, assumption checks, evaluation,
                                     competing algorithms and optimisation
figures/     charts and result tables produced by the script
```

## How to run

1. Install the libraries (scikit-learn 1.6 or newer is needed for shuffled GroupKFold):
   ```
   pip install -r requirements.txt
   ```
2. From the main folder, run:
   ```
   python3 notebooks/Objective_2_Code.py
   ```
3. Charts and tables are saved to `figures/`. Close each chart window to let the script continue. A full run takes about 5–15 minutes.

## Method

1. **EDA:** descriptive statistics, scatterplot and Pearson r of each variable against the response, correlation heatmap
2. **Multiple linear regression** (statsmodels OLS and scikit-learn `LinearRegression`), regression equation, standardised coefficients
3. **Assumption checks:** VIF, Breusch-Pagan, Shapiro-Wilk, Durbin-Watson, Cook's distance; cluster-robust SEs for Model 2.2
4. **Evaluation:** 80/20 hold-out test and 5-fold cross-validation repeated 20 times (R², MSE, RMSE, NRMSE, MAE), compared with a mean-only baseline. Model 2.2 keeps both rows of the same match in the same fold (grouped by `Match_ID`).
5. **Competing algorithms:** Random Forest and Gradient Boosting under the same CV
6. **Optimisation round 1 (Weeks 8–9):** log transform, removing the most collinear variable, z-score scaling, Yeo-Johnson transform – one change at a time, all inside pipelines (no leakage)
7. **Optimisation round 2:** Yeo-Johnson + Ridge and Yeo-Johnson + Lasso. The best version is chosen with CV on the 80% training part only, then checked once on the 20% hold-out.

## Results

| Model                               | In-sample R² / Adj. R² | CV R² | CV RMSE | Baseline RMSE | Hold-out R² / RMSE |
| ----------------------------------- | ---------------------- | ----- | ------- | ------------- | ------------------ |
| 2.1 Goal difference (OLS)           | 0.508 / 0.466          | 0.332 | 1.555   | 2.016         | 0.355 / 1.793      |
| 2.2 Goals scored (OLS)              | 0.257 / 0.227          | 0.129 | 1.270   | 1.386         | 0.156 / 1.471      |
| 2.2 optimised (Yeo-Johnson + Lasso) | –                      | 0.190 | 1.227   | 1.386         | 0.190 / 1.441      |

- Linear regression beat Random Forest and Gradient Boosting for both models.
- Model 2.1: no optimisation step clearly beat the original model, so the original OLS is kept.
- Model 2.2: Yeo-Johnson + Lasso improved CV R² from 0.129 to 0.190 and lowered hold-out RMSE from 1.471 to 1.441.

## Data sources

- **Match results (scores):** official FIFA World Cup 2026 match results (https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026). All 48 teams' goal totals in our dataset match FIFA's official team statistics.
- **FIFA ranking:** FIFA/Coca-Cola Men's World Ranking, June 2026 release (published 11 June 2026, before the opening match, so it is pre-match data) – https://inside.fifa.com/fifa-world-ranking/men
- **Squad market value:** Transfermarkt squad values, June 2026 (used as ln(value A / value B))
- **Squad average age:** official FIFA World Cup 2026 squad lists (2 June 2026)
- **Rest days, prior goals, points per game and clean-sheet rates:** calculated from earlier World Cup 2026 matches only, so nothing comes from the match being predicted.

## AI use

Generative AI was used to explain concepts, review and debug code. See the AI Usage Declaration in the submission.
