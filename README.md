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

All explanatory variables are known before kick-off. Form variables (e.g. prior goals scored) use only matches played before the match being predicted.

## Folder structure

```
data/        FIFA_WC2026_Dataset_2_1_104rows.xlsx, FIFA_WC2026_Dataset_2_2_208rows.xlsx
notebooks/   Objective_2_Code.py – OLS models, assumption checks, hold-out test and repeated cross-validation
figures/     diagnostic and correlation charts produced by the script
```

## How to run

1. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
2. From the main folder, run:
   ```
   python3 notebooks/Objective_2_Code.py
   ```
3. Charts are saved to `figures/`. Close each chart window to let the script continue.

## Method

- Multiple linear regression (statsmodels OLS and scikit-learn `LinearRegression`)
- Assumption checks: VIF, Breusch-Pagan, Shapiro-Wilk, Durbin-Watson, Cook's distance
- Evaluation: 80/20 hold-out test and 5-fold cross-validation repeated 20 times, compared with a mean-only baseline
- Model 2.2 keeps both rows of the same match in the same fold (grouped by `Match_ID`)

## Current results

| Model               | R²    | Adj. R² | CV R² (5-fold × 20) | CV RMSE | Baseline RMSE |
| ------------------- | ----- | ------- | ------------------- | ------- | ------------- |
| 2.1 Goal difference | 0.508 | 0.466   | 0.332               | 1.555   | 2.016         |
| 2.2 Goals scored    | 0.257 | 0.227   | 0.129               | 1.270   | 1.386         |

## Data sources

- **Match results (scores):** [source, e.g. FBref "Scores & Fixtures"]. Each team's total goals were cross-checked against FIFA's official team statistics.
- **FIFA ranking:** FIFA Men's World Ranking, [release date – check it, e.g. 11 June 2026] – https://www.fifa.com/en/world-rankings
- **Squad market value and average age:** [source to be confirmed]
- **Rest days, prior goals, points per game and clean-sheet rates:** calculated from earlier World Cup 2026 matches only, so nothing comes from the match being predicted.
