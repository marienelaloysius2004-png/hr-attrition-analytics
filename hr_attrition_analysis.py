"""
HR ATTRITION ANALYSIS  -  flagship portfolio project
=====================================================
What this script does, in order:
  1. Finds and loads the IBM HR attrition CSV in the same folder
  2. Cleans it and builds bands (tenure, age, income, promotion)
  3. Calculates attrition rate for every segment and saves charts
  4. Trains an explainable model (logistic regression) + a comparison model
  5. Scores every employee for flight risk (honest, cross-validated scores)
  6. Estimates the cost of attrition and of a retention plan
  7. Saves everything you need for the dashboard and the memo into ./output

Needs: pandas, numpy, matplotlib, scikit-learn  (all pre-installed in Google Colab)
"""

import glob
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# SETTINGS YOU CAN CHANGE (assumptions you will explain in your memo)
# ----------------------------------------------------------------------------
REPLACEMENT_COST_RATE = 0.50   # cost to replace one employee = 50% of annual salary
RETENTION_SUCCESS = 0.20       # a retention plan saves 20% of high-risk people
HIGH_RISK_TOP_PCT = 0.15       # top 15% of risk scores = "High"
MEDIUM_RISK_TOP_PCT = 0.40     # next 25% (up to top 40%) = "Medium"
OUT = "output"
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. LOAD THE DATA
# ----------------------------------------------------------------------------
print("\n=== STEP 1: LOADING DATA ===")


def find_csv():
    for f in glob.glob("*.csv"):
        try:
            if "Attrition" in pd.read_csv(f, nrows=1).columns:
                return f
        except Exception:
            pass
    raise SystemExit(
        "\nERROR: I could not find the HR dataset.\n"
        "Make sure the CSV file (WA_Fn-UseC_-HR-Employee-Attrition.csv) is in the same "
        "folder as this script, or uploaded in Colab's file panel on the left."
    )


CSV = find_csv()
df = pd.read_csv(CSV)
print(f"Loaded '{CSV}': {df.shape[0]} employees, {df.shape[1]} columns")

required = ["Attrition", "OverTime", "MonthlyIncome", "YearsAtCompany", "Age",
            "Department", "JobRole", "YearsSinceLastPromotion"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"\nERROR: this CSV is missing columns: {missing}. "
                     "Use the IBM HR Analytics Employee Attrition dataset from Kaggle.")

# ----------------------------------------------------------------------------
# 2. CLEAN + FEATURE ENGINEERING
# ----------------------------------------------------------------------------
print("\n=== STEP 2: CLEANING ===")
before = df.shape[1]
df = df.drop(columns=[c for c in ["EmployeeCount", "Over18", "StandardHours"] if c in df.columns])
print(f"Dropped {before - df.shape[1]} useless columns (same value for everyone)")
print(f"Missing values in whole dataset: {int(df.isna().sum().sum())}")
print(f"Duplicate rows: {int(df.duplicated().sum())}")

df["AttritionFlag"] = (df["Attrition"] == "Yes").astype(int)

df["TenureBand"] = pd.cut(df["YearsAtCompany"], [-1, 1, 3, 5, 10, 100],
                          labels=["1. 0-1 yrs", "2. 2-3 yrs", "3. 4-5 yrs", "4. 6-10 yrs", "5. 10+ yrs"])
df["AgeBand"] = pd.cut(df["Age"], [17, 25, 35, 45, 55, 100],
                       labels=["1. 18-25", "2. 26-35", "3. 36-45", "4. 46-55", "5. 56+"])
df["IncomeBand"] = pd.qcut(df["MonthlyIncome"], 4,
                           labels=["1. Q1 (lowest pay)", "2. Q2", "3. Q3", "4. Q4 (highest pay)"])
df["PromotionBand"] = pd.cut(df["YearsSinceLastPromotion"], [-1, 0, 2, 5, 100],
                             labels=["1. Promoted this year", "2. 1-2 yrs ago", "3. 3-5 yrs ago", "4. 6+ yrs ago"])

company_rate = df["AttritionFlag"].mean() * 100
print(f"Overall attrition rate: {company_rate:.1f}%  "
      f"({df['AttritionFlag'].sum()} of {len(df)} employees left)")

# ----------------------------------------------------------------------------
# 3. ATTRITION BY SEGMENT  (this is the "where is the problem" analysis)
# ----------------------------------------------------------------------------
print("\n=== STEP 3: ATTRITION BY SEGMENT ===")
segment_cols = ["Department", "JobRole", "OverTime", "BusinessTravel", "MaritalStatus",
                "Gender", "JobSatisfaction", "WorkLifeBalance", "EnvironmentSatisfaction",
                "TenureBand", "AgeBand", "IncomeBand", "PromotionBand", "StockOptionLevel"]
segment_cols = [c for c in segment_cols if c in df.columns]

rows = []
for col in segment_cols:
    g = df.groupby(col, observed=True)["AttritionFlag"].agg(["count", "sum", "mean"])
    for group, r in g.iterrows():
        rows.append({"Segment": col, "Group": str(group), "Employees": int(r["count"]),
                     "Leavers": int(r["sum"]), "AttritionRate": round(r["mean"] * 100, 1)})
seg = pd.DataFrame(rows)
seg.to_csv(f"{OUT}/attrition_by_segment.csv", index=False)


def bar_chart(col, title, filename, sort_by_rate=False):
    g = df.groupby(col, observed=True)["AttritionFlag"].mean() * 100
    if sort_by_rate:
        g = g.sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(g.index.astype(str), g.values, color="#3b6fb6")
    ax.axvline(company_rate, color="#d9534f", linestyle="--", label=f"Company average ({company_rate:.1f}%)")
    for i, v in enumerate(g.values):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Attrition rate (%)")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/{filename}", dpi=150)
    plt.show()


bar_chart("OverTime", "Attrition rate: overtime vs no overtime", "chart_overtime.png")
bar_chart("JobRole", "Attrition rate by job role", "chart_job_role.png", sort_by_rate=True)
bar_chart("Department", "Attrition rate by department", "chart_department.png", sort_by_rate=True)
bar_chart("TenureBand", "Attrition rate by tenure", "chart_tenure.png")
bar_chart("IncomeBand", "Attrition rate by pay quartile", "chart_income.png")
bar_chart("AgeBand", "Attrition rate by age group", "chart_age.png")
bar_chart("JobSatisfaction", "Attrition rate by job satisfaction (1=low, 4=high)", "chart_job_satisfaction.png")
bar_chart("PromotionBand", "Attrition rate by time since last promotion", "chart_promotion.png")

# ----------------------------------------------------------------------------
# 4. PREDICTIVE MODEL
# ----------------------------------------------------------------------------
print("\n=== STEP 4: BUILDING THE MODEL ===")
y = df["AttritionFlag"]
drop_cols = ["Attrition", "AttritionFlag", "EmployeeNumber",
             "TenureBand", "AgeBand", "IncomeBand", "PromotionBand"]
X = df.drop(columns=drop_cols, errors="ignore")
X = pd.get_dummies(X, drop_first=True).astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

logit = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))
logit.fit(X_train, y_train)
logit_auc = roc_auc_score(y_test, logit.predict_proba(X_test)[:, 1])

forest = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                class_weight="balanced_subsample", random_state=42)
forest.fit(X_train, y_train)
forest_auc = roc_auc_score(y_test, forest.predict_proba(X_test)[:, 1])

print(f"Logistic regression  ROC-AUC: {logit_auc:.3f}   (explainable - used for the dashboard)")
print(f"Random forest        ROC-AUC: {forest_auc:.3f}   (comparison)")
print("(ROC-AUC: 0.5 = coin flip, 1.0 = perfect. 0.75+ is useful for HR data.)")
print("\nLogistic regression, test set report:")
print(classification_report(y_test, logit.predict(X_test), target_names=["Stayed", "Left"]))

# ----- What drives attrition? (standardised coefficients -> odds ratios)
logit.fit(X, y)
coefs = pd.Series(logit.named_steps["logisticregression"].coef_[0], index=X.columns)
drivers = pd.DataFrame({
    "Feature": coefs.index,
    "Coefficient": coefs.values,
    "OddsRatioPer1SD": np.exp(coefs.values),
})
drivers["Effect"] = np.where(drivers["Coefficient"] > 0, "Raises attrition risk", "Lowers attrition risk")
drivers = drivers.reindex(drivers["Coefficient"].abs().sort_values(ascending=False).index)
drivers.round(3).to_csv(f"{OUT}/top_drivers.csv", index=False)

top = drivers.head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.barh(top["Feature"], top["Coefficient"],
        color=["#d9534f" if c > 0 else "#5cb85c" for c in top["Coefficient"]])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Top 12 drivers of attrition\n(red = raises risk, green = protects)", fontweight="bold")
ax.set_xlabel("Model coefficient (standardised)")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/chart_top_drivers.png", dpi=150)
plt.show()

# ----------------------------------------------------------------------------
# 5. FLIGHT-RISK SCORES FOR EVERY EMPLOYEE (cross-validated = honest scores)
# ----------------------------------------------------------------------------
print("\n=== STEP 5: FLIGHT-RISK SCORING ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df["RiskScore"] = cross_val_predict(logit, X, y, cv=cv, method="predict_proba")[:, 1].round(3)

pct = df["RiskScore"].rank(pct=True)
df["RiskBand"] = np.where(pct >= 1 - HIGH_RISK_TOP_PCT, "High",
                          np.where(pct >= 1 - MEDIUM_RISK_TOP_PCT, "Medium", "Low"))

band = df.groupby("RiskBand")["AttritionFlag"].agg(["count", "mean"])
band["mean"] = (band["mean"] * 100).round(1)
print("Actual attrition rate inside each risk band (proves the model works):")
print(band.rename(columns={"count": "Employees", "mean": "ActualAttrition%"}))

# ----------------------------------------------------------------------------
# 6. COST OF ATTRITION
# ----------------------------------------------------------------------------
print("\n=== STEP 6: COST OF ATTRITION ===")
df["EstReplacementCost"] = (df["MonthlyIncome"] * 12 * REPLACEMENT_COST_RATE).round(0)

actual_cost = df.loc[df["AttritionFlag"] == 1, "EstReplacementCost"].sum()
current_high = df[(df["AttritionFlag"] == 0) & (df["RiskBand"] == "High")]
exposure = current_high["EstReplacementCost"].sum()
savings = exposure * RETENTION_SUCCESS

print(f"Cost of people who already left: ${actual_cost:,.0f}")
print(f"Current employees in 'High' risk: {len(current_high)}")
print(f"Replacement cost exposed if they ALL left: ${exposure:,.0f}")
print(f"Savings if a retention plan keeps {RETENTION_SUCCESS:.0%} of them: ${savings:,.0f}")

# ----------------------------------------------------------------------------
# 7. SAVE FILES FOR THE DASHBOARD
# ----------------------------------------------------------------------------
print("\n=== STEP 7: SAVING FILES ===")
df.to_csv(f"{OUT}/hr_dashboard_data.csv", index=False)

watch_cols = [c for c in ["EmployeeNumber", "Department", "JobRole", "OverTime", "MonthlyIncome",
                          "YearsAtCompany", "YearsSinceLastPromotion", "JobSatisfaction",
                          "RiskScore", "RiskBand", "EstReplacementCost"] if c in df.columns]
current_high.sort_values("RiskScore", ascending=False)[watch_cols].to_csv(
    f"{OUT}/flight_risk_watchlist.csv", index=False)

# ----- Auto-written findings (copy these numbers into your memo)
big = seg[seg["Employees"] >= 30].sort_values("AttritionRate", ascending=False).head(6)
ot = df.groupby("OverTime")["AttritionFlag"].mean() * 100
lines = [
    "KEY FINDINGS (numbers to paste into your memo)",
    "=" * 50,
    f"- Headcount analysed: {len(df)}. Leavers: {int(df['AttritionFlag'].sum())}. "
    f"Overall attrition: {company_rate:.1f}%.",
    f"- Overtime: {ot.get('Yes', float('nan')):.1f}% attrition with overtime vs "
    f"{ot.get('No', float('nan')):.1f}% without "
    f"({ot.get('Yes', 0) / max(ot.get('No', 1), 0.01):.1f}x higher).",
    "- Highest-attrition segments (min. 30 employees):",
]
for _, r in big.iterrows():
    lines.append(f"    * {r['Segment']} = {r['Group']}: {r['AttritionRate']}% "
                 f"({r['Leavers']} of {r['Employees']} left)")
lines += [
    f"- Model: logistic regression, ROC-AUC {logit_auc:.2f} on unseen data "
    f"(random forest {forest_auc:.2f}).",
    f"- Top 3 risk drivers: {', '.join(drivers.head(3)['Feature'])}.",
    f"- Cost of past attrition (assuming {REPLACEMENT_COST_RATE:.0%} of annual salary per leaver): "
    f"${actual_cost:,.0f}.",
    f"- {len(current_high)} current employees are High flight-risk, exposing ${exposure:,.0f}. "
    f"Retaining {RETENTION_SUCCESS:.0%} would save about ${savings:,.0f}.",
]
text = "\n".join(lines)
with open(f"{OUT}/key_findings.txt", "w") as f:
    f.write(text)

print(text)
print("\nDONE. Files saved in the 'output' folder:")
for name in sorted(os.listdir(OUT)):
    print("   ", name)
