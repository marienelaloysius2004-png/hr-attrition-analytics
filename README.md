# HR Attrition Analytics: Who Leaves, Why, and Who Is Next

**Live dashboard:** [Overview](https://public.tableau.com/views/HRAttritionAnalytics_17928609810060/Overview) | [Flight Risk watchlist](https://public.tableau.com/views/HRAttritionAnalytics_17928609810060/FlightRisk)
**Recommendations memo:** [`Attrition_Recommendations.pdf.pdf`](Attrition_Recommendations.pdf.pdf)

## Problem
237 of 1,470 employees left, an attrition rate of **16.1%**, costing an estimated **$6.8M** in replacement costs (assuming 50% of annual salary per leaver). This project finds where attrition is concentrated, what drives it, and which current employees are most at risk, so HR can act early.

## Data
[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (Kaggle). A public sample of 1,470 employees and 35 fields. It is fictional sample data, not a real company's records. The CSV is not included in this repo; download it from Kaggle.

## Method
1. **Cleaning:** removed 3 constant columns, checked for missing values and duplicates (none found), and created tenure, age, pay and promotion bands.
2. **Segment analysis:** attrition rate for 14 employee attributes (`attrition_by_segment.csv`).
3. **Model:** logistic regression (explainable) compared with a random forest, using an 80/20 train/test split.
4. **Flight-risk scoring:** cross-validated risk scores for every employee, so no one is scored by a model that saw their own outcome. The top 15% of scores are "High" risk.
5. **Cost estimate:** replacement cost at 50% of annual salary; savings scenario at 20% retention of high-risk employees.
6. **Dashboard:** three Tableau Public pages: Overview, Why People Leave, Flight Risk.

## Key findings
| Finding | Result |
| --- | --- |
| Overall attrition | 16.1% (237 of 1,470) |
| Overtime vs no overtime | 30.5% vs 10.4% (2.9x higher) |
| Sales Representatives | 39.8% |
| Ages 18 to 25 | 35.8% |
| 0 to 1 years at the company | 34.9% |
| Lowest pay quartile | 29.3% |
| Model accuracy (ROC-AUC, unseen data) | 0.80 logistic regression, 0.79 random forest |
| Top 3 drivers | Overtime, years at company, frequent business travel |
| High-risk current employees | 97, exposing about $3.07M in replacement cost |
| Savings if 20% are retained | about $613,664 |

## Recommendations
1. Audit and cap overtime in high-attrition roles.
2. Strengthen the first year with 30/60/90-day check-ins and a buddy.
3. Review pay, targets and career path for Sales Representatives.
4. Benchmark pay in the lowest quartile against the market.
5. Hold stay conversations with the 97 high-risk employees within 30 days.

Details, owners and success measures are in the [memo](Attrition_Recommendations.pdf.pdf).

## Limitations
- Public sample data; patterns are correlations, not proven causes.
- Age, tenure and pay overlap, so their effects should not be added together.
- The 50% replacement cost and 20% retention rate are assumptions.
- The model used age, gender and marital status. Before any real use, these should be removed and the model tested for bias.

## How to reproduce
1. Download the dataset from Kaggle and put the CSV in the same folder as the script.
2. Open [Google Colab](https://colab.research.google.com), upload the CSV and `hr_attrition_analysis.py`.
3. Run `%run hr_attrition_analysis.py`.
4. Results appear in an `output/` folder: charts, `hr_dashboard_data.csv`, `flight_risk_watchlist.csv`, `top_drivers.csv` and `key_findings.txt`.

Requires Python with pandas, numpy, matplotlib and scikit-learn (all pre-installed in Colab).

## Tools
Python (pandas, scikit-learn, matplotlib), Tableau Public.

## Repo contents
- `hr_attrition_analysis.py`: the full analysis script
- `output/`: charts and result files
- `memo/`: recommendations memo (PDF)
- `screenshots/`: dashboard screenshots
