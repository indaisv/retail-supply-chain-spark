# Retail Supply Chain Optimization using Apache Spark and Machine Learning

A Big Data project that uses Apache Spark's distributed processing capabilities to analyze retail sales data, predict weekly sales, and automate inventory restocking decisions.

---

## Project Overview

Retail organizations generate massive volumes of transactional data weekly. This project demonstrates how Apache Spark and Machine Learning can be combined to:
- Process large-scale retail datasets using distributed computing
- Predict weekly sales using Linear Regression (Spark MLlib)
- Automate restock decisions based on predicted demand
- Identify slow-moving inventory using KPI metrics

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Apache Spark (PySpark) | Distributed data processing |
| Spark MLlib | Machine learning (Linear Regression) |
| Spark SQL | Data querying and transformation |
| VectorAssembler | Feature engineering |
| Google Colab | Execution environment |
| Python | Programming language |

---

## Dataset

- **Source:** Walmart Sales Dataset (Kaggle)
- **Files:** `sales data-set.csv` | `Features data set.csv` | `stores data-set.csv`
- **Features used:** Store, Department, Temperature, Fuel Price, CPI, Unemployment, Weekly Sales

---

## System Architecture

```
Dataset Loading (3 CSV files)
        |
Distributed Join Operations (Spark)
        |
Data Preprocessing and Type Casting
        |
Feature Engineering (VectorAssembler)
        |
Linear Regression Model Training
        |
Sales Prediction Generation
        |
Restock Decision Logic
        |
Model Evaluation (RMSE and R2)
```

---

## How to Run

**Option 1 — Google Colab (Recommended)**
```python
# Install PySpark in Colab
!pip install pyspark

# Upload dataset files and run retail_supply_chain.py
```

**Option 2 — Local Setup**
```bash
pip install pyspark
python retail_supply_chain.py
```

---

## Key Features

**1. Distributed Data Pipeline**
Joins three large retail datasets using Spark distributed joins, demonstrating scalable data integration.

**2. Inventory KPIs**
- Stock Turnover Ratio — measures how efficiently inventory is sold
- Slow-Moving Inventory Detection — flags stores below average sales threshold

**3. Automated Restock Logic**
```
If Predicted Sales > Current Weekly Sales → Restock_Required = 1
Else → Restock_Required = 0
```

**4. Model Evaluation**

| Metric | Value |
|--------|-------|
| RMSE | ~22,773 |
| R2 Score | ~0.001 |

Note: The low R2 score highlights an important data science insight — temperature, fuel price, CPI, and unemployment alone are insufficient predictors of weekly sales. Future improvements would include promotional data, holiday flags, and store type as additional features.

---

## Business Impact

- Reduces stock shortages through demand forecasting
- Minimizes overstocking and excess inventory costs
- Enables data-driven inventory decisions at scale
- Demonstrates real-world Big Data analytics in retail operations

---

## Author

**Viraj Indais**
- Email: indaisviraj@gmail.com
- LinkedIn: https://linkedin.com/in/viraj-indais
- GitHub: https://github.com/indaisv
