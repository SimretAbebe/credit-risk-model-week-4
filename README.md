# # Credit Risk Probability Model for Alternative Data

## Project Overview

This project aims to build an end-to-end credit risk probability model using alternative transaction data from an eCommerce platform. The system is designed to help financial institutions estimate customer creditworthiness for buy-now-pay-later services when traditional credit history is unavailable.

The project involves data preprocessing, feature engineering, proxy target variable creation, machine learning model development, API deployment, and MLOps practices such as experiment tracking and CI/CD automation.

## Project Structure

* `data/` → stores raw and processed datasets
* `notebooks/` → contains exploratory data analysis notebooks
* `src/` → source code for preprocessing, training, prediction, and API services
* `tests/` → unit testing files
* `.github/workflows/` → CI/CD automation workflows
* `Dockerfile` and `docker-compose.yml` → containerization setup



## Credit Scoring Business Understanding

### 1. Basel II and the Need for Interpretable Models

The Basel II Accord emphasizes proper risk measurement, transparency, and regulatory compliance in financial systems. Because credit scoring models influence important financial decisions such as loan approvals and credit limits, banks must be able to explain how predictions are made.

An interpretable and well-documented model allows financial institutions to justify lending decisions, monitor model behavior, and comply with regulatory auditing requirements. In regulated environments, transparency is critical because stakeholders need to understand which factors contribute to customer risk predictions.

Therefore, simpler and explainable models are often preferred in financial contexts, even if more complex models may sometimes provide slightly better predictive performance.


### 2. Proxy Variables and Business Risks

The dataset used in this project does not contain a direct loan default label. Since supervised machine learning models require target labels for training, a proxy variable must be created to estimate customer credit risk behavior.

In this project, behavioral patterns such as Recency, Frequency, and Monetary (RFM) activity can be used to categorize customers into potential high-risk and low-risk groups. This proxy target acts as an estimated representation of default behavior.

However, proxy-based prediction introduces business risks because the generated labels may not perfectly represent actual loan repayment outcomes. Incorrect proxy labels may lead to rejecting reliable customers or approving risky customers, which can negatively affect customer trust, profitability, and fairness in lending decisions.


### 3. Trade-offs Between Interpretable and High-Performance Models

In regulated financial systems, there is an important trade-off between model interpretability and predictive performance.

Interpretable models such as Logistic Regression combined with Weight of Evidence (WoE) transformations are easier to explain, validate, and document. These models are preferred in many banking environments because regulators and business stakeholders can clearly understand the reasoning behind predictions.

On the other hand, high-performance models such as Gradient Boosting or XGBoost can capture more complex behavioral patterns and often achieve better predictive accuracy. However, these models are generally more difficult to interpret and may reduce transparency in decision-making.

Choosing the appropriate model therefore requires balancing regulatory compliance, explainability, fairness, and predictive performance based on the business context.
## Dataset

The dataset used in this project originates from the **Xente Challenge** (Kaggle competition). It can be downloaded from:

https://www.kaggle.com/competitions/xente-challenge/data

**Instructions:**
1. Sign in to Kaggle (create a free account if needed).
2. Accept the competition rules.
3. Download the `transactions.csv` (or provided zip) and place it in the `data/raw/` directory.
4. Run the preprocessing script `src/preprocess.py` to generate processed files in `data/processed/`.

## Project Conventions

- **Branching:** Use `main` for stable releases. Feature work should be done on `feature/<name>` branches and merged via pull requests.
- **Coding style:** Follow PEP 8, run `flake8` and `black` before committing. Type hints are encouraged.
- **Commit messages:** Follow the conventional commits spec (`type(scope): description`).
- **Testing:** Add unit tests in `tests/` and ensure `pytest` passes locally before CI.
