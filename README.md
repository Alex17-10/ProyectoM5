> 🌐 **Note:** The Spanish version of this README is available as a PDF. Some content (charts, diagrams, screenshots) is easier to follow in the PDF version.
>
> 🌐 **Nota:** La versión en español de este README está disponible en formato PDF. Algunos contenidos (gráficas, diagramas, capturas) se ven más claros en la versión PDF.

---

# Credit Default Prediction

## Integrative Project

An end-to-end Data Science project developed for the Data and Analytics team of a financial institution. An Advanced Junior Data Scientist builds, deploys and monitors a predictive model that, based on credit history, anticipates whether a new customer will pay on time or default.

The repository covers the complete MLOps cycle: data quality and preparation, EDA, feature engineering, model training and selection, deployment as an API (FastAPI + Docker), prediction submission, data drift monitoring, and a Streamlit visualization dashboard.

## Table of Contents

0. [Quick Execution (Step by Step)](#0-quick-execution-step-by-step)
1. [General Description and Business Case](#1-general-description-and-business-case)
2. [Overall Project Flow](#2-overall-project-flow)
3. [Architecture and Folder Structure](#3-architecture-and-folder-structure)
4. [Input Data and Preparation: Cargar_datos.py](#4-input-data-and-preparation-cargar_datospy)
5. [Exploratory Data Analysis (EDA): Comprension_eda.py](#5-exploratory-data-analysis-eda-comprension_edapy)
6. [Feature Engineering: ft_engineering.py](#6-feature-engineering-ft_engineeringpy)
7. [Model Training and Evaluation: model_training_evaluation.py](#7-model-training-and-evaluation-model_training_evaluationpy)
8. [Model Deployment: model_deploy.py](#8-model-deployment-model_deploypy)
9. [Prediction API (FastAPI): API.py](#9-prediction-api-fastapi-apipy)
10. [Sending Records to the API: enviar_predicciones_API.py](#10-sending-records-to-the-api-enviar_predicciones_apipy)
11. [Model Monitoring and Data Drift: model_monitoring.py](#11-model-monitoring-and-data-drift-model_monitoringpy)
12. [Streamlit Application: APP.py](#12-streamlit-application-apppy)
13. [Methodological Considerations](#13-methodological-considerations)
14. [Containerization with Docker](#14-containerization-with-docker)
15. [Version Control with Git and GitHub](#15-version-control-with-git-and-github)
16. [Technology Stack](#16-technology-stack)
17. [Conclusion](#17-conclusion)

---

## 0. Quick Execution (Step by Step)

Reference guide for project review. The detail of each stage is developed in the following sections.

**Fundamental rule:** all scripts must stay in the project root folder (alongside the files and folders they consume) and be run from that folder. Paths are built relative to the location of the script itself, so moving it, renaming artifacts or running it from another location causes file-not-found errors. The root folder can live at any path on the computer; the only thing that must be preserved is the internal structure.

### Environment Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

### Execution Order

| # | Command | What it does |
|---|---|---|
| 1 | `python Cargar_datos.py` | Cleans `Base_de_datos.csv` and generates `Base_de_datos_preparada.csv` |
| 2 | `python Comprension_eda.py` | Regenerates the full EDA (reports, CSV files and 10 charts) |
| 3 | `python ft_engineering.py` | Generates transformed train/test sets and the preprocessing pipeline |
| 4 | `python model_training_evaluation.py` | Trains and compares the 3 models, saves `modelo_final.joblib` |
| 5 | `python model_deploy.py` | Applies the final model and generates validation predictions/reports |
| 6 | `uvicorn API:app --host 127.0.0.1 --port 8000` | Starts the API; leave it running in Terminal 1 |
| 7 | Test `POST /predict/batch` at `/docs` with the sample JSON (section 9) | Manual check that the service responds correctly |
| 8 | In Terminal 2, without closing the API: `python enviar_predicciones_API.py` | Sends up to 100 records to the batch endpoint and generates `predicciones_api_monitoring.csv` |
| 9 | `python model_monitoring.py` | Validates the predictions, computes metrics and data drift |
| 10 | `python -m streamlit run APP.py` | Starts the monitoring visualization dashboard |
| 11 (optional) | `docker build -t modelo-creditos-api .` / `docker run -d -p 8000:8000 ...` | Packages and serves the API in a container, as an alternative to step 6 |

```
Terminal 1                          Terminal 2
───────────                         ───────────
uvicorn API:app                     (wait for the API to be up)
--host 127.0.0.1 --port 8000   →    python enviar_predicciones_API.py
(stays open)                        (runs and finishes)
```

The API must stay active while `/predict/batch` is tested from `/docs` and while `enviar_predicciones_API.py` runs. It can be closed afterward, since `model_monitoring.py` and `APP.py` only read the files that have already been generated.

---

## 1. General Description and Business Case

You have started working on the Data and Analytics team of a financial company as an Advanced Junior Data Scientist. Your first assignment is to develop a predictive model using machine learning techniques, based on historical credit information, with the goal of anticipating the behavior of new users.

The company works under a structured project scheme in which every initiative must follow a strictly defined folder architecture. This structure cannot be modified, because production deployment processes are automated through validation pipelines in Jenkins. Any change in folder organization could cause significant delays in going to production.

The target variable used in the project is `Pago_atiempo` (paid on time). The classification used by the scripts interprets class 0 as possible default and class 1 as paid on time. The problem has an imbalanced class distribution, so the analysis must not rely on Accuracy alone.

**Business priority:** minimize the risk of originating credits that will not be repaid. The model selection criterion prioritizes recall of class 0 over overall accuracy.

## 2. Overall Project Flow

| Stage | Main script | Main input | Main output |
|---|---|---|---|
| Preparation and quality | `Cargar_datos.py` | `Base_de_datos.csv` | `Base_de_datos_preparada.csv` |
| EDA | `Comprension_eda.py` | `Base_de_datos_preparada.csv` | EDA report, CSV files and 10 charts |
| Feature engineering | `ft_engineering.py` | `Base_de_datos_preparada.csv` | Transformed train/test sets and pipeline |
| Training | `model_training_evaluation.py` | Transformed train/test sets | Evaluated models and `modelo_final.joblib` |
| Deployment | `model_deploy.py` | Model + pipeline + data | Predictions and reports |
| Service | `API.py` | Model + pipeline | Prediction endpoints |
| Submission to API | `enviar_predicciones_API.py` | `Base_de_datos.csv` | `predicciones_api_monitoring.csv` |
| Monitoring | `model_monitoring.py` | Prepared base + API predictions | Metrics, drift and temporal analysis |
| Dashboard | `APP.py` | Monitoring results | Streamlit interface |

The scripts build their paths from their own location or from relative paths: moving a file, renaming an artifact or running from another folder can cause file-not-found errors.

## 3. Architecture and Folder Structure

```
proyecto-credito-ml/
│
├── Base_de_datos.csv                        # Raw dataset (10,763 rows x 23 columns)
├── Base_de_datos_preparada.csv              # Output of Cargar_datos.py
├── Observaciones_calidad_datos.csv          # Data quality summary
│
├── Cargar_datos.py
├── Comprension_eda.py
├── ft_engineering.py
├── model_training_evaluation.py
├── model_deploy.py
├── API.py
├── enviar_predicciones_API.py
├── model_monitoring.py
├── APP.py
│
├── resultados_eda_entrega/                  # EDA reports, CSVs and graficos/ (10 images)
├── resultados_feature_engineering/          # X/y train-test, pipeline_preprocesamiento.joblib
├── resultados_model_training_evaluation/    # Models, confusion matrices, ROC curve
├── resultados_model_deploy/                 # Deployment predictions and reports
├── resultados_model_monitoring/             # Metrics, drift, alerts and temporal analysis
│   └── data_drift_graficos/
│
├── modelo_final.joblib                      # Selected model (root and in results)
├── predicciones_api_monitoring.csv          # History of predictions served by the API
│
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

This structure must not be modified: the Jenkins validation pipelines depend on it for going to production.

> **Figure (see PDF):** architecture diagram showing the flow of artifacts and scripts end to end, from `Base_de_datos.csv` through `Cargar_datos.py`, `ft_engineering.py`, `model_training_evaluation.py`, `API.py` (FastAPI + Docker), `model_monitoring.py` (KS · PSI · Jensen-Shannon · Chi²) to `APP.py` (Streamlit).

## 4. Input Data and Preparation: `Cargar_datos.py`

The original file `Base_de_datos.csv` contains 10,763 records and 23 columns. The target variable `Pago_atiempo` has 10,252 records of class 1 and 511 of class 0, roughly 95.25% and 4.75% respectively. This is a strong imbalance that is preserved during the analysis and handled with per-class and balanced metrics instead of relying on global Accuracy.

`Cargar_datos.py` works as a data quality control barrier: it reads the data, reviews dimensions, types, nulls and duplicates, and applies specific validations on dates, ages, financial variables, scores, balances, categories and the target. When it finds clearly invalid values it converts them to nulls, instead of inventing a replacement. It does not delete records en masse: it keeps the structure and leaves the detected problems ready for later stages to handle.

### Validations Applied

- Ages outside the 18–100 range
- Negative values in variables where they are not valid
- Salaries less than or equal to zero
- Installments greater than salary
- Negative scores
- Inconsistent balances (e.g. principal balance or arrears balance greater than total balance)
- `tendencia_ingresos` categories outside the three expected ones (Creciente, Decreciente, Estable)
- Invalid values of `Pago_atiempo`
- Extreme values using the interquartile range (IQR) criterion

The script also creates initial derived variables: loan year, month, day of week and hour; installment-to-salary ratio; debt-to-salary ratio; estimated outstanding balance. Resulting infinite values are converted to nulls.

### Documented Execution Results

| Indicator | Value |
|---|---|
| Records | 10,763 |
| Columns | 23 |
| Initial null values detected | 7,175 |
| Fully duplicated rows | 0 |
| Ages out of range (18–100) | 150 |
| Negative scores | 135 |
| Negative Datacrédito scores | 1 |
| Salaries ≤ 0 | 24 |
| Installments greater than salary | 11 |
| `tendencia_ingresos` values outside category | 58 |

**Outputs:** `Base_de_datos_preparada.csv` and `Observaciones_calidad_datos.csv`.

**Why before the EDA?** To prevent the exploratory analysis from mixing original data with already identifiable errors. This way the EDA receives a consistent version and can focus on describing distributions and patterns, instead of repeating basic integrity checks.

## 5. Exploratory Data Analysis (EDA): `Comprension_eda.py`

It takes `Base_de_datos_preparada.csv` and builds a reproducible exploratory analysis. On start, it deletes and recreates the `resultados_eda_entrega/` folder so that the results of one run do not mix with files from a previous run.

The EDA does not train the model: it describes the data structure, characterizes numeric and categorical variables, reviews the target, studies relationships with `Pago_atiempo`, looks at correlations, detects atypical behavior, and reviews the time dimension.

### What It Analyzes

- General summary (records, columns, cells, nulls, variable classification)
- Data quality per column and target distribution
- Descriptive statistics and frequencies of categorical variables
- Comparison of numeric and categorical variables by `Pago_atiempo`
- Correlations between numeric variables and with the target
- Extreme values and temporal behavior of loans

### The 10 Generated Charts (`resultados_eda_entrega/graficos/`)

> **All charts are shown in the PDF version.** Descriptions:

1. `01_distribuciones_numericas.png`: histograms of the main numeric variables (loan capital, term, age, salary, scores, balances). Strong concentration and skewness in financial variables, and extreme outliers in `salario_cliente` and `total_otros_prestamos`.
2. `02_boxplots_numericos.png`: comparative boxplots of numeric variables on the same scale, showing outliers far from the rest in `salario_cliente` and `total_otros_prestamos`.
3. `03_distribucion_target.png`: distribution of `Pago_atiempo`: 10,252 on-time payment records versus 511 not paid on time. It is the most direct chart to show the target imbalance (counts, not probabilities of future predictions).
4. `04_frecuencias_categoricas.png`: frequencies of `tipo_credito`, `tipo_laboral` and `tendencia_ingresos`. `tipo_credito` is dominated by two categories; `tipo_laboral` is concentrated in "Empleado"; `tendencia_ingresos` is concentrated in "Creciente".
5. `05_numericas_por_target.png`: boxplots of numeric variables (loan capital, term, age, salary, other loans, agreed installment) split by `Pago_atiempo` class, to observe distribution differences between payers and defaulters.
6. `06_categoricas_por_target.png`: percentage of `Pago_atiempo` within each category of `tipo_credito`, `tipo_laboral` and `tendencia_ingresos`. Credit type code 6 shows a notably higher default proportion than the rest.
7. `07_matriz_correlacion.png`: correlation matrix of all numeric variables, including derived ones. `puntaje` shows the highest correlation with `Pago_atiempo`, which reinforces the decision to exclude it because of possible information leakage.
8. `08_relaciones_financieras.png`: relationship between `salario_cliente` and `capital_prestado`. The point cloud shows a poorly defined relationship, strongly affected by extreme salary values.
9. `09_comportamiento_puntajes.png`: behavior of `puntaje` by `Pago_atiempo`: customers who pay on time (class 1) are concentrated in high scores, while class 0 shows a much wider and lower distribution.
10. `10_comportamiento_temporal.png`: number of loans per month. A peak is seen between December and January, followed by a sustained downward trend in the following months.

### Reports and CSV Files Generated

`Reporte_EDA.xlsx`, `Reporte_EDA.txt`, `resumen_inicial.csv`, `resumen_outliers.csv`, `resumen_temporal.csv`, `calidad_datos.csv`, `clasificacion_variables.csv`, `distribucion_target.csv`, `estadisticas_numericas.csv`, `frecuencias_categoricas.csv`, `comparacion_numerica_target.csv`, `comportamiento_categoricas_target.csv`, `correlaciones.csv`, `correlaciones_target.csv`, `hallazgos.csv`, `matriz_correlacion.csv`.

### Findings Passed to the Next Stage

The target is strongly imbalanced; there are missing values and financial behaviors that must be handled in preprocessing; `puntaje` is identified as a potentially problematic variable because of possible information leakage and is excluded from modeling (see sections 6 and 13).

*Correlation, differences between groups, or high later importance of a variable should not be automatically interpreted as causality. The EDA describes observed relationships and supports technical decisions; it does not prove that a variable causes default.*

## 6. Feature Engineering: `ft_engineering.py`

It converts the prepared data into a representation suitable for training models: the model cannot directly receive a mix of dates, text, missing values and different financial scales.

### Feature Creation

- `fecha_prestamo` (loan date) is transformed into year, month, day, day of week, week of year, quarter, hour and a binary weekend variable; the original date is dropped afterward.
- Derived financial variables: `relacion_cuota_salario`, `relacion_deuda_salario`, `saldo_pendiente_estimado`.
- `tipo_credito` is converted to an explicit categorical variable.

**Separation and leakage prevention:** `Pago_atiempo` is separated as the target; `puntaje` is explicitly excluded because of possible data leakage (see section 13).

**Train/test split:** `train_test_split` with 80% training / 20% test, `random_state=42`, `stratify=y`. Over 10,763 records this is equivalent to 8,610 training records and 2,153 test records.

### Preprocessing (`ColumnTransformer`)

- Numeric → median imputation + `StandardScaler`
- Categorical → imputation with the most frequent category + `OneHotEncoder(handle_unknown="ignore")`
- The preprocessor is fit only on training data and then applied to the test set, preventing test-set information leakage.

After separating the target, `puntaje` and the date, 27 predictor variables remain (24 numeric + 3 categorical). With One-Hot encoding of the categories present, the transformed representation reaches 35 features.

**Outputs (`resultados_feature_engineering/`):** `X_train_transformado.csv`, `X_test_transformado.csv`, `y_train.csv`, `y_test.csv`, `pipeline_preprocesamiento.joblib`, `nombres_caracteristicas.csv`, `resumen_valores_nulos.csv`, `resumen_variables.csv`, `resumen_feature_engineering.txt`.

## 7. Model Training and Evaluation: `model_training_evaluation.py`

It receives the already transformed sets and trains three algorithms:

| Model | Configuration documented in the code | Role in the project |
|---|---|---|
| Logistic Regression | `class_weight="balanced"`, `max_iter=2000` | Reference model (linear technique) |
| Random Forest | 300 trees, `class_weight="balanced"` | Non-linear, ensemble-based approach |
| Calibrated SVM | Balanced `SVC` + `CalibratedClassifierCV` (sigmoid, cv=3) | Comparison with another way of separating classes |

`class_weight="balanced"` is used in the models that support it to account for the imbalance during training. The base SVM is wrapped in `CalibratedClassifierCV` so that probabilities (`predict_proba`) can be obtained.

**Why several models?** No technique is assumed in advance to be adequate. All three are trained on the same data and evaluated with a common structure, so the final model can be selected according to the project's goal: detecting class 0 records (possible default).

**Metrics computed:** Accuracy, Balanced Accuracy, ROC-AUC, PR-AUC, Precision/Recall/F1 per class (0 and 1) and macro, confusion matrices. Cross-validation uses 5 folds and macro F1-score as the metric, to observe behavior across different splits of the training data.

**Final model selection criterion:** the code sorts models by (1) Recall of class 0, (2) F1-score of class 0 and (3) Balanced Accuracy. The first by that order is saved as `modelo_final.joblib`, both in `resultados_model_training_evaluation/` and in the project root folder.

This criterion reflects a methodological decision: it is especially important not to lose the ability to identify "does not pay on time" records. The additional metrics remain relevant for understanding overall behavior and the balance between classes.

> **Note:** the exact values from `resultados_model_training_evaluation/resultados_modelos.csv` are to be filled in when running the pipeline in the review environment.

## 8. Model Deployment: `model_deploy.py`

It rebuilds the same derived variables needed by the model, separates `Pago_atiempo` when available, removes `puntaje` and the original date, transforms the data with the saved pipeline, and checks that the resulting number of features matches what the model expects.

It then generates the prediction, obtains per-class probabilities when the model allows it, and creates a readable interpretation: class 0 → "Posible incumplimiento" (possible default), class 1 → "Pago a tiempo" (paid on time). If real `Pago_atiempo` values exist in the input data, it also computes Accuracy, Balanced Accuracy, Precision, Recall and F1 per class, and a confusion matrix.

## 9. Prediction API (FastAPI): `API.py`

It turns the model into an HTTP service. On startup, it loads `modelo_final.joblib` and `resultados_feature_engineering/pipeline_preprocesamiento.joblib`. Before predicting, it rebuilds the same derived variables used in feature engineering (temporal ones, installment-to-salary ratio, debt-to-salary ratio, estimated outstanding balance), applies the same pipeline and runs the model, reproducing exactly the transformations used in training.

### Endpoints

| Endpoint | Method | Function |
|---|---|---|
| `/` | GET | Confirms the API is running and lists the available endpoints |
| `/health` | GET | Checks that the API, model and pipeline are loaded |
| `/predict` | POST | Processes a single record |
| `/predict/batch` | POST | Processes a list of records |

### API Response

Each prediction returns: a UUID identifier (identifies the prediction; it is not an encryption mechanism), the predicted class, the readable result, the default probability, the on-time payment probability, the risk level and the prediction time in milliseconds.

### Risk Level Thresholds (based on default probability)

| Default probability | Risk level |
|---|---|
| < 0.40 | Low |
| 0.40 – < 0.70 | Medium |
| ≥ 0.70 | High |

Each prediction is appended to `predicciones_api_monitoring.csv`; this file connects the API directly with the monitoring stage.

### Sample Request (`POST /predict/batch`)

Use it to test the service manually from `http://127.0.0.1:8000/docs` (step 7 of the quick execution):

```json
{
  "registros": [
    {
      "tipo_credito": "Libre inversión",
      "fecha_prestamo": "2024-03-15",
      "capital_prestado": 5000000,
      "plazo_meses": 24,
      "edad_cliente": 35,
      "tipo_laboral": "Empleado",
      "salario_cliente": 2500000,
      "total_otros_prestamos": 800000,
      "cuota_pactada": 250000,
      "puntaje_datacredito": 720,
      "cant_creditosvigentes": 2,
      "huella_consulta": 1,
      "saldo_mora": 0,
      "saldo_total": 800000,
      "saldo_principal": 700000,
      "saldo_mora_codeudor": 0,
      "creditos_sectorFinanciero": 1,
      "creditos_sectorCooperativo": 0,
      "creditos_sectorReal": 1,
      "promedio_ingresos_datacredito": 2400000,
      "tendencia_ingresos": "Estable"
    }
  ]
}
```

> **Figure (see PDF):** screenshot of the interactive API documentation (Swagger UI) at `http://127.0.0.1:8000/docs`.

## 10. Sending Records to the API: `enviar_predicciones_API.py`

It simulates the arrival of new records at the service. It reads `Base_de_datos.csv`, checks that the required columns exist, selects up to 100 records with `random_state=42`, converts types and fills in missing values before building the request (numeric → sample median; categorical → "Desconocido"; date → defined format; integer columns → rounded and converted).

It sends the records to the `/predict/batch` endpoint, so the API must be running before executing this script. The response is saved as `predicciones_api_monitoring.csv`.

```
Terminal 1: uvicorn API:app --host 127.0.0.1 --port 8000
Terminal 2: python enviar_predicciones_API.py
```

The two-terminal separation is intentional: one keeps the service available and the other acts as the client sending the records.

## 11. Model Monitoring and Data Drift: `model_monitoring.py`

It compares a reference base (`Base_de_datos_preparada.csv`) against the recent predictions generated by the API (`predicciones_api_monitoring.csv`).

**Prior validation:** only predictions where the predicted class is 0 or 1, probabilities are between 0 and 1, and the sum of default probability + on-time payment probability is approximately between 0.99 and 1.01 are considered valid for monitoring.

### What Is Monitored

- Distribution of predicted classes
- Default probability (mean, median, minimum, maximum)
- Risk levels and percentage of high risk
- Prediction time
- Changes in variable distributions relative to the reference
- Temporal behavior of predictions

### Data Drift

| Variable type | Metrics used |
|---|---|
| Numeric | Kolmogorov–Smirnov (KS test), PSI, Jensen–Shannon |
| Categorical | PSI, Jensen–Shannon, Chi-square |

Detection rules (among other criteria): p-value < 0.05, PSI ≥ 0.25 and Jensen–Shannon ≥ 0.10, depending on the variable type and metric.

### Results Observed in the Current Sample (101 records processed by the API)

| Indicator | Value |
|---|---|
| Valid predictions analyzed | 101 |
| Possible default | 53 (52.48%) |
| Paid on time | 48 (47.52%) |
| Default probability, mean | 51.92% |
| Default probability, median | 51.12% |
| Default probability, minimum | 20.13% |
| Default probability, maximum | 100% |
| Low risk | 24 |
| Medium risk | 65 |
| High risk | 12 |
| Mean prediction time | 27.21 ms |
| Variables evaluated for drift | 20 |
| Variables with drift detected | 15 (75%) |

**75% means "15 of 20 evaluated variables"**: it does not mean that 75% of customers changed behavior, nor that 75% have high risk. This result corresponds to this specific sample and run; it should not be interpreted as a permanent condition of the model or of future data.

Variables showing drift signals in this run include `tendencia_ingresos`, `promedio_ingresos_datacredito`, `cuota_pactada` and `tipo_credito` (full detail in `data_drift_resultados.csv`).

These percentages correspond to the model's predictions on the sample processed by the API. They should not be confused with the historical distribution of the original dataset (95.25% on-time payments / 4.75% not paid on time), nor do they imply that 52.48% of those customers will actually default.

> **Figure (see PDF):** `analisis_temporal_predicciones.png`, temporal evolution of predictions (example run, single sample).

**Outputs (`resultados_model_monitoring/`):** `metricas_monitoring.csv`, `distribucion_predicciones.csv`, `predicciones_validas_monitoring.csv`, `resumen_niveles_riesgo.csv`, `alertas_monitoring.txt`, `data_drift_resultados.csv`, `data_drift_alertas.txt`, `variables_con_drift.csv`, `data_drift_graficos/`, `analisis_temporal.csv`, `analisis_temporal_predicciones.png`.

## 12. Streamlit Application: `APP.py`

It builds an interface to consult the monitoring results. It does not retrain the model: it reads the files generated by `model_monitoring.py` and presents them visually.

### Sections

| Section | Content |
|---|---|
| Summary | Total predictions, % default, mean probability, number of high-risk cases, mean time, and three charts: prediction distribution, default probability distribution and risk level distribution |
| Predictions | Filter by risk level and by default probability range |
| Data Drift | Monitored variables, variables with drift, full results table and charts from `data_drift_graficos/` |
| Temporal analysis | Temporal table and `analisis_temporal_predicciones.png` chart |

Streamlit works as the monitoring visualization layer: it does not replace the previous scripts, it consumes their results.

> **Figure (see PDF):** monitoring dashboard, `APP.py` (Streamlit), Summary section.

## 13. Methodological Considerations

- **Class imbalance:** it is a central characteristic of the problem. That is why `class_weight="balanced"` is used in Logistic Regression, Random Forest and the base SVM, and why classes are evaluated separately.
- **Exclusion of `puntaje`:** done because of possible information leakage. It does not mean the score is irrelevant; it was considered methodologically safer not to use it as a predictor in this implementation.
- **Probabilities:** those produced by the model are classifier outputs for the evaluated sample. The risk level is a categorization defined by the service's thresholds and does not, by itself, constitute a causal conclusion about a person's future behavior.
- **Data drift:** it does not by itself prove that the model has gotten worse: it indicates that the compared distributions show differences under the metrics and thresholds used. Evaluating the real impact on performance requires, later on, real observed outcomes for the current predictions.

## 14. Containerization with Docker

The project uses Docker to package and run the prediction service (the FastAPI API, its dependencies and the artifacts needed for inference) in a reproducible way.

### Dockerfile

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY API.py .
COPY modelo_final.joblib .
COPY resultados_feature_engineering/pipeline_preprocesamiento.joblib .
EXPOSE 8000
CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "8000"]
```

It uses `python:3.13-slim` as the base image, sets `/app` as the working directory, copies `requirements.txt` and installs dependencies, and includes only `API.py`, `modelo_final.joblib` and `pipeline_preprocesamiento.joblib`, the minimum needed to serve the model. It exposes port 8000 and starts Uvicorn automatically when the container starts.

### Building the Image

From the project root folder (where the Dockerfile, `requirements.txt`, the model and the other required files are):

```bash
docker build -t modelo-creditos-api .
```

### Running the Container

```bash
docker run -d -p 8000:8000 --name modelo-creditos-api modelo-creditos-api
```

- `-d` runs the container in the background.
- `-p 8000:8000` maps the host's port 8000 to FastAPI's port 8000 inside the container.
- `modelo-creditos-api` (as `--name`) allows the container to be identified with Docker commands.

### Service Verification

- Interactive documentation: http://127.0.0.1:8000/docs
- Service status: http://127.0.0.1:8000/health, which confirms that the service, the model and the preprocessing pipeline loaded correctly.

## 15. Version Control with Git and GitHub

The project uses Git to control versions and keep the history of changes. The repository is initialized from the root folder, keeping the file structure needed for the scripts to run correctly.

**Branch workflow:** `main` is the main branch and represents the integrated version of the project. Changes are developed in independent working branches and merged into `main` once completed and reviewed, avoiding direct changes to the main version during each development stage.

### Delivery History (commits/branches)

| Delivery | Content included |
|---|---|
| 1 | `Cargar_datos.py`, `Comprension_eda.py`, `ft_engineering.py` and their results (`Base_de_datos_preparada.csv`, `Observaciones_calidad_datos.csv`, `resultados_eda_entrega/`, `resultados_feature_engineering/`) |
| 2 | `model_training_evaluation.py`, `model_deploy.py` and their results (`resultados_model_training_evaluation/`, `resultados_model_deploy/`, `modelo_final.joblib`) |
| 3 | `API.py`, `APP.py`, `Dockerfile`, `enviar_predicciones_API.py`, `model_monitoring.py` and all their corresponding results (`predicciones_api_monitoring.csv`, `resultados_model_monitoring/`) |

Each commit identifies the changes made in its stage, making it easier to follow the evolution of the code and recover earlier versions when needed.

### File Structure and Control: `.gitignore`

```
__pycache__/
*.pyc
.venv/
venv/
.env
*.log
.streamlit/secrets.toml
```

The results and artifacts that are part of the delivery are kept in the repository: the `resultados_*` folders, `modelo_final.joblib` and `predicciones_api_monitoring.csv`.

### Remote Repository

The project is synchronized with a remote GitHub repository, which keeps the code and versioned files in a central repository, facilitates delivery and maintains a history of changes. The internal structure must also be preserved in the remote repository: names or locations of files used by the scripts must not be changed without updating the corresponding paths in the code.

## 16. Technology Stack

- **Language:** Python 3.13
- **Data analysis and manipulation:** pandas, numpy, tabulate
- **Visualization:** matplotlib, seaborn
- **Modeling:** scikit-learn, xgboost, joblib
- **Model serving:** FastAPI, Pydantic, Uvicorn
- **Monitoring:** SciPy (statistical tests), Streamlit
- **Containerization:** Docker
- **Version control:** Git / GitHub

## 17. Conclusion

The project builds a complete, traceable Machine Learning flow for credit default prediction. It starts with a quality review that turns the original data into a prepared dataset; continues with an EDA that documents the structure, distribution and relationships of the data; and converts that information into usable features through a reproducible pipeline.

The modeling stage compares three approaches (Logistic Regression, Random Forest and calibrated SVM) using multiple metrics and cross-validation. Final model selection prioritizes detection of the "does not pay on time" class, and the selected model is persisted so it can be reused without repeating training.

From that artifact, the project moves from a Machine Learning experiment to an inference process: deployment applies the pipeline and the model to new records, FastAPI exposes the prediction through endpoints (containerized with Docker), the submission script simulates the arrival of new data, and monitoring analyzes the predictions and possible changes in distributions. Finally, Streamlit brings the monitoring results together in a visual interface.

The solution is not limited to training a model: it integrates preparation, analysis, modeling, reuse, serving, monitoring and visualization, keeping an explicit relationship between the files that produce and consume each stage.
