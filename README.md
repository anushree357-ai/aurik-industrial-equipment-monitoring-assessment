# Industrial Equipment Monitoring Platform Assessment

## Overview

This repository contains my technical assessment submission for the Data Scientist / AI-ML Founding Employee role at Aurik Technologies.

The project demonstrates an end-to-end data science pipeline for industrial equipment monitoring using heterogeneous data from multiple equipment vendors.

The implemented solution validates and normalizes vendor data, creates machine-level features, and generates explainable Health, Confidence, and Readiness scores for each machine.

A small time-aware Machine Learning Proof of Concept (POC) is also included to evaluate the feasibility of predicting future adverse machine outcomes.

---

## Problem Statement

A manufacturing plant receives equipment data from multiple vendors in different schemas, formats, units, identifiers, and timestamps.

The objective is to transform this heterogeneous raw data into a consistent machine-level representation that helps operations and maintenance teams understand:

- Which machines are healthy
- Which machines require attention
- Why a machine has been flagged
- How reliable the available data is
- Whether sufficient data exists for future predictive ML models

---

## Solution Pipeline

The implemented pipeline follows:

Raw Multi-Vendor Data  
→ Data Ingestion  
→ Data Validation  
→ Data Normalization  
→ Feature Engineering  
→ Machine-Level Aggregation  
→ Health / Confidence / Readiness Scoring  
→ Final Machine Health Output

---

## Data Sources

The assessment dataset contains:

- `asset_master.csv` — machine reference and equipment metadata
- `vendor_a_sensor_events.csv` — sensor and machine event data
- `vendor_b_alert_events.csv` — alerts, temperature, vibration and power readings
- `vendor_c_inspection_maintenance.csv` — inspection and maintenance information
- `maintenance_records.csv` — historical maintenance work orders
- `ground_truth_outcomes.csv` — labeled machine outcome records
- `alert_type_dictionary.csv` — vendor alert type reference

---

## Data Validation

The validation layer checks for:

- Missing values
- Duplicate records
- Invalid machine identifiers
- Schema inconsistencies
- Data quality issues

The assessment data contains examples of missing sensor values, duplicate inspection records, and invalid machine identifiers, which are detected before downstream processing.

---

## Data Normalization

Vendor datasets use different schemas and measurement formats.

The normalization layer standardizes:

- Machine identifiers
- Timestamp formats
- Temperature units
- Vibration measurements
- Confidence representations
- Vendor-specific column names

This creates a consistent representation for machine-level analysis.

---

## Feature Engineering

Vendor-level observations are aggregated into machine-level features such as:

- Average and maximum temperature
- Average and maximum vibration
- Power consumption
- Sensor confidence
- Alert/event counts
- Days since service
- Maintenance frequency
- Historical downtime
- Multi-vendor data coverage

---

## Machine Health Scoring

The primary implemented solution is an explainable rule-based Machine Health Scoring System.

Each machine receives:

### Health Score

Represents the operational condition of the machine based on sensor behavior, equipment-rated limits, service history, and historical downtime.

### Confidence Score

Represents the reliability of the available vendor observations.

### Readiness Score

Represents whether sufficient sensor, maintenance, service, multi-vendor, and confidence information is available for advanced analytics or future ML development.

Machines are classified into:

- Healthy
- Monitor
- Attention Required
- Critical

The final machine-level results are available in:

`outputs/machine_health_scores.csv`

---

## Machine Learning POC

A small Machine Learning Proof of Concept is included as a future predictive-maintenance feasibility experiment.

The POC creates time-aware features using the 7-day historical period before each labeled outcome.

Binary target:

- `0` — No Issue
- `1` — Adverse Outcome

Adverse outcomes include:

- Unplanned downtime
- Minor stoppage
- Quality defect spike

A Logistic Regression baseline was evaluated using a chronological train/test split rather than a random split to reduce temporal leakage.

Two feature experiments were evaluated:

1. Vendor A sensor features
2. Multi-vendor sensor, alert, inspection, and maintenance features

The POC produced limited predictive performance. Investigation showed that vendor telemetry ends earlier than several ground-truth outcomes, which reduces recent historical coverage for later observations.

Therefore, the ML implementation is treated as a feasibility POC rather than a production-ready predictive model.

A production ML system would require additional temporally aligned labeled history, stronger feature validation, cross-validation appropriate for time-dependent data, and further model comparison.

---

## Project Structure

```text
aurik-industrial-equipment-monitoring-assessment/
│
├── src/
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── normalization.py
│   ├── feature_engineering.py
│   ├── scoring.py
│   ├── main.py
│   ├── ml_poc.py
│   ├── ml_feature_builder.py
│   └── ml_multivendor_poc.py
│
├── outputs/
│   └── machine_health_scores.csv
│
├── asset_master.csv
├── vendor_a_sensor_events.csv
├── vendor_b_alert_events.csv
├── vendor_c_inspection_maintenance.csv
├── maintenance_records.csv
├── ground_truth_outcomes.csv
├── alert_type_dictionary.csv
├── data_dictionary.md
├── requirements.txt
└── README.md
```

## How to Run

1. Create a virtual environment

```bash
python -m venv .venv
```

2. Activate the environment

```bash
.venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the main machine health pipeline

```bash
python src/main.py
```

The final machine-level output will be generated at:

```text
outputs/machine_health_scores.csv
```

5. Run the ML baseline POC

```bash
python src/ml_poc.py
```

6. Run the multi-vendor ML POC

```bash
python src/ml_multivendor_poc.py
```

---

## Key Design Decisions

- Rule-based scoring is used as the primary solution because it is explainable and can operate with limited labeled data.
- Raw vendor identifiers are preserved while normalized representations are created.
- Machine-rated limits are used when evaluating sensor behavior instead of applying identical thresholds to every machine.
- ML features use only historical observations before each outcome to reduce data leakage.
- ML results are presented as experimental findings rather than production performance claims.

---

## Future Improvements

With additional production data, the system can be extended with:

- Predictive maintenance
- Failure probability prediction
- Remaining Useful Life (RUL) estimation
- Anomaly detection
- Model monitoring and drift detection
- Backend API implementation
- Automated model retraining

---

## Assessment Deliverables

The repository also contains the assessment presentation in:

- PowerPoint format
- PDF format

The presentation covers the business understanding, data challenges, scoring design, API design, future ML strategy, and high-level system architecture.

---

## Author

**Anushree Kalbandhe**