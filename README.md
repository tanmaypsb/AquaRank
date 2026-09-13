# AquaRank

A machine learning prototype that predicts and ranks the **Top 10 fishing grounds** based on expected fishing productivity using historical fisheries data and environmental conditions.

> **Project Status:** Prototype  
> **Current Scope:** Tropical Tuna + Billfish fisheries represented in the selected IOTC datasets  
> **Primary Output:** Ranked Top 10 fishing grounds with predicted productivity scores

---

## Overview

Finding productive fishing grounds traditionally depends heavily on fishermen's experience, historical knowledge, and changing ocean conditions.

This project explores a data-driven approach by combining:

- Historical fish catch
- Fishing effort
- Sea Surface Temperature (SST)
- Fishing-ground location
- Month/season
- Fishing gear
- Species group

The system converts historical catch and effort into **Catch Per Unit Effort (CPUE)**, uses a **Random Forest Regressor** to predict productivity, and ranks fishing grounds to generate a **Top 10 recommendation**.

The goal is to provide a decision-support system that can answer a question such as:

> **"Based on the available historical and environmental data, which fishing grounds are expected to have higher fishing productivity for this month?"**

---

## Project Pipeline

```text
Historical Catch Data
        +
Historical Effort Data
        ↓
Catch-Effort Matching
        ↓
Catch Aggregation
        ↓
CPUE Calculation
        ↓
Fishing-Ground Coordinates
        ↓
SST Matching
        ↓
Feature Engineering
        ↓
Random Forest Regressor
        ↓
Predicted LOG(CPUE)
        ↓
Predicted CPUE
        ↓
Rank Fishing Grounds
        ↓
TOP 10 FISHING GROUNDS
```

---

## Key Features

- Historical catch and fishing-effort based modelling
- Uses **CPUE** instead of raw catch as the productivity measure
- Combines **Tropical Tuna** and **Billfish** observations
- Uses **Sea Surface Temperature (SST)** as an environmental feature
- Includes spatial features such as latitude and longitude
- Includes month/season information
- Includes fishing gear
- Includes species group
- Includes fishing-ground identity/history
- Uses a **time-based train/test split**
- Generates a ranked **Top 10 fishing-ground prediction**
- Saves the final trained model as a `.joblib` file

---

# Dataset

The prototype uses geo-referenced catch and effort data from the **Indian Ocean Tuna Commission (IOTC)**.

## Tropical Tuna

The Tropical Tuna dataset contains:

- Yellowfin Tuna
- Skipjack Tuna
- Bigeye Tuna

## Billfish

The Billfish dataset contains:

- Swordfish
- Indo-Pacific Sailfish
- Black Marlin
- Blue Marlin
- Striped Marlin

The Indian Billfish data initially contained 701 catch records. After aggregation, catch-effort matching, coordinate filtering, modelling-period filtering, and SST filtering, 208 usable observations remained.

---

# Why CPUE?

Using raw catch alone can be misleading because a larger catch does not necessarily mean that a fishing ground is more productive. The amount of fishing effort also matters.

The project therefore uses:

```text
CPUE = Total Catch / Fishing Effort
```

The final SST-valid observations used HOOKS as the effort unit.

The model uses a logarithmic target:

```text
LOG_CPUE = log(CPUE)
```

The predicted logarithmic value is converted back to CPUE for ranking fishing grounds.

---

# Data Preparation

The original IOTC datasets contain hundreds of thousands of global observations. These datasets include many fleets, countries, locations, species, fisheries, and years.

The project applies several stages of filtering and preparation.

```text
Global IOTC Data
        ↓
India Fleet Only
        ↓
Catch + Effort Matching
        ↓
Catch Aggregation
        ↓
CPUE Calculation
        ↓
Fishing-Ground Coordinate Matching
        ↓
2005–2014 Modelling Period
        ↓
SST Matching
        ↓
Valid CPUE + SST Observations
        ↓
ML Dataset
```

## Tropical Tuna Data Reduction

```text
Global Catch Records
~912,678
        ↓
India
1,291 catch records
        ↓
Catch + Effort Processing
980 observations
        ↓
Fishing-Ground Coordinates
925 observations
        ↓
2005–2014
419 observations
        ↓
Valid SST
290 observations
```

## Billfish Data Reduction

```text
Global Catch Records
~521,177
        ↓
India
701 catch records
        ↓
Catch + Effort Processing
480 observations
        ↓
Fishing-Ground Coordinates
437 observations
        ↓
2005–2014
287 observations
        ↓
Valid SST
208 observations
```

## Final Dataset

```text
Tropical Tuna : 290
Billfish      : 208
------------------
Total         : 498
```

---

# Sea Surface Temperature

Sea Surface Temperature was obtained from the **INCOIS Tropical Merged Imager (TMI)** dataset.

The SST dataset used in the prototype covers:

```text
2005–2014
```

with approximately:

```text
0.25° spatial resolution
```

SST is matched to fishing-ground coordinates and month.

Chlorophyll-a was investigated as a possible environmental feature but was **not included in the final model** because the available datasets did not provide sufficient historical overlap for reliable training within the project timeframe.

---

# Features Used by the Model

The final Random Forest model uses:

```text
SST
CENTER_LAT
CENTER_LON
MONTH_SIN
MONTH_COS
GEAR
SPECIES_GROUP
FISHING_GROUND_CODE
```

## Feature Description

| Feature | Description |
|---|---|
| `SST` | Sea Surface Temperature |
| `CENTER_LAT` | Latitude of the fishing-ground center |
| `CENTER_LON` | Longitude of the fishing-ground center |
| `MONTH_SIN` | Cyclic representation of month |
| `MONTH_COS` | Cyclic representation of month |
| `GEAR` | Fishing gear / fishing method |
| `SPECIES_GROUP` | `TUNA` or `BILLFISH` |
| `FISHING_GROUND_CODE` | IOTC fishing-ground identifier |

---

# Target Variable

The model predicts:

```text
LOG_CPUE
```

which is calculated as:

```text
LOG_CPUE = log(CPUE)
```

where:

```text
CPUE = Total Catch / Fishing Effort
```

After prediction:

```text
Predicted LOG(CPUE)
        ↓
      exp()
        ↓
Predicted CPUE
        ↓
      Ranking
```

---

# Why Random Forest Regressor?

The target variable is a **continuous value**, so this is a regression problem rather than a classification problem.

The model therefore uses:

```text
RandomForestRegressor
```

Random Forest was selected because fishing productivity is unlikely to have a simple linear relationship with environmental and spatial variables.

For example:

```text
SST + Location + Month
```

may interact in non-linear ways.

Random Forest can:

- Model non-linear relationships
- Capture interactions between variables
- Handle mixed numerical and categorical features after preprocessing
- Work well with structured/tabular datasets
- Provide a strong baseline without requiring a specific mathematical relationship between variables
- Be relatively robust to noisy tabular data

Categorical features such as `GEAR`, `SPECIES_GROUP`, and `FISHING_GROUND_CODE` are one-hot encoded through the preprocessing pipeline.

---

# Model Architecture

```text
Numerical Features
    ├── SST
    ├── Latitude
    ├── Longitude
    ├── Month Sin
    └── Month Cos

Categorical Features
    ├── Gear
    ├── Species Group
    └── Fishing Ground

                ↓

        Preprocessing
        One-Hot Encoding

                ↓

      Random Forest Regressor

                ↓

          LOG(CPUE)

                ↓

         CPUE Productivity

                ↓

       Fishing Ground Ranking
```

---

# Model Evaluation

A random train/test split was intentionally avoided because random splitting can allow observations from future years to influence the training process.

Instead, the prototype uses a **time-based split**.

```text
Training Period:
2005–2011

Testing Period:
2012–2014
```

## Ground-Aware Model Results

The evaluated development model achieved:

| Metric | Score |
|---|---:|
| MAE | **1.0316** |
| RMSE | **1.2702** |
| R² | **0.4055** |

### Metric Meaning

**MAE — Mean Absolute Error**

Measures the average absolute prediction error.

Lower is better.

**RMSE — Root Mean Squared Error**

Penalizes larger errors more strongly than MAE.

Lower is better.

**R² — Coefficient of Determination**

Measures how much variation in the target is explained by the model.

Higher is better.

---

# Final Production Model

The evaluation model was trained using:

```text
2005–2011 → Training
2012–2014 → Testing
```

After evaluation, the final production model was retrained using **all 498 usable observations** from 2005–2014.

```text
All 498 observations
        ↓
Final Random Forest
        ↓
fishing_ground_model_final.joblib
```

The final model is intended for inference after training.

---

# Prediction System

The prediction engine accepts a requested year and month.

Example:

```text
Year  = 2014
Month = 7
```

The system then:

1. Identifies the relevant SST information.
2. Creates candidate fishing-ground inputs.
3. Combines environmental, spatial, seasonal, gear, species, and ground information.
4. Runs the trained Random Forest.
5. Predicts `LOG_CPUE`.
6. Converts predictions back to CPUE.
7. Aggregates predictions by fishing ground.
8. Sorts grounds from highest to lowest predicted productivity.
9. Returns the Top 10.

Conceptually:

```text
Year + Month
      ↓
Environmental Conditions
      +
Historical Spatial Information
      ↓
Random Forest
      ↓
Predicted Productivity
      ↓
Rank
      ↓
Top 10 Fishing Grounds
```

The productivity value is a **predicted CPUE/productivity score**.

It is not a prediction of tonnes of fish.

---

# Model Artifact

The final trained model is saved as:

```text
fishing_ground_model_final.joblib
```

The file contains the trained preprocessing + Random Forest pipeline and model metadata.

A separate prediction script/notebook can load the model using:

```python
import joblib

saved = joblib.load("fishing_ground_model_final.joblib")

model = saved["model"]
features = saved["features"]
```

The training notebook should be kept with the model artifact because the model depends on the same preprocessing and feature construction used during training.

---

# Project Structure

Recommended GitHub repository structure:

```text
fishing-ground-predictor/
│
├── README.md
│
├── notebooks/
│   └── fishing_ground_model.ipynb
│
├── models/
│   └── fishing_ground_model_final.joblib
│
├── src/
│   ├── preprocessing.py
│   ├── prediction.py
│   └── train.py
│
├── data/
│   └── README.md
│
└── requirements.txt
```

Large raw datasets should not normally be committed directly to GitHub. Instead, document their official download sources and instructions.

---

# Reproducibility

To reproduce the project:

1. Download the IOTC Tropical Tuna catch and effort data.
2. Download the IOTC Billfish catch and effort data.
3. Download the IOTC fishing-ground spatial grid.
4. Download the INCOIS TMI SST dataset used in this project.
5. Run the preprocessing and feature-engineering pipeline.
6. Calculate CPUE.
7. Match fishing-ground coordinates.
8. Match SST conditions.
9. Train the Random Forest model.
10. Save the trained model as a `.joblib` file.
11. Use the prediction engine to generate ranked fishing grounds.

---

# Limitations

## 1. Species Scope

The current model does **not predict every fish species** found in Indian waters.

It is trained using the selected IOTC:

- Tropical Tuna
- Billfish

datasets.

Therefore, the output should be interpreted as a productivity prediction for the fisheries represented by these datasets.

---

## 2. Historical SST

The current SST dataset covers:

```text
2005–2014
```

Therefore, the prototype is not yet a live 2026 forecasting system.

A current SST data source would be required to generate genuine present-day environmental predictions.

---

## 3. Limited Usable Observations

The source datasets are globally large, but only a small portion represents the Indian fleet.

Additional filtering was required for:

- Indian fleet
- Matching catch and effort
- Valid fishing-ground coordinates
- 2005–2014 modelling period
- Valid SST

This resulted in:

```text
498 usable observations
```

for the final model.

---

## 4. Spatial Generalization

The model performs better for fishing grounds that have historical representation in the training data.

Completely unseen fishing grounds are more difficult because the model has limited direct historical evidence for those locations.

---

## 5. Productivity vs Biomass

The model predicts:

```text
CPUE / Productivity
```

It does not directly predict:

```text
Fish biomass
```

or:

```text
Guaranteed catch
```

---

## 6. No Guarantee of Catch

A Top-10 result means:

> These fishing grounds have the highest predicted productivity under the available historical and environmental information.

It does not mean:

> Fish are guaranteed to be present at that exact location and time.

Actual fishing conditions can depend on many additional variables such as weather, currents, prey availability, fishing pressure, regulations, and local knowledge.

---

# Future Improvements

The model can be improved significantly with additional high-quality and current datasets.

Potential improvements include:

### Environmental Features

- Current SST
- Chlorophyll-a
- Ocean currents
- Wind speed and direction
- Bathymetry / depth
- Ocean productivity indicators

### Fisheries Data

- More recent Indian catch observations
- More detailed monthly fisheries data
- Additional fishing fleets
- More species
- More fishing grounds
- More years of spatially resolved catch and effort data

### Machine Learning Improvements

- Spatial cross-validation
- Time-series validation
- Hyperparameter optimization
- Gradient boosting models
- XGBoost / LightGBM
- Ensemble models
- Uncertainty estimation
- Confidence scores for Top-10 recommendations

### Product Improvements

A future version could provide:

```text
Top 10 Fishing Grounds
        +
Coordinates
        +
Predicted Productivity
        +
Confidence Score
        +
Current Ocean Conditions
        +
Weather/Safety Information
```

---

# Future System Architecture

The long-term system could evolve into:

```text
Historical Catch + Effort
            +
       Current SST
            +
       Chlorophyll
            +
       Ocean Currents
            +
          Wind
            +
       Bathymetry
            +
        Seasonality
            ↓
      Machine Learning
            ↓
   Predicted Productivity
            +
      Confidence Score
            ↓
      Rank Ground Areas
            ↓
       TOP 10 GROUNDS
```

---

# Data Sources

## IOTC Tropical Tuna Catch & Effort

Official IOTC dataset:

https://iotc.org/WPTT/28AS/Data/03-CE

---

## IOTC Billfish Catch & Effort

Official IOTC dataset:

https://iotc.org/WPB/24/Data/03-CE

---

## INCOIS TMI SST

Official INCOIS ERDDAP dataset:

https://erddap.incois.gov.in/erddap/griddap/incois_tmi_3day_datasets.html

---

# Disclaimer

This project is intended for **research, prototyping, educational use, and decision-support experimentation**.

The predictions are statistical estimates generated from historical fisheries observations and available environmental information.

They should not be considered:

- Guaranteed fishing advice
- Guaranteed fish catch
- A direct estimate of fish biomass
- A replacement for local fishing knowledge
- A replacement for weather or maritime safety information
- A replacement for official fishing regulations or advisories

---

# Author

**Tanmay Pratap**
