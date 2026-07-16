========================================================================
                 CRYPTOCURRENCY PRICE PREDICTION PLATFORM
========================================================================

A production-ready, end-to-end Machine Learning web application that 
predicts cryptocurrency price trends. This project consolidates an 
experimental 6-stage (M1-M6) machine learning pipeline into high-
performance, automated Python scripts (train.py and app.py) served 
through a lightweight Flask web interface.


------------------------------------------------------------------------
 KEY FEATURES
------------------------------------------------------------------------
- End-to-End Pipeline: Integrates data ingestion, continuous time-
  series feature engineering, model optimization, and interactive web 
  serving.
- Robust Feature Engineering: Generates crucial market signals like 
  Simple Moving Averages (SMA), Exponential Moving Averages (EMA), and 
  Relative Strength Index (RSI).
- Time-Series Integrity: Built explicitly to handle continuous 24/7 
  financial markets, preventing temporal leakage and lookahead bias.
- Interactive UI: A responsive, clean web dashboard 
  (templates/index.html) to display prediction outputs and model insights.


------------------------------------------------------------------------
 THE M1-M6 MACHINE LEARNING WORKFLOW
------------------------------------------------------------------------
The backend pipeline (consolidated within train.py and serialized to 
crypto_model.pkl) implements the industry-standard ML lifecycle across 
six distinct execution phases:

[ M1: Preprocessing ] --> [ M2: Feature Eng ] --> [ M3: Baseline Model ]
                                                           |
                                                           v
[ M6: Deployment ]    <-- [ M5: Evaluation ]  <-- [ M4: Hyper-Tuning ]


 M1: Data Acquisition & Preprocessing
* Action: Ingests raw historical cryptocurrency OHLCV (Open, High, Low, 
  Close, Volume) market data.
* Specialization: Since crypto markets trade 24/7, missing timestamps 
  or API communication gaps are resolved using forward-filling and linear 
  time-series interpolation rather than standard row deletion, 
  preserving chronological sequence.

 M2: Feature Engineering & Scaling
* Action: Extracts high-signal technical indicators to translate price 
  action into predictive features.
* Scaling: Features are normalized using standard scaling: z = (x - mu) / sigma
  This ensures that massive historical price surges do not over-index 
  the model. A correlation matrix is utilized to prune redundant, highly 
  collinear features.

 M3: Sequential Splitting & Baseline Modeling (Regression & Classification)
* Action: Establishes the initial validation framework and baseline estimators.
* Models Explored:
  - Linear Regression: Implemented as a baseline regression model to test 
    direct continuous next-step price value prediction.
  - Logistic Regression: Trained as a baseline classifier to test binary 
    trend classification (predicting whether the price will go UP or DOWN).
* Integrity: Avoids randomized splits (which cause lookahead bias). Uses 
  a strict, chronological Sequential Time-Series Split (70% Train, 30% Test) 
  to ensure the model only learns from the past to predict the future.

 M4: Hyperparameter Tuning & Optimization (Ensemble Tree-Based Models)
* Action: Optimizes estimator parameters to capture non-linear market regimes.
* Models Explored:
  - Decision Tree: Built as a simple non-linear baseline. While intuitive, 
    individual decision trees are highly prone to overfitting on volatile 
    financial data.
  - Random Forest: Deployed as a robust ensemble method. Combining the 
    predictions of multiple decision trees significantly mitigates 
    variance, generalizes better, and captures highly complex market trends.
* Tuning: Implements a rolling-window or expanding-window grid-search 
  validation pipeline (GridSearchCV / RandomizedSearchCV) to tune decision 
  boundaries (e.g., Tree Depth, Estimators, Learning Rates) without 
  leaking validation targets.

 M5: Advanced Financial & Metric Evaluation
* Action: Assesses model strength using a combination of traditional statistical 
  regression metrics and realistic financial indicators.
* Metrics: Tracks Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), 
  and directional accuracy (correct trend direction %).

 M6: Model Serialization & Production Readiness
* Action: Prepares the optimal trained model for low-latency live inference.
* Output: Re-fits the model parameters on the full dataset to capture the 
  most recent market regime and serializes it to disk as crypto_model.pkl.


------------------------------------------------------------------------
📁 REPOSITORY STRUCTURE
------------------------------------------------------------------------
├── app.py                  # Flask web application & API routing
├── train.py                # Complete M1-M6 machine learning pipeline
├── crypto_model.pkl        # Serialized, high-performance trained model
├── requirements.txt        # Package dependencies for reproduction
└── templates/
    └── index.html          # Web dashboard interface (frontend)


------------------------------------------------------------------------
 INSTALLATION & USAGE
------------------------------------------------------------------------

1. Clone the Repository:
   git clone https://github.com/Raghavi-R12/crypto-price-prediction.git
   cd crypto-price-prediction

2. Set Up a Virtual Environment (Optional but Recommended):
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. Install Dependencies:
   pip install -r requirements.txt

4. (Optional) Re-train the Model:
   If you wish to run the full M1-M6 pipeline locally to generate a fresh 
   crypto_model.pkl:
   python train.py

5. Run the Application:
   Start the local Flask development server:
   python app.py
   
   Open your browser and navigate to http://127.0.0.1:5000 to interact 
   with the platform!


------------------------------------------------------------------------
 TECHNOLOGIES & MODELS USED
------------------------------------------------------------------------
- Languages: Python, HTML5, CSS3
- Web Framework: Flask
- Machine Learning & Analytics: Scikit-Learn, Pandas, NumPy, Joblib
- Key Algorithms Tested:
  * Linear Regression
  * Logistic Regression
  * Decision Tree Regressor/Classifier
  * Random Forest Regressor/Classifier (Final Production Model)
========================================================================
