# Multirotor Flight Log System Identification & Control Design

This repository contains a Python-based system identification and model-based controller design pipeline for multirotors (e.g., Betaflight quadcopters). It extracts the open-loop plant dynamics from flight log CSVs and optimizes discrete-time PID controllers to match target performance metrics.

---

## Workspace Structure

- `id_pipeline.py`: Runs ARX (Autoregressive with Exogenous Input) system identification on flight logs.
- `analyze_models.py`: Analyzes the identified discrete-time models (poles, zeros, stability, DC gain, and Bode plots).
- `design_controller.py`: Designs and simulates discrete-time PID controllers by optimizing tracking error (ITAE) and control effort.
- `run_pipeline.py`: Master automation script that executes identification, analysis, and control design sequentially in one command.
- `raw_logs/`: Directory containing raw `.BBL` binary log files.
- `out/`: Contains the generated outputs:
  - `arx_model_axis{0,1,2}.pkl`: Saved model coefficients.
  - `fit_axis{0,1,2}.png`: Time-domain predictions vs. true gyro rate.
  - `pzmap_axis{0,1,2}.png` & `bode_axis{0,1,2}.png`: Frequency-domain and pole-zero analysis.
  - `control_design_axis{0,1,2}.png`: Closed-loop step response simulations.
  - `model_analysis_report.md` & `control_design_report.md`: Structured markdown reports.

---

## Getting Started

### 1. Setup Environment
Initialize a virtual environment and install the required dependencies:

```powershell
python -m venv .venv
.venv\Scripts\pip.exe install -r requirements.txt
```

### 2. Run the Entire Pipeline (Recommended)
You can run the entire identification and design process for all three axes with a single command:

```powershell
.venv\Scripts\python.exe run_pipeline.py
```

### 3. Running Steps Individually (Optional)

If you prefer to run steps manually or tweak model order parameters:

#### A. Run System Identification
Run the identification pipeline for each axis (0 = Roll, 1 = Pitch, 2 = Yaw):

```powershell
# Roll (Axis 0)
.venv\Scripts\python.exe id_pipeline.py --axis 0 --na 1 --nb 1 --nk 2

# Pitch (Axis 1)
.venv\Scripts\python.exe id_pipeline.py --axis 1 --na 1 --nb 1 --nk 2

# Yaw (Axis 2)
.venv\Scripts\python.exe id_pipeline.py --axis 2 --na 1 --nb 1 --nk 2
```

#### B. Analyze Plant Models
Run the analysis script to inspect poles/zeros, DC gains, stability, and generate Bode frequency response plots for the identified systems:

```powershell
.venv\Scripts\python.exe analyze_models.py
```

#### C. Design and Optimize Controller
Run the design script to optimize discrete-time PID controllers. It uses a constrained optimization routine (`scipy.optimize.minimize`) to search for the best gains under the ITAE (Integral of Time-weighted Absolute Error) metric and simulates the closed-loop step response:

```powershell
.venv\Scripts\python.exe design_controller.py
```

---

## Model Bounds & Constraints

To ensure that the identified models and designed controllers represent physically realistic systems, the pipeline implements several key constraints:
- **Stable Poles**: Autoregressive poles $a_i$ are constrained to $[0.0, 0.995]$ in `id_pipeline.py`, preventing unstable models from being identified from noisy, closed-loop flight data.
- **Physical Feedback Direction**: Input gains $b_j$ are constrained to be strictly negative ($\le -10^{-5}$), guaranteeing that positive PID control action results in negative gyro rates (matching standard physics directions).
- **ITAE Optimization**: PID tuning minimizes tracking errors while penalizing high-frequency control derivative spikes to avoid motor heating and actuator wear.

---

## Exporting New Logs

If you have new `.BBL` binary log files in the future, you can export them to CSVs using the globally installed `bbl_parser` CLI tool:

```powershell
bbl_parser --csv --output-dir out/ raw_logs/
```

