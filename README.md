# Multirotor Flight Log System Identification & Control Design

This repository contains a Python-based system identification and model-based controller design pipeline for multirotors (e.g., Betaflight quadcopters). It extracts the open-loop plant dynamics from flight log CSVs and optimizes discrete-time PID controllers to match target performance metrics.

---

## Workspace Structure

- `id_pipeline.py`: Runs ARX (Autoregressive with Exogenous Input) system identification on flight logs.
- `analyze_models.py`: Analyzes the identified discrete-time models (poles, zeros, stability, DC gain, and Bode plots).
- `design_controller.py`: Designs and simulates discrete-time PID controllers by optimizing tracking error (ITAE) and control effort.
- `run_pipeline.py`: Master automation script that executes identification, analysis, control design, and manifest compilation in one command.
- `raw_logs/`: Directory containing raw `.BBL` binary log files.
- `out/`: Contains the structured generated outputs:
  - `csv/`: Decoded CSV telemetry logs (e.g., `btfl_all.24.csv`) and headers.
  - `models/`: Saved model coefficient pickles (`arx_model_axis{0,1,2}.pkl`).
  - `plots/`: Plant fit plots, Bode diagrams, pole-zero maps, and step responses.
  - `reports/`: Markdown analysis and controller tuning reports.
  - `log_manifest.csv` & `log_manifest.json`: Auto-generated indexes of all CSV logs with duration, samples, max throttle, gyro variance, and flight/bench classification.

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
bbl_parser --csv --output-dir out/csv/ raw_logs/
```

