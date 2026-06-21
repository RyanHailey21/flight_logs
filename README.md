# Multirotor Barometer Altitude System Identification & Control Design

This repository contains a Python-based system identification and model-based controller design pipeline for multirotor altitude hold systems (e.g., Betaflight quadcopters using barometer sensors). 

---

## Rationale: 10 Hz Downsampling & 2nd-Order Dynamics

1. **Downsampling to 10 Hz**: The flight controller loop runs at a very high rate ($\approx 1000 \text{ Hz}$), but the barometer sensor (`baroAlt`) only updates at approximately $10 \text{ Hz}$ (about once every $100 \text{ ms}$). Running system identification directly on the raw data would attempt to fit the staircase sensor output, resulting in non-physical high-frequency poles. The pipeline automatically groups telemetry data into $100 \text{ ms}$ blocks, averaging throttle input (acting as an anti-aliasing low-pass filter) and extracting the latest altitude reading at the end of each block.
2. **2nd-Order Physics**: Vertical quadcopter dynamics from throttle thrust command $u(t)$ to altitude $y(t)$ are modeled as a 2nd-order autoregressive (ARX) system representing acceleration, velocity integration, and altitude integration:
   $$G(z) = \frac{b_1 z^{-1} + b_2 z^{-2}}{1 - a_1 z^{-1} - a_2 z^{-2}} z^{-nk}$$
   The solver identifies a stable integration pole ($z \approx 0.999$) and calculates the steady-state hover throttle bias dynamically.
3. **Takeoff Offset Normalization**: Barometer readings depend on ambient pressure and start at different initial offsets for each flight. The pipeline normalizes altitude data by subtracting the initial takeoff altitude of each flight session. This ensures all logs start at exactly $0.0\text{ m}$, enabling multi-log parameter estimation.
4. **Heuristic Active Flight Filtering**: To prevent bench tests (USB configuration or motor tests on a stand) or short arming/disarming clips from polluting the fitting, logs are filtered by default. A log is classified as an active flight only if it has a duration of $\ge 15.0\text{ seconds}$, a maximum throttle of $\ge 1300$, and active gyro roll/pitch variance $> 5.0$. Gyro data is strictly used for this binary classification filter and is *not* modeled by the altitude fitting process.

---

## Workspace Structure

- `id_pipeline.py`: Runs the ARX system identification on 10 Hz downsampled and takeoff-normalized telemetry data, fitting the model from throttle stick input (`rcCommand[3]`) to barometer altitude (`baroAlt`).
- `analyze_models.py`: Analyzes the identified discrete-time altitude model (poles, stability, DC gain, and Bode plots).
- `design_controller.py`: Simulates the closed-loop step response using the exact cascaded loop structure running in the flight firmware (Proportional outer loop on altitude error + Proportional-Integral inner loop on vertical velocity error) and optimizes the C++ parameters (`HOLD_KP`, `HOLD_KD`, and `HOLD_KI`).
- `run_pipeline.py`: Master automation script that executes identification, analysis, control design, and log manifest compilation in a single command.
- `raw_logs/`: Directory containing raw `.BBL` binary log files.
- `out/`: Contains the structured generated outputs:
  - `csv/`: Decoded CSV telemetry logs (e.g., `btfl_005.csv`) and headers.
  - `models/`: Saved altitude model coefficients (`arx_model_altitude.pkl`).
  - `plots/`: Plant fit plot (`fit_altitude.png`), Bode plot (`bode_altitude.png`), pole-zero map (`pzmap_altitude.png`), and step response simulation plot (`control_design_altitude.png`).
  - `reports/`: Markdown analysis and tuning reports.
  - `log_manifest.csv` & `log_manifest.json`: Auto-generated indexes of all CSV logs detailing sample counts, duration, maximum throttle, and gyro activity, and classifying them as flight logs or bench tests.

---

## Getting Started

### 1. Setup Environment
Initialize a virtual environment and install the required dependencies:

```powershell
python -m venv .venv
.venv\Scripts\pip.exe install -r requirements.txt
```

### 2. Run the Entire Pipeline (Recommended)
You can run the entire system identification and PID tuning pipeline on a specific flight log in one command:

```powershell
.venv\Scripts\python.exe run_pipeline.py --file btfl_005.csv
```

### 3. Running Steps Individually (Optional)
If you prefer to run steps manually:

#### A. Run Altitude System Identification
```powershell
.venv\Scripts\python.exe id_pipeline.py --file btfl_005.csv --na 2 --nb 2 --nk 1
```

#### B. Analyze Plant Stability
```powershell
.venv\Scripts\python.exe analyze_models.py
```

#### C. Tune and Optimize PID Controller
```powershell
.venv\Scripts\python.exe design_controller.py
```

---

## Exporting New Logs

To parse new `.BBL` binary log files in the future, you can export them to CSVs using the globally installed `bbl_parser` CLI tool:

```powershell
bbl_parser --output-dir out/csv/ raw_logs/
```
Once exported, you can execute the master pipeline on the new CSV file.
