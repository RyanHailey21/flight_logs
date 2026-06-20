#!/usr/bin/env python3
"""Simple system-identification pipeline for Betaflight Blackbox (.bbl) logs.

Features:
- Parse CSVs from bbl_parser
- Calculate PID Sum (input) and Gyro (output) for a specific axis (0=Roll, 1=Pitch, 2=Yaw)
- Build ARX regressors per-log to prevent boundary crossing
- Include input latency (nk offset)
- Fit an ARX linear model
"""
import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


def load_logs(folder: Path, file_name: str = None):
    dfs = []
    csv_dir = folder / "out" / "csv"
    if not csv_dir.exists():
        csv_dir = folder / "csv"
    if not csv_dir.exists():
        csv_dir = folder

    if file_name:
        p = Path(file_name)
        if not p.exists():
            p = csv_dir / file_name
        if not p.exists():
            p = folder / file_name
        if not p.exists():
            p = folder / "out" / file_name
        paths = [p] if p.exists() else []
    else:
        csvs = sorted(csv_dir.glob("*.csv"))
        paths = [p for p in csvs if "headers.csv" not in p.name and p.name != "combined.csv"]

    for p in paths:
        try:
            df = pd.read_csv(p, low_memory=False)
            df.columns = [c.strip() for c in df.columns] # Clean column names
            df["__source_file"] = p.name
            dfs.append(df)
            print(f"Loaded {p.name} ({len(df)} rows)")
        except Exception as e:
            print(f"Skipping {p.name}: {e}")
            
    if not dfs:
        raise RuntimeError("No readable log CSVs found. Please run a blackbox decoder first.")
    return dfs


def get_sampling_time(folder: Path, verbose: bool = True):
    """Estimate average sampling time Ts in seconds from CSV logs."""
    csv_dir = folder / "out" / "csv"
    if not csv_dir.exists():
        csv_dir = folder / "csv"
    if not csv_dir.exists():
        csv_dir = folder
        
    csvs = sorted(csv_dir.glob("*.csv"))
    csvs = [p for p in csvs if "headers.csv" not in p.name and p.name != "combined.csv"]
    
    for p in csvs:
        try:
            # Read first 1000 rows
            df = pd.read_csv(p, nrows=1000)
            df.columns = [c.strip() for c in df.columns]
            if "time (us)" in df.columns:
                time_col = "time (us)"
            elif "time" in df.columns:
                time_col = "time"
            else:
                time_col = None

            if time_col is not None:
                time_diffs = np.diff(pd.to_numeric(df[time_col], errors='coerce').dropna().values)
                # Filter out negative differences or zero differences
                time_diffs = time_diffs[time_diffs > 0]
                if len(time_diffs) > 0:
                    Ts = np.mean(time_diffs) * 1e-6 # convert us to seconds
                    if verbose:
                        print(f"Estimated Ts from {p.name}: {Ts:.6f} s ({1/Ts:.2f} Hz)")
                    return Ts
        except Exception as e:
            if verbose:
                print(f"Could not read time from {p.name}: {e}")
            
    default_Ts = 0.00025  # 4 kHz default
    if verbose:
        print(f"Using default Ts: {default_Ts:.6f} s ({1/default_Ts:.2f} Hz)")
    return default_Ts

def build_arx_matrices(dfs, axis, na=2, nb=2, nk=0):
    # u = PID sum = P + I + D + F
    # y = Gyro
    
    y_col = f"gyroADC[{axis}]"
    p_col = f"axisP[{axis}]"
    i_col = f"axisI[{axis}]"
    d_col = f"axisD[{axis}]"
    f_col = f"axisF[{axis}]"
    
    Xs = []
    ys = []
    
    for df in dfs:
        # Require essential columns (y, P, I, F are essential; D might be missing on yaw)
        essential_cols = [y_col, p_col, i_col, f_col]
        if not all(c in df.columns for c in essential_cols):
            print(f"Missing essential columns in {df['__source_file'].iloc[0]}, skipping.")
            continue
            
        y = pd.to_numeric(df[y_col], errors='coerce').fillna(0).values
        
        p_val = pd.to_numeric(df[p_col], errors='coerce').fillna(0).values
        i_val = pd.to_numeric(df[i_col], errors='coerce').fillna(0).values
        f_val = pd.to_numeric(df[f_col], errors='coerce').fillna(0).values
        
        if d_col in df.columns:
            d_val = pd.to_numeric(df[d_col], errors='coerce').fillna(0).values
        else:
            d_val = np.zeros_like(y)
            
        u = p_val + i_val + d_val + f_val
        
        N = len(y)
        for t in range(max(na, nb + nk), N):
            row = []
            # past outputs y(t-1) ... y(t-na)
            for i in range(1, na + 1):
                row.append(y[t - i])
            # past inputs u(t-nk-1) ... u(t-nk-nb)
            for j in range(1, nb + 1):
                row.append(u[t - nk - j])
            
            if np.isfinite(row).all() and np.isfinite(y[t]):
                Xs.append(row)
                ys.append(y[t])
                
    return np.array(Xs), np.array(ys)


class ConstrainedModel:
    def __init__(self, coef, intercept):
        self.coef_ = coef
        self.intercept_ = intercept
    def predict(self, X):
        return X @ self.coef_ + self.intercept_



def compile_manifest(folder: Path):
    """Scan the csv subfolder and compile metadata about all flight logs."""
    csv_dir = folder / "out" / "csv"
    if not csv_dir.exists():
        csv_dir = folder / "csv"
    if not csv_dir.exists():
        print("No CSV directory found to compile manifest from.")
        return
        
    csvs = sorted(csv_dir.glob("*.csv"))
    csvs = [p for p in csvs if "headers.csv" not in p.name and p.name != "combined.csv"]
    
    records = []
    print(f"Compiling manifest for {len(csvs)} logs...")
    
    for p in csvs:
        try:
            df = pd.read_csv(p)
            df.columns = [c.strip() for c in df.columns]
            
            rows = len(df)
            duration = 0.0
            if "time (us)" in df.columns:
                t_vals = pd.to_numeric(df["time (us)"], errors='coerce').dropna().values
                if len(t_vals) > 1:
                    duration = (t_vals[-1] - t_vals[0]) * 1e-6
            elif "time" in df.columns:
                t_vals = pd.to_numeric(df["time"], errors='coerce').dropna().values
                if len(t_vals) > 1:
                    duration = t_vals[-1] - t_vals[0]
                    
            max_th = 0.0
            if "rcCommand[3]" in df.columns:
                max_th = float(pd.to_numeric(df["rcCommand[3]"], errors='coerce').max())
            elif "rcCommand" in df.columns:
                max_th = float(pd.to_numeric(df["rcCommand"], errors='coerce').max())
                
            gyro_std = [0.0, 0.0, 0.0]
            for axis in [0, 1, 2]:
                col = f"gyroADC[{axis}]"
                if col in df.columns:
                    gyro_std[axis] = float(pd.to_numeric(df[col], errors='coerce').std())
                    
            is_active = (max_th > 1180) and (max(gyro_std[0], gyro_std[1]) > 5.0)
            flight_type = "Active Flight" if is_active else "Bench Test"
            
            records.append({
                "filename": p.name,
                "samples": rows,
                "duration_sec": round(duration, 2),
                "max_throttle": round(max_th, 1),
                "gyro_std_roll": round(gyro_std[0], 2),
                "gyro_std_pitch": round(gyro_std[1], 2),
                "gyro_std_yaw": round(gyro_std[2], 2),
                "flight_type": flight_type
            })
        except Exception as e:
            print(f"Error scanning {p.name}: {e}")
            
    if not records:
        print("No valid CSV logs parsed for manifest.")
        return
        
    manifest_df = pd.DataFrame(records)
    
    out_dir = folder / "out"
    out_dir.mkdir(exist_ok=True)
    
    csv_path = out_dir / "log_manifest.csv"
    json_path = out_dir / "log_manifest.json"
    
    manifest_df.to_csv(csv_path, index=False)
    manifest_df.to_json(json_path, orient="records", indent=2)
    
    print(f"Saved manifest to {csv_path} and {json_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=".", help="Folder with .csv logs")
    ap.add_argument("--file", default=None, help="Specific CSV file to load")
    ap.add_argument("--axis", type=int, default=0, help="0=Roll, 1=Pitch, 2=Yaw")
    ap.add_argument("--na", type=int, default=4, help="Output lag order")
    ap.add_argument("--nb", type=int, default=4, help="Input lag order")
    ap.add_argument("--nk", type=int, default=2, help="Input pure delay (latency frames)")
    ap.add_argument("--compile-manifest", action="store_true", help="Compile manifest of all CSV logs")
    args = ap.parse_args()

    folder = Path(args.folder)
    out = Path(folder) / "out"
    out.mkdir(exist_ok=True)

    if args.compile_manifest:
        compile_manifest(folder)
        return

    dfs = load_logs(folder, args.file)
    
    print(f"Building regressors for Axis {args.axis} (na={args.na}, nb={args.nb}, nk={args.nk})...")
    X, y = build_arx_matrices(dfs, axis=args.axis, na=args.na, nb=args.nb, nk=args.nk)
    
    if len(y) == 0:
        print("No valid data collected. Check column mappings.")
        return
        
    print(f"Fitting model on {len(y)} samples...")
    # Fit with constraints: gains b_j must be non-positive (negative feedback direction)
    # Add column of ones for intercept
    X_fit = np.column_stack([X, np.ones(len(X))])
    
    # Bounds: a_i are stable [0.0, 0.995], b_j are strictly negative [-inf, -1e-5], intercept is unconstrained
    lb = [0.0] * args.na + [-np.inf] * args.nb + [-np.inf]
    ub = [0.995] * args.na + [-1e-5] * args.nb + [np.inf]
    
    res = lsq_linear(X_fit, y, bounds=(lb, ub))
    
    # Extract coefficients
    coef = res.x[:-1]
    intercept = res.x[-1]
    
    model = ConstrainedModel(coef, intercept)
    
    print("Model coefficients (a_1..a_na, b_1..b_nb):", model.coef_)
    
    # Save model
    models_dir = out / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / f"arx_model_axis{args.axis}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model, 
            "axis": args.axis, 
            "na": args.na, 
            "nb": args.nb, 
            "nk": args.nk
        }, f)
    print(f"Saved model to {model_path}")

    # Plot on an interesting slice
    plot_len = min(len(y), 1000)
    plt.figure(figsize=(10, 4))
    
    # Start part way in assuming we want to skip initial transients
    start = min(1000, len(y) // 2)
    end = start + plot_len
    
    if start >= len(y):
        start, end = 0, plot_len
        
    y_slice = y[start:end]
    yhat_slice = model.predict(X[start:end])
    
    plt.plot(y_slice, label="True Gyro", color="black", linewidth=1.5)
    plt.plot(yhat_slice, label="Predicted Gyro", color="red", linestyle="--", alpha=0.9)
    plt.legend()
    plt.title(f"Plant ARX Fit (Axis {args.axis})")
    plt.xlabel("Samples")
    plt.ylabel("Gyro Rate")
    plt.tight_layout()
    
    plots_dir = out / "plots"
    plots_dir.mkdir(exist_ok=True)
    figpath = plots_dir / f"fit_axis{args.axis}.png"
    plt.savefig(figpath)
    print(f"Saved fit plot to {figpath}")

if __name__ == '__main__':
    main()