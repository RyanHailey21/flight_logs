#!/usr/bin/env python3
"""System identification pipeline for multirotor barometer altitude control.

Identifies a 2nd-order discrete-time transfer function from throttle stick command
(rcCommand[3]) to barometer altitude (baroAlt) using 10 Hz downsampled log data.
"""
import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


def load_and_downsample_with_cache(p: Path, block_size: int):
    """Loads and downsamples data, utilizing a fast cache if available."""
    cache_dir = p.parent.parent / "cache"
    if not cache_dir.exists() and p.parent.name == "csv":
        cache_dir.mkdir(parents=True, exist_ok=True)
    elif not cache_dir.exists():
        cache_dir = p.parent / "out" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
    cache_file = cache_dir / f"{p.stem}.npz"
    csv_mtime = p.stat().st_mtime
    
    if cache_file.exists() and cache_file.stat().st_mtime > csv_mtime:
        try:
            data = np.load(cache_file, allow_pickle=True)
            is_active = bool(data["is_active"])
            duration = float(data["duration"])
            max_th = float(data["max_throttle"])
            gyro_std = [float(v) for v in data["gyro_std"]]
            
            if not is_active:
                return None, {
                    "is_active": False,
                    "duration": duration,
                    "max_throttle": max_th,
                    "gyro_std": gyro_std
                }
            
            return (data["u"], data["y"], data["t"]), {
                "is_active": True,
                "duration": duration,
                "max_throttle": max_th,
                "gyro_std": gyro_std
            }
        except Exception as e:
            print(f"  [Cache Error] Failed to read {cache_file.name}: {e}. Re-analyzing raw log.")
            
    # If not cached or cache invalid, run analysis
    try:
        df = pd.read_csv(p, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        
        # Calculate duration
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
        u_col = "rcCommand[3]"
        if u_col in df.columns:
            max_th = pd.to_numeric(df[u_col], errors='coerce').max()
        elif "rcCommand" in df.columns:
            max_th = pd.to_numeric(df["rcCommand"], errors='coerce').max()
            
        gyro_std = [0.0, 0.0, 0.0]
        for axis in [0, 1, 2]:
            col = f"gyroADC[{axis}]"
            if col in df.columns:
                gyro_std[axis] = float(pd.to_numeric(df[col], errors='coerce').std())
                
        is_active = (max_th >= 1300.0) and (duration >= 15.0) and (max(gyro_std[0], gyro_std[1]) > 5.0)
        
        if not is_active:
            # Cache inactive status
            np.savez(
                cache_file,
                is_active=False,
                duration=duration,
                max_throttle=max_th,
                gyro_std=gyro_std,
                u=np.array([]),
                y=np.array([]),
                t=np.array([])
            )
            return None, {
                "is_active": False,
                "duration": duration,
                "max_throttle": max_th,
                "gyro_std": gyro_std
            }
            
        downsampled = downsample_data(df, block_size)
        if downsampled is None:
            np.savez(
                cache_file,
                is_active=False,
                duration=duration,
                max_throttle=max_th,
                gyro_std=gyro_std,
                u=np.array([]),
                y=np.array([]),
                t=np.array([])
            )
            return None, {
                "is_active": False,
                "duration": duration,
                "max_throttle": max_th,
                "gyro_std": gyro_std
            }
            
        u_down, y_down, t_down = downsampled
        
        # Cache active data
        np.savez(
            cache_file,
            is_active=True,
            duration=duration,
            max_throttle=max_th,
            gyro_std=gyro_std,
            u=u_down,
            y=y_down,
            t=t_down
        )
        return (u_down, y_down, t_down), {
            "is_active": True,
            "duration": duration,
            "max_throttle": max_th,
            "gyro_std": gyro_std
        }
    except Exception as e:
        print(f"Error parsing log {p.name}: {e}")
        return None, None


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
                time_diffs = time_diffs[time_diffs > 0]
                if len(time_diffs) > 0:
                    Ts = np.mean(time_diffs) * 1e-6
                    if verbose:
                        print(f"Estimated Ts from {p.name}: {Ts:.6f} s ({1/Ts:.2f} Hz)")
                    return Ts
        except Exception as e:
            if verbose:
                print(f"Could not read time from {p.name}: {e}")
            
    default_Ts = 0.001  # 1 kHz default
    if verbose:
        print(f"Using default Ts: {default_Ts:.6f} s ({1/default_Ts:.2f} Hz)")
    return default_Ts


def downsample_data(df, block_size):
    """Downsamples high-rate telemetry data to 10 Hz to match barometer update rate, focusing on the in-air segment."""
    y_col = "baroAlt"
    u_col = "rcCommand[3]"
    
    if y_col not in df.columns or u_col not in df.columns:
        return None
        
    # 1. Filter to only include in-air flight segment (altitude has risen at least 50 cm above takeoff baseline)
    y_raw_full = pd.to_numeric(df[y_col], errors='coerce').ffill().bfill().values
    if len(y_raw_full) == 0:
        return None
    takeoff_base = y_raw_full[0]
    takeoff_idx = np.where(y_raw_full > takeoff_base + 50.0)[0]
    if len(takeoff_idx) == 0:
        return None
    start_idx = takeoff_idx[0]
    
    df_flight = df.iloc[start_idx:].copy()
    
    # 2. Extract values for the flight segment
    y_raw = pd.to_numeric(df_flight[y_col], errors='coerce').ffill().bfill().values
    u_raw = pd.to_numeric(df_flight[u_col], errors='coerce').ffill().bfill().values
    t_raw = pd.to_numeric(df_flight["time (us)"], errors='coerce').values * 1e-6
    
    # Normalize altitude: subtract initial takeoff altitude offset and convert to meters
    if len(y_raw) > 0:
        y_raw = (y_raw - y_raw[0]) / 100.0
        
    N = len(df_flight)
    N_down = N // block_size
    
    y_down = []
    u_down = []
    t_down = []
    
    for i in range(N_down):
        start = i * block_size
        end = start + block_size
        
        # Take latest altitude at the end of the block
        y_down.append(y_raw[end - 1])
        # Take average throttle command over the block (anti-aliasing)
        u_down.append(np.mean(u_raw[start:end]))
        # Take latest timestamp
        t_down.append(t_raw[end - 1])
        
    return np.array(u_down), np.array(y_down), np.array(t_down)


def build_altitude_arx_matrices_from_cache(paths, block_size, na=2, nb=2, nk=1, verbose=True):
    """Loads cached/downsampled data and constructs regressor matrix for altitude system ID."""
    u_all = []
    y_all = []
    
    for p in paths:
        res, meta = load_and_downsample_with_cache(p, block_size)
        if res is None:
            if meta and verbose:
                print(f"Skipping {p.name} (Non-Flight/Short Test: duration={meta['duration']:.1f}s, max throttle={meta['max_throttle']:.0f}, gyro std=[{meta['gyro_std'][0]:.1f}, {meta['gyro_std'][1]:.1f}])")
            continue
            
        u, y, _ = res
        u_all.append(u)
        y_all.append(y)
        if verbose:
            print(f"Loaded {p.name} (Cached: {len(u)} samples)")
            
    if not u_all:
        return np.array([]), np.array([])
        
    Xs = []
    ys = []
    
    for u, y in zip(u_all, y_all):
        N = len(y)
        for t in range(max(na, nb + nk), N):
            row = []
            # Past outputs: y(t-1) ... y(t-na)
            for i in range(1, na + 1):
                row.append(y[t - i])
            # Past inputs: u(t-nk-1) ... u(t-nk-nb)
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
    """Scan the csv subfolder and compile metadata about all flight logs utilizing cache for speed."""
    csv_dir = folder / "out" / "csv"
    if not csv_dir.exists():
        csv_dir = folder / "csv"
    if not csv_dir.exists():
        print("No CSV directory found to compile manifest from.")
        return
        
    csvs = sorted(csv_dir.glob("*.csv"))
    csvs = [p for p in csvs if "headers.csv" not in p.name and p.name != "combined.csv"]
    
    # We need block_size to call cache function
    Ts_raw = get_sampling_time(folder, verbose=False)
    block_size = int(round(0.1 / Ts_raw))
    
    records = []
    print(f"Compiling manifest for {len(csvs)} logs...")
    
    for p in csvs:
        try:
            _, meta = load_and_downsample_with_cache(p, block_size)
            if meta is None:
                continue
                
            flight_type = "Active Flight" if meta["is_active"] else "Bench Test"
            
            records.append({
                "filename": p.name,
                "duration_sec": round(meta["duration"], 2),
                "max_throttle": round(meta["max_throttle"], 1),
                "gyro_std_roll": round(meta["gyro_std"][0], 2),
                "gyro_std_pitch": round(meta["gyro_std"][1], 2),
                "gyro_std_yaw": round(meta["gyro_std"][2], 2),
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
    ap.add_argument("--na", type=int, default=2, help="Output lag order")
    ap.add_argument("--nb", type=int, default=2, help="Input lag order")
    ap.add_argument("--nk", type=int, default=1, help="Input pure delay (latency frames)")
    ap.add_argument("--compile-manifest", action="store_true", help="Compile manifest of all CSV logs")
    args = ap.parse_args()

    folder = Path(args.folder)
    out = Path(folder) / "out"
    out.mkdir(exist_ok=True)

    if args.compile_manifest:
        compile_manifest(folder)
        return

    Ts_raw = get_sampling_time(folder, verbose=False)
    block_size = int(round(0.1 / Ts_raw))
    print(f"Loop sampling rate: {1/Ts_raw:.1f} Hz. Downsampling block size: {block_size} (Target: 10 Hz)")

    # Get paths to analyze
    csv_dir = folder / "out" / "csv"
    if not csv_dir.exists():
        csv_dir = folder / "csv"
    if not csv_dir.exists():
        csv_dir = folder

    if args.file:
        p = Path(args.file)
        if not p.exists():
            p = csv_dir / args.file
        if not p.exists():
            p = folder / args.file
        if not p.exists():
            p = folder / "out" / args.file
        paths = [p] if p.exists() else []
    else:
        csvs = sorted(csv_dir.glob("*.csv"))
        paths = [p for p in csvs if "headers.csv" not in p.name and p.name != "combined.csv"]

    print(f"Loading and building regressors for Altitude (na={args.na}, nb={args.nb}, nk={args.nk})...")
    X, y = build_altitude_arx_matrices_from_cache(paths, block_size, na=args.na, nb=args.nb, nk=args.nk)
    
    if len(y) == 0:
        print("No valid data collected. Check column mappings.")
        return
        
    print(f"Fitting model on {len(y)} samples...")
    # Add intercept column
    X_fit = np.column_stack([X, np.ones(len(X))])
    
    # Box bounds to enforce stability and physical parameter behavior:
    # a_1 in [0.0, 1.99]
    # a_2 in [-0.99, 0.0]
    # b_1, b_2 in [0.0, 10.0] (must be positive because increasing throttle increases altitude)
    # intercept in [-inf, inf]
    lb = [0.0, -0.99] + [0.0] * args.nb + [-np.inf]
    ub = [1.99, 0.0] + [10.0] * args.nb + [np.inf]
    
    res = lsq_linear(X_fit, y, bounds=(lb, ub))
    
    # Extract coefficients
    coef = res.x[:-1]
    intercept = res.x[-1]
    
    model = ConstrainedModel(coef, intercept)
    
    print("Model coefficients (a_1..a_na, b_1..b_nb):", model.coef_)
    
    # Save model
    models_dir = out / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "arx_model_altitude.pkl"
    
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "na": args.na,
            "nb": args.nb,
            "nk": args.nk,
            "block_size": block_size,
            "Ts_raw": Ts_raw,
            "Ts_down": Ts_raw * block_size
        }, f)
    print(f"Saved model to {model_path}")

    # Plot on downsampled data
    plot_len = min(len(y), 200)
    plt.figure(figsize=(10, 4))
    
    start = 0
    end = plot_len
    
    y_slice = y[start:end]
    yhat_slice = model.predict(X[start:end])
    
    plt.plot(y_slice, label="True Altitude", color="black", linewidth=1.5)
    plt.plot(yhat_slice, label="Predicted Altitude (1-Step Ahead)", color="red", linestyle="--", alpha=0.9)
    plt.legend()
    plt.title("Altitude ARX Fit (Downsampled to 10 Hz)")
    plt.xlabel("Samples (at 10 Hz)")
    plt.ylabel("Altitude (m)")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    
    plots_dir = out / "plots"
    plots_dir.mkdir(exist_ok=True)
    figpath = plots_dir / "fit_altitude.png"
    plt.savefig(figpath)
    plt.close()
    print(f"Saved fit plot to {figpath}")


if __name__ == '__main__':
    main()