#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from id_pipeline import ConstrainedModel, get_sampling_time

def simulate_closed_loop(a, b, nk, Ts, Kp, Ki, Kd, sign_gain, N_steps=150):
    """Simulate closed-loop step response of the identified plant with PID control."""
    r = np.zeros(N_steps)
    r[10:] = 1.0  # Step input at t=10
    
    y = np.zeros(N_steps)
    u = np.zeros(N_steps)
    e = np.zeros(N_steps)
    
    I_acc = 0.0
    na = len(a)
    nb = len(b)
    
    for t in range(1, N_steps):
        # Error based on feedback
        e[t] = r[t] - y[t-1]
        
        # PID control law
        P_term = Kp * e[t]
        I_acc += Ki * Ts * e[t]
        # Anti-windup (limit I-term contribution)
        I_acc = np.clip(I_acc, -10.0, 10.0)
        
        # Derivative on measurement (gyro rate y) to prevent derivative kick
        if t >= 2:
            D_term = - Kd * (y[t-1] - y[t-2]) / Ts
        else:
            D_term = 0.0
        
        u_raw = P_term + I_acc + D_term
        u[t] = sign_gain * u_raw
        
        # Limit control input (actuator saturation)
        u[t] = np.clip(u[t], -500.0, 500.0)
        
        # Plant equation: y(t) = a*y_past + b*u_past
        y_val = 0.0
        for i in range(1, na + 1):
            if t - i >= 0:
                y_val += a[i-1] * y[t - i]
        for j in range(1, nb + 1):
            if t - nk - j >= 0:
                y_val += b[j-1] * u[t - nk - j]
                
        y[t] = y_val
        
        # If output diverges, stop and return large values
        if np.abs(y[t]) > 1e4 or np.isnan(y[t]):
            y[t:] = 1e4
            break
            
    return r, y, u, e

def design_controller():
    folder = Path(".")
    out = folder / "out"
    models_dir = out / "models"
    plots_dir = out / "plots"
    reports_dir = out / "reports"
    
    models_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    
    Ts = get_sampling_time(folder)
    print(f"Sampling time: {Ts:.6f} s ({1/Ts:.1f} Hz)")
    
    axes_names = {0: "Roll", 1: "Pitch", 2: "Yaw"}
    
    # Baseline gains estimated from flight logs
    # Roll: Kp = 1.44, Kd = 0.0024
    # Pitch: Kp = 1.50, Kd = 0.0069
    # Yaw: Kp = 1.43, Kd = 0
    # Let's set Ki based on typical Betaflight I/P ratios (around 1.7)
    baseline_gains = {
        0: {"Kp": 1.44, "Ki": 1.44 * 1.7, "Kd": 0.0024},
        1: {"Kp": 1.50, "Ki": 1.50 * 1.7, "Kd": 0.0069},
        2: {"Kp": 1.43, "Ki": 1.43 * 1.7, "Kd": 0.0}
    }
    
    controller_report = []
    controller_report.append("# Model-Based Controller Design and Tuning Report\n")
    
    for axis in [0, 1, 2]:
        model_path = models_dir / f"arx_model_axis{axis}.pkl"
        if not model_path.exists():
            continue
            
        with open(model_path, "rb") as f:
            data = pickle.load(f)
            
        model = data["model"]
        na = data["na"]
        nb = data["nb"]
        nk = data["nk"]
        coef = model.coef_
        
        a = coef[:na]
        b = coef[na:]
        
        # Calculate sign of DC gain to ensure negative feedback
        sum_b = np.sum(b)
        sum_a = np.sum(a)
        dc_gain = sum_b / (1.0 - sum_a)
        sign_gain = np.sign(dc_gain)
        
        axis_name = axes_names[axis]
        print(f"\nDesigning PID controller for Axis {axis} ({axis_name})...")
        
        # Cost function to minimize
        def cost_func(params):
            Kp, Ki, Kd = params
            r, y, u, e = simulate_closed_loop(a, b, nk, Ts, Kp, Ki, Kd, sign_gain)
            
            # If system diverged, return huge cost
            if np.any(np.abs(y) >= 1e3):
                return 1e9
                
            # ITAE: Integral of Time-weighted Absolute Error (skip first 10 steps before step)
            itae = 0.0
            for t in range(10, len(y)):
                time_val = (t - 10) * Ts
                itae += time_val * np.abs(r[t] - y[t])
                
            # Overshoot penalty
            overshoot = np.max(y) - 1.0
            overshoot_penalty = 1000.0 * (overshoot ** 2) if overshoot > 0.05 else 0.0
            
            # Control effort penalty (smoothness)
            du = np.diff(u)
            effort_penalty = 0.02 * np.sum(du ** 2)
            
            # Steady state error penalty
            sse_penalty = 100.0 * (np.mean(e[-20:]) ** 2)
            
            return itae + overshoot_penalty + effort_penalty + sse_penalty
            
        # Initial guess: [Kp, Ki, Kd]
        p_base = baseline_gains[axis]
        x0 = [p_base["Kp"], p_base["Ki"], p_base["Kd"]]
        
        # Bounds: Kp > 0, Ki >= 0, Kd >= 0 (and Kd=0 for Yaw)
        if axis == 2:
            bounds = [(0.1, 10.0), (0.0, 50.0), (0.0, 0.0)] # No D-term on Yaw
        else:
            bounds = [(0.1, 10.0), (0.0, 50.0), (0.0, 0.1)]
            
        res = minimize(cost_func, x0, bounds=bounds, method="L-BFGS-B")
        
        Kp_opt, Ki_opt, Kd_opt = res.x
        print(f"  Optimized PID Gains: Kp={Kp_opt:.4f}, Ki={Ki_opt:.4f}, Kd={Kd_opt:.6f}")
        print(f"  Baseline PID Gains:  Kp={p_base['Kp']:.4f}, Ki={p_base['Ki']:.4f}, Kd={p_base['Kd']:.6f}")
        
        # Simulate both controllers
        r, y_opt, u_opt, e_opt = simulate_closed_loop(a, b, nk, Ts, Kp_opt, Ki_opt, Kd_opt, sign_gain)
        _, y_base, u_base, e_base = simulate_closed_loop(a, b, nk, Ts, p_base["Kp"], p_base["Ki"], p_base["Kd"], sign_gain)
        
        # Plot and save step response comparison
        t_arr = np.arange(len(r)) * Ts * 1000 # in ms
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        
        ax1.plot(t_arr, r, 'k--', label="Setpoint", alpha=0.7)
        ax1.plot(t_arr, y_base, 'r-', label="Baseline (Flight Logs)", lw=1.5)
        ax1.plot(t_arr, y_opt, 'b-', label="Optimized PID", lw=2)
        ax1.set_ylabel("Gyro Rate (Normalized)")
        ax1.set_title(f"Closed-Loop Step Response Comparison - {axis_name} Axis")
        ax1.grid(True, linestyle=':', alpha=0.5)
        ax1.legend()
        
        ax2.plot(t_arr, u_base, 'r-', label="Baseline Actuation", alpha=0.7)
        ax2.plot(t_arr, u_opt, 'b-', label="Optimized Actuation", alpha=0.9)
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Control Signal (PID Sum)")
        ax2.grid(True, linestyle=':', alpha=0.5)
        ax2.legend()
        
        plt.tight_layout()
        plot_path = plots_dir / f"control_design_axis{axis}.png"
        plt.savefig(plot_path)
        plt.close()
        
        print(f"  Saved comparison plot to {plot_path}")
        
        # Markdown report entry
        report_section = f"""## Axis {axis}: {axis_name}

| Controller | $K_p$ | $K_i$ | $K_d$ | Features |
|---|---|---|---|---|
| **Baseline (Flight Logs)** | {p_base["Kp"]:.4f} | {p_base["Ki"]:.4f} | {p_base["Kd"]:.6f} | Stable, tuned during test flight |
| **Optimized PID (Model-Based)** | {Kp_opt:.4f} | {Ki_opt:.4f} | {Kd_opt:.6f} | Minimizes ITAE error & actuator rates |

### Step Response Simulation
![Closed-loop response comparison](file:///{plot_path.resolve().as_posix()})

### Key Design Notes
- **Feedback Sign**: The plant DC gain is {dc_gain:.6f}. A **{"negative" if dc_gain < 0 else "positive"}** feedback sign correction has been applied to the control loop.
- **Actuator Activity**: The optimized gains result in **{"smoother" if np.sum(np.diff(u_opt)**2) < np.sum(np.diff(u_base)**2) else "more aggressive"}** actuator commands compared to the baseline flight logs, balancing response speed and motor heating.
"""
        controller_report.append(report_section)
        
    # Write report
    report_file = reports_dir / "control_design_report.md"
    with open(report_file, "w") as f:
        f.writelines(controller_report)
    print(f"\nSaved control design report to {report_file}")

if __name__ == "__main__":
    design_controller()
