#!/usr/bin/env python3
"""Model-based PID controller design and closed-loop step response simulation for barometer altitude control."""
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from id_pipeline import ConstrainedModel


def simulate_closed_loop_altitude_cascaded(a, b, nk, Ts, HOLD_KP, HOLD_KD, HOLD_KI, u_hover, intercept, N_steps=300):
    """Simulates closed-loop altitude step response using the flight controller's exact cascaded structure."""
    # Control configuration parameters matching ESP32 firmware
    ALT_RAMP_RATE_MPS = 3.5     # Reference altitude ramp limit (m/s)
    NEAR_TARGET_M = 0.5         # Near-target zone (meters)
    NEAR_TARGET_FACTOR = 0.5    # Cushioning scaling factor
    MAX_CLIMB_MPS_HOLD = 3.5    # Maximum climb speed limit (m/s)
    MAX_DESCENT_MPS_HOLD = 1.0  # Maximum descent speed limit (m/s)
    THR_DOWN_OFFSET_US = 250.0  # Maximum allowable decrease from hover throttle
    THR_UP_OFFSET_US = 350.0    # Maximum allowable increase from hover throttle
    KI_VSPEED_LIMIT = 10.0      # Inner integral saturation limit
    
    r = np.zeros(N_steps)
    r[20:] = 18.3  # Setpoint step of 18.3 meters at t = 2.0 seconds
    
    y = np.zeros(N_steps)
    u = np.zeros(N_steps)
    e = np.zeros(N_steps)
    
    # Initialize past command history to hover throttle
    u[:] = u_hover
    
    vspeedIntegral = 0.0
    internalSetpoint = 0.0  # Bumpless start from current altitude (0.0 m)
    
    na = len(a)
    nb = len(b)
    
    for t in range(2, N_steps):
        dt = Ts
        
        # Current and previous altitude in meters (natively in meters)
        altitude_m = y[t-1]
        altitude_last_m = y[t-2]
        
        # 1. Reference shaping: ramp internal setpoint toward target altitude
        target_m = r[t]
        if internalSetpoint < target_m:
            internalSetpoint = min(internalSetpoint + ALT_RAMP_RATE_MPS * dt, target_m)
        elif internalSetpoint > target_m:
            internalSetpoint = max(internalSetpoint - ALT_RAMP_RATE_MPS * dt, target_m)
            
        # 2. Outer loop: altitude error -> desired vertical speed
        altError = internalSetpoint - altitude_m
        e[t] = altError  # Error in meters for cost evaluation
        
        maxClimb = MAX_CLIMB_MPS_HOLD
        maxDesc = MAX_DESCENT_MPS_HOLD
        
        # Near target cushioning
        if abs(altError) < NEAR_TARGET_M:
            factor = abs(altError) / NEAR_TARGET_M
            factor = NEAR_TARGET_FACTOR + (1.0 - NEAR_TARGET_FACTOR) * factor
            maxClimb *= factor
            maxDesc *= factor
            
        desiredVspeed = np.clip(HOLD_KP * altError, -maxDesc, maxClimb)
        
        # 3. filteredVario (vertical velocity measurement)
        filteredVario = (altitude_m - altitude_last_m) / dt
        
        # 4. Inner PI loop with conditional anti-windup
        vspeedError = desiredVspeed - filteredVario
        thrMin = u_hover - THR_DOWN_OFFSET_US
        thrMax = u_hover + THR_UP_OFFSET_US
        
        rawThrottle = u_hover + HOLD_KD * vspeedError + HOLD_KI * vspeedIntegral
        satHigh = (rawThrottle > thrMax) and (vspeedError > 0)
        satLow = (rawThrottle < thrMin) and (vspeedError < 0)
        
        # Only integrate when not saturated in the direction of the error
        if not satHigh and not satLow:
            vspeedIntegral += vspeedError * dt
            vspeedIntegral = np.clip(vspeedIntegral, -KI_VSPEED_LIMIT, KI_VSPEED_LIMIT)
            
        finalThrottle = u_hover + HOLD_KD * vspeedError + HOLD_KI * vspeedIntegral
        u[t] = np.clip(finalThrottle, thrMin, thrMax)
        
        # Plant equation (2nd-order ARX): y(t) = a*y_past + b*u_past + intercept
        # Corrected: use steady-state values (0 for y, u_hover for u) for negative time indices
        y_val = intercept
        for i in range(1, na + 1):
            val_y = y[t - i] if t - i >= 0 else 0.0
            y_val += a[i-1] * val_y
        for j in range(1, nb + 1):
            val_u = u[t - nk - j] if t - nk - j >= 0 else u_hover
            y_val += b[j-1] * val_u
                
        y[t] = y_val
        
        # Handle divergence
        if np.abs(y[t]) > 1e3 or np.isnan(y[t]):
            y[t:] = 1e3
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
    
    model_path = models_dir / "arx_model_altitude.pkl"
    if not model_path.exists():
        print(f"Altitude model not found at {model_path}. Exiting.")
        return
        
    with open(model_path, "rb") as f:
        data = pickle.load(f)
        
    model = data["model"]
    na = data["na"]
    nb = data["nb"]
    nk = data["nk"]
    Ts = data["Ts_down"]  # ~0.1 s
    coef = model.coef_
    intercept = model.intercept_
    
    a = coef[:na]
    b = coef[na:]
    
    # Calculate hover throttle bias
    sum_b = np.sum(b)
    if np.abs(sum_b) > 1e-6:
        u_hover = -intercept / sum_b
    else:
        u_hover = 1350.0
    u_hover = np.clip(u_hover, 1100.0, 1600.0)
    
    print(f"\nDesigning Cascaded PID controller for Altitude...")
    print(f"  Calculated Steady-State Hover Throttle: {u_hover:.1f}")
    
    # Cost function to minimize
    def cost_func(params):
        HOLD_KP, HOLD_KD, HOLD_KI = params
        r, y, u, e = simulate_closed_loop_altitude_cascaded(a, b, nk, Ts, HOLD_KP, HOLD_KD, HOLD_KI, u_hover, intercept)
        
        if np.any(np.abs(y) >= 1e3):
            return 1e9
            
        # ITAE: Integral of Time-weighted Absolute Error (scaled by 100 to match cm landscape)
        itae = 0.0
        for t in range(20, len(y)):
            time_val = (t - 20) * Ts
            itae += time_val * np.abs(r[t] - y[t]) * 100.0
            
        # Overshoot penalty (target setpoint is 18.3 m, tolerate up to 10% or 1.83 m)
        overshoot = np.max(y) - 18.3
        overshoot_penalty = 1000.0 * ((overshoot * 100.0) ** 2) if overshoot > 1.83 else 0.0
        
        # Control effort penalty (changes in throttle command to prevent twitching)
        du = np.diff(u)
        effort_penalty = 0.1 * np.sum(du ** 2)
        
        # Steady-state error penalty (scaled to cm)
        sse_penalty = 100.0 * ((np.mean(e[-30:]) * 100.0) ** 2)
        
        return itae + overshoot_penalty + effort_penalty + sse_penalty
        
    # Baseline gains for comparison
    p_base = {"HOLD_KP": 1.0, "HOLD_KD": 120.0, "HOLD_KI": 15.0}
    x0 = [p_base["HOLD_KP"], p_base["HOLD_KD"], p_base["HOLD_KI"]]
    
    # Optimization Bounds (Safe flight envelope to prevent noise amplification):
    # KP (0.1 -> 3.0), KD (10.0 -> 200.0), KI (0.0 -> 40.0)
    bounds = [(0.1, 3.0), (10.0, 200.0), (0.0, 40.0)]
    
    res = minimize(cost_func, x0, bounds=bounds, method="L-BFGS-B")
    HOLD_KP_opt, HOLD_KD_opt, HOLD_KI_opt = res.x
    
    print(f"  Optimized Cascaded Gains: HOLD_KP={HOLD_KP_opt:.4f}, HOLD_KD={HOLD_KD_opt:.4f}, HOLD_KI={HOLD_KI_opt:.4f}")
    print(f"  Baseline Cascaded Gains:  HOLD_KP={p_base['HOLD_KP']:.4f}, HOLD_KD={p_base['HOLD_KD']:.4f}, HOLD_KI={p_base['HOLD_KI']:.4f}")
    
    # Simulate both controllers
    r, y_opt, u_opt, e_opt = simulate_closed_loop_altitude_cascaded(a, b, nk, Ts, HOLD_KP_opt, HOLD_KD_opt, HOLD_KI_opt, u_hover, intercept)
    _, y_base, u_base, e_base = simulate_closed_loop_altitude_cascaded(a, b, nk, Ts, p_base["HOLD_KP"], p_base["HOLD_KD"], p_base["HOLD_KI"], u_hover, intercept)
    
    # Plot results
    t_arr = np.arange(len(r)) * Ts  # in seconds
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    ax1.plot(t_arr, r, 'k--', label="Setpoint", alpha=0.7)
    ax1.plot(t_arr, y_base, 'r-', label="Baseline Controller", lw=1.5)
    ax1.plot(t_arr, y_opt, 'b-', label="Optimized Cascaded PID", lw=2)
    ax1.set_ylabel("Altitude (m)")
    ax1.set_title("Closed-Loop Altitude Step Response (Cascaded Loop)")
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend()
    
    ax2.plot(t_arr, u_base, 'r-', label="Baseline Throttle", alpha=0.7)
    ax2.plot(t_arr, u_opt, 'b-', label="Optimized Throttle", alpha=0.9)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Throttle command (rcCommand[3])")
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    plot_path = plots_dir / "control_design_altitude.png"
    plt.savefig(plot_path)
    plt.close()
    
    print(f"  Saved comparison plot to {plot_path}")
    
    # Write report
    report_content = []
    report_content.append("# Model-Based Altitude Controller Design and Tuning Report\n\n")
    report_content.append("| Controller | HOLD_KP (Outer P) | HOLD_KD (Inner P) | HOLD_KI (Inner I) | Features |\n")
    report_content.append("|---|---|---|---|---|\n")
    report_content.append(f"| **Baseline Controller** | {p_base['HOLD_KP']:.4f} | {p_base['HOLD_KD']:.4f} | {p_base['HOLD_KI']:.4f} | Conservative default gains |\n")
    report_content.append(f"| **Optimized PID (Model-Based)** | {HOLD_KP_opt:.4f} | {HOLD_KD_opt:.4f} | {HOLD_KI_opt:.4f} | Minimizes ITAE error & overshoot |\n\n")
    
    report_content.append("### Step Response Simulation\n")
    report_content.append(f"![Closed-loop response comparison](../plots/{plot_path.name})\n\n")
    
    report_content.append("### Key Design Notes\n")
    report_content.append(f"- **Calculated Hover Throttle**: {u_hover:.1f}\n")
    report_content.append(f"- **Structure Matching**: The tuning simulation uses the exact cascaded structure running in the ESP32 firmware (Proportional outer loop on altitude error + Proportional-Integral inner loop on vertical speed error) with saturation clipping and anti-windup.\n")
    report_content.append(f"- **Gains Interpretation**:\n")
    report_content.append(f"  - `HOLD_KP` converts altitude error (meters) to target climbing/descending rate (meters/second).\n")
    report_content.append(f"  - `HOLD_KD` converts vertical speed error (meters/second) to pulse width modulation throttle offset (microseconds).\n")
    report_content.append(f"  - `HOLD_KI` integrates the speed error (converting to meters) to command throttle bias.\n")
    
    report_file = reports_dir / "control_design_report.md"
    with open(report_file, "w") as f:
        f.writelines(report_content)
    print(f"Saved control design report to {report_file}")
    
    # Auto-generate C++ header file for integration
    header_file = out / "altitude_gains.h"
    header_content = f"""// Auto-generated by design_controller.py
// This file contains model-based optimized controller gains.
#pragma once

namespace AltitudeParameters {{
    constexpr float HOLD_KP = {HOLD_KP_opt:.6f}f;
    constexpr float HOLD_KD = {HOLD_KD_opt:.6f}f;
    constexpr float HOLD_KI = {HOLD_KI_opt:.6f}f;
    constexpr float HOVER_THROTTLE = {u_hover:.6f}f;
}}
"""
    with open(header_file, "w") as f:
        f.write(header_content)
    print(f"Saved C++ header parameters to {header_file}")


if __name__ == "__main__":
    design_controller()
