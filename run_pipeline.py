#!/usr/bin/env python3
"""Master pipeline execution script for multirotor system ID and control design.

Runs:
1. System Identification (Axes 0, 1, and 2 as 1st-order models)
2. Model Analysis (Poles, stability, DC gain, and Bode plots)
3. Controller Design (ITAE PID optimization and step response simulation)
"""
import subprocess
import sys
from pathlib import Path

import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None, help="Specific CSV log file to parse (e.g. btfl_all.24.csv)")
    args = ap.parse_args()

    python_bin = sys.executable

    # Ensure output directory structure exists
    out = Path("out")
    for sub in ["csv", "models", "plots", "reports"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    
    print("=================================================================")
    print("   Multirotor System ID and Control Design Master Pipeline       ")
    print("=================================================================\n")
    
    # Step 1: System Identification
    print("--- STEP 1: Running System Identification for Roll, Pitch, and Yaw ---")
    for axis in [0, 1, 2]:
        axis_name = {0: "Roll", 1: "Pitch", 2: "Yaw"}[axis]
        print(f"Fitting 1st-order model for Axis {axis} ({axis_name})...")
        
        cmd = [python_bin, "id_pipeline.py", "--axis", str(axis), "--na", "1", "--nb", "1", "--nk", "2"]
        if args.file:
            cmd += ["--file", args.file]
            
        try:
            subprocess.run(cmd, check=True)
            print(f"  Axis {axis_name} Model Fit Complete.\n")
        except subprocess.CalledProcessError as e:
            print(f"[Error] Error during System ID for Axis {axis_name}: {e}")
            sys.exit(1)
            
    # Step 2: Model Analysis
    print("--- STEP 2: Running Frequency-Domain and Stability Analysis ---")
    try:
        subprocess.run([python_bin, "analyze_models.py"], check=True)
        print("  Model Analysis and Report Generation Complete.\n")
    except subprocess.CalledProcessError as e:
        print(f"[Error] Error during Model Analysis: {e}")
        sys.exit(1)
        
    # Step 3: Controller Design
    print("--- STEP 3: Optimizing PID Controller Gains and Simulating Response ---")
    try:
        subprocess.run([python_bin, "design_controller.py"], check=True)
        print("  Controller Design and Report Generation Complete.\n")
    except subprocess.CalledProcessError as e:
        print(f"[Error] Error during Controller Design: {e}")
        sys.exit(1)

    # Step 4: Log Manifest Generation
    print("--- STEP 4: Compiling Log Manifest ---")
    try:
        subprocess.run([python_bin, "id_pipeline.py", "--compile-manifest"], check=True)
        print("  Log Manifest Generation Complete.\n")
    except subprocess.CalledProcessError as e:
        print(f"[Error] Error during Log Manifest Generation: {e}")
        sys.exit(1)
        
    print("=================================================================")
    print("   Pipeline Execution Complete!                                  ")
    print("   Outputs generated in the 'out/' directory:                    ")
    print("     - ARX Fit Plots: out/plots/fit_axis{0,1,2}.png              ")
    print("     - Pole-Zero Maps: out/plots/pzmap_axis{0,1,2}.png           ")
    print("     - Bode Plots: out/plots/bode_axis{0,1,2}.png                ")
    print("     - Control Design Plots: out/plots/control_design_axis{0,1,2}.png")
    print("     - Reports: out/reports/model_analysis_report.md, out/reports/control_design_report.md")
    print("     - Log Manifests: out/log_manifest.json, out/log_manifest.csv")
    print("=================================================================")

if __name__ == "__main__":
    main()
