#!/usr/bin/env python3
"""Master pipeline execution script for multirotor barometer altitude control.

Runs:
1. Altitude System Identification (2nd-order model fit at 10 Hz)
2. Model Analysis (Poles, stability, DC gain, and Bode plots)
3. Controller Design (ITAE PID optimization and step response simulation)
4. Log Manifest compilation
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
    print("   Multirotor Barometer Altitude Control Master Pipeline        ")
    print("=================================================================\n")
    
    # Step 1: System Identification
    print("--- STEP 1: Running System Identification for Altitude ---")
    print("Fitting 2nd-order altitude model...")
    
    cmd = [python_bin, "id_pipeline.py"]
    if args.file:
        cmd += ["--file", args.file]
        
    try:
        subprocess.run(cmd, check=True)
        print("  Altitude Model Fit Complete.\n")
    except subprocess.CalledProcessError as e:
        print(f"[Error] Error during Altitude System ID: {e}")
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
    print("     - ARX Fit Plot: out/plots/fit_altitude.png                  ")
    print("     - Pole-Zero Map: out/plots/pzmap_altitude.png               ")
    print("     - Bode Plot: out/plots/bode_altitude.png                    ")
    print("     - Control Design Plot: out/plots/control_design_altitude.png")
    print("     - Reports: out/reports/model_analysis_report.md, out/reports/control_design_report.md")
    print("     - Log Manifests: out/log_manifest.json, out/log_manifest.csv")
    print("     - C++ Header Parameters: out/altitude_gains.h               ")
    print("=================================================================")

if __name__ == "__main__":
    main()
