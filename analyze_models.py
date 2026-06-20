#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from id_pipeline import ConstrainedModel, get_sampling_time

def main():
    folder = Path(".")
    out = folder / "out"
    Ts = get_sampling_time(folder)
    
    axes_names = {0: "Roll", 1: "Pitch", 2: "Yaw"}
    
    analysis_report = []
    analysis_report.append("# System Identification Model Analysis Report\n")
    
    for axis in [0, 1, 2]:
        model_path = out / f"arx_model_axis{axis}.pkl"
        if not model_path.exists():
            print(f"Model for axis {axis} not found at {model_path}, skipping.")
            continue
            
        with open(model_path, "rb") as f:
            data = pickle.load(f)
            
        model = data["model"]
        na = data["na"]
        nb = data["nb"]
        nk = data["nk"]
        coef = model.coef_
        intercept = model.intercept_
        
        a = coef[:na]
        b = coef[na:]
        
        # Calculate poles and zeros
        # Denominator: z^na - a_1 z^(na-1) - ... - a_na = 0
        den_poly = np.zeros(na + 1)
        den_poly[0] = 1.0
        den_poly[1:] = -a
        poles = np.roots(den_poly)
        
        # Numerator: b_1 z^(nb-1) + b_2 z^(nb-2) + ... + b_nb = 0
        num_poly = b
        zeros = np.roots(num_poly)
        
        # DC Gain: G(1)
        sum_b = np.sum(b)
        sum_a = np.sum(a)
        dc_gain = sum_b / (1.0 - sum_a)
        
        # Stability check
        stable = all(np.abs(p) < 1.0 for p in poles)
        stability_str = "Stable" if stable else "UNSTABLE"
        
        # Output info
        axis_name = axes_names[axis]
        print(f"\n=== Axis {axis} ({axis_name}) ===")
        print(f"  Poles: {poles}")
        print(f"  Poles magnitudes: {np.abs(poles)}")
        print(f"  Zeros: {zeros}")
        print(f"  Zeros magnitudes: {np.abs(zeros)}")
        print(f"  DC Gain: {dc_gain:.6f}")
        print(f"  Stability: {stability_str}")
        print(f"  Intercept: {intercept:.6f}")
        
        # Dynamic LaTeX string for transfer function
        num_terms = []
        for i, val in enumerate(b, 1):
            num_terms.append(f"{val:+.4f} z^{{-{i}}}")
        num_str = " ".join(num_terms).strip()
        if num_str.startswith("+"):
            num_str = num_str[1:].strip()
            
        den_terms = ["1"]
        for i, val in enumerate(a, 1):
            # Note: A(z) = 1 - a_1 z^-1 - a_2 z^-2 ...
            den_terms.append(f"{-val:+.4f} z^{{-{i}}}")
        den_str = " ".join(den_terms).strip()
        
        # Markdown summary formatting
        report_section = f"""## Axis {axis}: {axis_name}

- **Discrete Transfer Function**:
  $$G(z) = \\frac{{{num_str}}}{{{den_str}}} z^{{-{nk}}}$$
- **Poles**: {', '.join([f"{p.real:.4f} + {p.imag:.4f}j" if p.imag != 0 else f"{p.real:.4f}" for p in poles])}
- **Poles Magnitudes**: {', '.join([f"{np.abs(p):.4f}" for p in poles])}
- **Zeros**: {', '.join([f"{z.real:.4f} + {z.imag:.4f}j" if z.imag != 0 else f"{z.real:.4f}" for z in zeros])}
- **DC Gain**: {dc_gain:.6f}
- **Stability**: **{stability_str}**
- **Intercept**: {intercept:.6f}
"""
        analysis_report.append(report_section)
        
        # Plot Pole-Zero Map
        plt.figure(figsize=(6, 6))
        theta = np.linspace(0, 2*np.pi, 200)
        plt.plot(np.cos(theta), np.sin(theta), 'k--', label='Unit Circle', alpha=0.5)
        plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5)
        
        plt.scatter(poles.real, poles.imag, color='red', marker='x', s=100, label='Poles', zorder=3)
        if len(zeros) > 0:
            plt.scatter(zeros.real, zeros.imag, color='blue', marker='o', s=100, facecolors='none', edgecolors='blue', label='Zeros', zorder=3)
            
        plt.title(f"Pole-Zero Map - {axis_name} Axis")
        plt.xlabel("Real")
        plt.ylabel("Imaginary")
        plt.grid(True, which='both', linestyle=':', alpha=0.5)
        plt.legend()
        plt.axis('equal')
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.tight_layout()
        pz_path = out / f"pzmap_axis{axis}.png"
        plt.savefig(pz_path)
        plt.close()
        
        # Compute frequency response (Bode plot)
        frequencies = np.logspace(0, np.log10(1 / (2 * Ts)), 500) # 1 Hz to Nyquist
        omega = 2 * np.pi * frequencies
        
        # z = e^(j * omega * Ts)
        z = np.exp(1j * omega * Ts)
        
        # Evaluate transfer function G(z)
        num = sum(b_j * (z**(-j)) for j, b_j in enumerate(b, 1))
        den = 1.0 - sum(a_i * (z**(-i)) for i, a_i in enumerate(a, 1))
        G = (z**(-nk)) * num / den
        
        magnitude_db = 20 * np.log10(np.abs(G))
        phase_deg = np.angle(G) * 180 / np.pi
        # Unwrap phase to avoid 360 deg jumps
        phase_deg = np.unwrap(phase_deg)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        
        ax1.semilogx(frequencies, magnitude_db, color='blue', lw=2)
        ax1.set_ylabel("Magnitude (dB)")
        ax1.set_title(f"Bode Plot - {axis_name} Axis")
        ax1.grid(True, which='both', linestyle=':', alpha=0.5)
        
        ax2.semilogx(frequencies, phase_deg, color='red', lw=2)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (deg)")
        ax2.grid(True, which='both', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        bode_path = out / f"bode_axis{axis}.png"
        plt.savefig(bode_path)
        plt.close()
        
        print(f"  Saved Pole-Zero Map to {pz_path}")
        print(f"  Saved Bode Plot to {bode_path}")

    # Write report
    report_file = out / "model_analysis_report.md"
    with open(report_file, "w") as f:
        f.writelines(analysis_report)
    print(f"\nSaved analysis report to {report_file}")

if __name__ == "__main__":
    main()
