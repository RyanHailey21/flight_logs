import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from id_pipeline import ConstrainedModel


def main():
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
    Ts = data["Ts_down"] # Use downsampled Ts (~0.1 s)
    coef = model.coef_
    intercept = model.intercept_
    
    a = coef[:na]
    b = coef[na:]
    
    # Calculate poles and zeros
    # Denominator: z^2 - a_1 z - a_2 = 0
    den_poly = np.zeros(na + 1)
    den_poly[0] = 1.0
    den_poly[1:] = -a
    poles = np.roots(den_poly)
    
    # Numerator: b_1 z + b_2 = 0 (for nb=2)
    # Zeros of transfer function G(z) = B(z)/A(z)
    num_poly = b
    zeros = np.roots(num_poly) if len(b) > 1 else np.array([])
    
    # DC Gain: G(1)
    sum_b = np.sum(b)
    sum_a = np.sum(a)
    dc_gain = sum_b / (1.0 - sum_a)
    
    # Stability check: all poles inside unit circle
    stable = all(np.abs(p) < 1.0 for p in poles)
    stability_str = "Stable" if stable else "UNSTABLE"
    
    print("\n=== Barometer Altitude Model Analysis ===")
    print(f"  Sampling time: {Ts:.4f} s ({1.0/Ts:.2f} Hz)")
    print(f"  Poles: {poles}")
    print(f"  Poles magnitudes: {np.abs(poles)}")
    print(f"  Zeros: {zeros}")
    print(f"  DC Gain: {dc_gain:.6f}")
    print(f"  Stability: {stability_str}")
    print(f"  Intercept: {intercept:.6f}")
    
    # Format transfer function for report
    num_terms = []
    for i, val in enumerate(b, 1):
        num_terms.append(f"{val:+.4f} z^{{-{i}}}")
    num_str = " ".join(num_terms).strip()
    if num_str.startswith("+"):
        num_str = num_str[1:].strip()
        
    den_terms = ["1"]
    for i, val in enumerate(a, 1):
        den_terms.append(f"{-val:+.4f} z^{{-{i}}}")
    den_str = " ".join(den_terms).strip()
    
    # Create report markdown
    report_content = []
    report_content.append("# Barometer Altitude Model Analysis Report\n\n")
    report_content.append("## Discrete-Time Transfer Function Model\n\n")
    report_content.append(f"- **Discrete Transfer Function**:\n\n")
    report_content.append(f"$$G(z) = \\frac{{{num_str}}}{{{den_str}}} z^{{-{nk}}}$$\n\n")
    report_content.append(f"- **Downsampled Sampling Time ($T_s$)**: {Ts:.4f} s ({1.0/Ts:.2f} Hz)\n")
    report_content.append(f"- **Poles**: {', '.join([f'{p.real:.4f} + {p.imag:.4f}j' if p.imag != 0 else f'{p.real:.4f}' for p in poles])}\n")
    report_content.append(f"- **Poles Magnitudes**: {', '.join([f'{np.abs(p):.4f}' for p in poles])}\n")
    if len(zeros) > 0:
        report_content.append(f"- **Zeros**: {', '.join([f'{z.real:.4f} + {z.imag:.4f}j' if z.imag != 0 else f'{z.real:.4f}' for z in zeros])}\n")
    else:
        report_content.append(f"- **Zeros**: None\n")
    report_content.append(f"- **DC Gain**: {dc_gain:.6f}\n")
    report_content.append(f"- **Stability**: **{stability_str}**\n")
    report_content.append(f"- **Intercept**: {intercept:.6f}\n\n")
    
    # Plot Pole-Zero Map
    plt.figure(figsize=(6, 6))
    theta = np.linspace(0, 2*np.pi, 200)
    plt.plot(np.cos(theta), np.sin(theta), 'k--', label='Unit Circle', alpha=0.5)
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    
    plt.scatter(poles.real, poles.imag, color='red', marker='x', s=100, label='Poles', zorder=3)
    if len(zeros) > 0:
        plt.scatter(zeros.real, zeros.imag, color='blue', marker='o', s=100, facecolors='none', edgecolors='blue', label='Zeros', zorder=3)
        
    plt.title("Pole-Zero Map - Barometer Altitude Model")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend()
    plt.axis('equal')
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.tight_layout()
    
    pz_path = plots_dir / "pzmap_altitude.png"
    plt.savefig(pz_path)
    plt.close()
    
    # Compute frequency response (Bode plot)
    frequencies = np.logspace(-1, np.log10(1 / (2 * Ts)), 500) # 0.1 Hz to Nyquist
    omega = 2 * np.pi * frequencies
    
    # z = e^(j * omega * Ts)
    z = np.exp(1j * omega * Ts)
    
    # Evaluate transfer function G(z)
    num = sum(b_j * (z**(-j)) for j, b_j in enumerate(b, 1))
    den = 1.0 - sum(a_i * (z**(-i)) for i, a_i in enumerate(a, 1))
    G = (z**(-nk)) * num / den
    
    magnitude_db = 20 * np.log10(np.abs(G))
    phase_deg = np.angle(G) * 180 / np.pi
    phase_deg = np.unwrap(phase_deg)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    ax1.semilogx(frequencies, magnitude_db, color='blue', lw=2)
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_title("Bode Plot - Barometer Altitude Model")
    ax1.grid(True, which='both', linestyle=':', alpha=0.5)
    
    ax2.semilogx(frequencies, phase_deg, color='red', lw=2)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (deg)")
    ax2.grid(True, which='both', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    bode_path = plots_dir / "bode_altitude.png"
    plt.savefig(bode_path)
    plt.close()
    
    print(f"  Saved Pole-Zero Map to {pz_path}")
    print(f"  Saved Bode Plot to {bode_path}")
    
    # Write report
    report_file = reports_dir / "model_analysis_report.md"
    with open(report_file, "w") as f:
        f.writelines(report_content)
    print(f"Saved analysis report to {report_file}")


if __name__ == "__main__":
    main()
