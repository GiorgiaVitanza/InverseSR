import os
import matplotlib
# Backend non-interattivo per HPC/cluster senza interfaccia grafica
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp


def compute_power_spectrum_and_cross(synth, target, lx=128.0, ly=128.0, lz=16.0):
    """
    Calcola il Power Spectrum Normalizzato, il Cross-Correlation Coefficient r(k)
    e la Transfer Function T(k) tra SR e Target.
    """
    synth = np.squeeze(synth).astype(np.float64)
    target = np.squeeze(target).astype(np.float64)

    Nz, Ny, Nx = synth.shape
    V = lx * ly * lz

    # Standardizzazione (Media = 0, Varianza = 1)
    s_norm = (synth - np.mean(synth)) / (np.std(synth) + 1e-8)
    t_norm = (target - np.mean(target)) / (np.std(target) + 1e-8)

    # FFT 3D
    fft_s = np.fft.fftn(s_norm)
    fft_t = np.fft.fftn(t_norm)

    # Auto e Cross Power Spectra
    pk_s_3d = (np.abs(fft_s) ** 2) * (V / (Nx * Ny * Nz) ** 2)
    pk_t_3d = (np.abs(fft_t) ** 2) * (V / (Nx * Ny * Nz) ** 2)
    pk_st_3d = np.real(fft_s * np.conj(fft_t)) * (V / (Nx * Ny * Nz) ** 2)

    # Frequenze k per ogni asse
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=lx / Nx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=ly / Ny)
    kz = 2 * np.pi * np.fft.fftfreq(Nz, d=lz / Nz)

    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
    k_mag = np.sqrt(KX**2 + KY**2 + KZ**2)

    k_flat = k_mag.flatten()
    pks_flat = pk_s_3d.flatten()
    pkt_flat = pk_t_3d.flatten()
    pkst_flat = pk_st_3d.flatten()

    # Rimuove la componente DC (k = 0)
    mask = k_flat > 0
    k_flat = k_flat[mask]
    pks_flat = pks_flat[mask]
    pkt_flat = pkt_flat[mask]
    pkst_flat = pkst_flat[mask]

    k_min = min(2 * np.pi / lx, 2 * np.pi / ly, 2 * np.pi / lz)
    k_max = k_flat.max()

    num_bins = max(min(Nx, Ny, Nz) // 2, 8)
    k_bins = np.linspace(k_min, k_max, num_bins)
    digits = np.digitize(k_flat, k_bins)

    k_vals, pks_vals, pkt_vals, r_k_vals, tk_vals = [], [], [], [], []
    for i in range(1, len(k_bins)):
        mask_bin = digits == i
        if np.any(mask_bin):
            k_m = k_flat[mask_bin].mean()
            ps = pks_flat[mask_bin].mean()
            pt = pkt_flat[mask_bin].mean()
            pst = pkst_flat[mask_bin].mean()

            # Cross-correlation coefficient r(k)
            r_k = pst / (np.sqrt(ps * pt) + 1e-8)
            # Transfer Function T(k) = P_SR / P_Target
            t_k = ps / (pt + 1e-8)

            k_vals.append(k_m)
            pks_vals.append(ps)
            pkt_vals.append(pt)
            r_k_vals.append(r_k)
            tk_vals.append(t_k)

    return (
        np.array(k_vals),
        np.array(pks_vals),
        np.array(pkt_vals),
        np.array(r_k_vals),
        np.array(tk_vals),
    )


def save_power_spectrum_plots(k_vals, pk_sr, pk_ref, r_k, t_k, out_dir):
    """Genera e salva un grafico a 3 pannelli con Power Spectrum, r(k) e T(k)."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    # 1. Power Spectrum P(k)
    ax1.loglog(k_vals, pk_ref, "k-", label="Target (Original)", linewidth=2)
    ax1.loglog(
        k_vals, pk_sr, "r--", label="Reconstructed (SR)", linewidth=2
    )
    ax1.set_ylabel(r"$P(k)$ [Norm]")
    ax1.set_title(
        "Power Spectrum & Structural Fidelity Analysis",
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # 2. Cross-Correlation r(k)
    ax2.semilogx(
        k_vals,
        r_k,
        "b-o",
        label=r"$r(k) = P_{ST} / \sqrt{P_S P_T}$",
        linewidth=2,
    )
    ax2.axhline(0.90, color="green", linestyle=":", label="Soglia r(k) = 0.90")
    ax2.set_ylabel(r"$r(k)$ (Correlazione)")
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend(loc="lower left")

    # 3. Transfer Function T(k)
    ax3.semilogx(
        k_vals,
        t_k,
        "m-s",
        label=r"$T(k) = P_{SR}(k) / P_{Target}(k)$",
        linewidth=2,
    )
    ax3.axhline(
        1.0, color="black", linestyle="--", alpha=0.7, label="Risposta Ideale (1.0)"
    )
    ax3.set_xlabel(r"k [1/pixel]")
    ax3.set_ylabel(r"$T(k)$ (Transfer Func)")
    ax3.set_ylim(0.0, max(2.0, np.max(t_k) * 1.1))
    ax3.grid(True, which="both", ls="--", alpha=0.5)
    ax3.legend(loc="upper right")

    plt.tight_layout()

    # Salva sia in PNG (per rapida consultazione) che in PDF (vettoriale per tesi/paper)
    png_path = os.path.join(out_dir, "power_spectrum_analysis.png")
    pdf_path = os.path.join(out_dir, "power_spectrum_analysis.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"\n✅ Grafici salvati con successo:")
    print(f" ├── PNG: {png_path}")
    print(f" └── PDF: {pdf_path}")


if __name__ == "__main__":
    BASE_PATH = "./data/outputs/BRGM_ddim_z3_lambda50_global_arcsinh_cont_ldev_None_600steps_70ddim_NOPERC/restoration_outputs"

    synth_path = os.path.join(BASE_PATH, "reconstructed_synth.npy")
    target_path = os.path.join(BASE_PATH, "target_original.npy")

    synth_cube = np.load(synth_path)
    target_cube = np.load(target_path)

    # Calcolo di tutte le metriche
    k_vals, pk_sr_norm, pk_ref_norm, r_k, t_k = (
        compute_power_spectrum_and_cross(
            synth_cube, target_cube, lx=128.0, ly=128.0, lz=16.0
        )
    )

    # Salva i grafici su disco
    save_power_spectrum_plots(
        k_vals, pk_sr_norm, pk_ref_norm, r_k, t_k, BASE_PATH
    )