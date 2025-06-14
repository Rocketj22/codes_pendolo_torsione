'''val = [1, 2, 3]
err = [2, 3, 4]
x = unp.uarray(val, err)
print(f"{x[0] + x[1]:.2uS}")'''


import os
import re
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from matplotlib.ticker import MultipleLocator, FuncFormatter
import fractions

# === 0. PARAMETRI ===
frequency = 50
print_option = 0
mode = 1
# 0 decide the range where to work
# 1 do the job
data_dir = r"C:\Users\loren\Desktop\Torsione\data"
data_path = os.path.join(data_dir, "*.txt")
t_range_dir = r"C:\Users\loren\Desktop\Torsione"
t_range_path = os.path.join(t_range_dir, "t_range.txt")

pdf_folder_risonanza = r"C:\Users\loren\Desktop\Torsione\grafici\risonanza"
pdf_folder_smorzamento = r"C:\Users\loren\Desktop\Torsione\grafici\smorzamento"
pdf_folder_fitMassimi = r"C:\Users\loren\Desktop\Torsione\grafici\fitMassimi"
pdf_folder_forzante = r"C:\Users\loren\Desktop\Torsione\grafici\forzante"
pdf_folder_scarto_smorzamento = r"C:\Users\loren\Desktop\Torsione\grafici\scarto_smorzamento"

risonanza_out = r"C:\Users\loren\Desktop\Torsione\risonanza_param.txt"
smorzamento_out = r"C:\Users\loren\Desktop\Torsione\smorzamento_param.txt"
forzante_out = r"C:\Users\loren\Desktop\Torsione\forzante_param.txt"

# reset the file before working on it
if mode == 0:
    open(t_range_path, "w").close()

r_reset = open(risonanza_out, 'w').close()
s_reset = open(smorzamento_out, 'w').close()
f_reset = open(forzante_out, 'w').close()

r = open(risonanza_out, 'a')
s = open(smorzamento_out, 'a')
f = open(forzante_out, 'a')
r.write("file\tA\tA_err\tomega\tomega_err\tphi\tphi_err\tC\tC_err\tchi^2_red\n")
s.write("file\tA\tA_err\tlam\tlam_err\tomega\tomega_err\tphi\tphi_err\tchi^2_red\n")
f.write("file\tA\tA_err\tomega\tomega_err\tfrequenza\tfrequenza_err\tphi\tphi_err\tC\tC_err\tchi^2_red\n")



# === 1. FUNCTIONS ===
def seno_param(t, A, omega, phi, C):
    return A * np.sin(omega*t + phi) + C


def damped_sine(t, A, lam, omega, phi, C):
        """
        seno smorzato
        """
        return A * np.exp(-lam * t) * np.sin(omega * t + phi) + C


def exp_fit(x, m, q):
    return np.exp(m * x + q)


def pi_fraction_formatter(x, pos):
    frac = fractions.Fraction(x/np.pi).limit_denominator(8)  # Limita ai denominatori tipo 2, 4, 8
    num, den = frac.numerator, frac.denominator

    if x == 0:
        return "0"
    sign = "-" if num < 0 else ""
    num = abs(num)

    if num == 0:
        return "0"
    elif num == 1 and den == 1:
        return r"${}\pi$".format(sign)
    elif den == 1:
        return r"${}{}\pi$".format(sign, num)
    else:
        # Es: $\frac{\pi}{4}$, $\frac{3\pi}{8}$
        pi_str = r"\pi" if num == 1 else r"{}\pi".format(num)
        return r"${}\frac{{{}}}{{{}}}$".format(sign, pi_str, den)


def Azione_forzante(df, t_max, t_min):
    # AZIONE FORZANTE
    mask = (df['t'] >= t_min) & (df['t'] <= t_max)
    x = df.loc[mask, 't'].to_numpy()
    y = df.loc[mask, 'F'].to_numpy()

    fft_vals = np.fft.rfft(y - y.mean())
    freqs = np.fft.rfftfreq(x.size, d=(x[1] - x[0]))
    f0 = freqs[np.argmax(np.abs(fft_vals))]
    omega0_guess = 2 * np.pi * f0
    p0 = [
        (y.max() - y.min()) / 2,  # A
        omega0_guess,  # ω
        0.0,  # φ
        y.mean()  # C
    ]

    popt, pcov = curve_fit(seno_param, x, y, p0=p0)
    A_fit, omega_fit, phi_fit, C_fit = popt
    sigma = np.sqrt(np.diag(pcov))
    A_err, omega_err, phi_err, C_err = sigma

    print(f"azione forzante")
    print(f"A     = {abs(A_fit):.4f} ± {A_err:.4f}")
    print(f"ω     = {omega_fit:.4f} ± {omega_err:.4f} rad/s")
    print(f"v     = {(omega_fit / (2 * np.pi)):.4f}")
    # print(f"φ     = {phi_fit:.4f} ± {phi_err:.4f} rad")

    plt.figure(figsize=(10, 6))
    y_fit = seno_param(x, *popt)
    plt.scatter(x, y, color='red', s=30, label='picchi')
    plt.plot(x, y_fit, '-', lw=2, alpha=0.8, label="curve fit")

    plt.xlabel('tempo [s]', fontsize=14)
    plt.ylabel('ampiezza', fontsize=14)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    pdf_path_forzante = os.path.join(pdf_folder_forzante, f"{name}.pdf")
    plt.savefig(pdf_path_forzante, format="pdf", dpi=1000, bbox_inches="tight")
    if print_option != 0:
        plt.show()
    plt.close()

    frequency_fit = omega_fit / (2 * np.pi)
    frequency_fit_err = omega_err / (2 * np.pi)
    f.write(f"{name}\t{abs(A_fit)}\t{A_err}\t{omega_fit}\t{omega_err}\t{frequency_fit}\t{frequency_fit_err}\t{phi_fit}\t{phi_err}\t{C_fit}\t{C_err}\n")


def Risonanza_attiva(df, t_max, t_min):
    # FIT RISONANZA ATTIVA
    mask = (df['t'] >= t_min) & (df['t'] <= t_max)
    x = df.loc[mask, 't'].to_numpy()
    y = df.loc[mask, 'A'].to_numpy() * 2 * np.pi

    fft_vals = np.fft.rfft(y - y.mean())
    freqs = np.fft.rfftfreq(x.size, d=(x[1] - x[0]))
    f0 = freqs[np.argmax(np.abs(fft_vals))]
    omega0_guess = 2 * np.pi * f0
    p0 = [
        (y.max() - y.min()) / 2,  # A
        omega0_guess,  # ω
        0.1,  # φ
        y.mean()  # C
    ]
    popt, pcov = curve_fit(seno_param, x, y, p0=p0)
    A_fit, omega_fit, phi_fit, C_fit = popt
    sigma = np.sqrt(np.diag(pcov))
    A_err, omega_err, phi_err, C_err = sigma

    print(f"\nAnalisi file {name} mHz")
    print(f"risonanza attiva")
    print(f"A     = {abs(A_fit):.4f} ± {A_err:.4f}")
    print(f"ω     = {omega_fit:.4f} ± {omega_err:.4f} rad/s")
    print(f"v     = {(omega_fit / (2 * np.pi)):.4f}")
    # print(f"φ     = {phi_fit:.4f} ± {phi_err:.4f} rad")

    plt.figure(figsize=(14, 7))
    # x = x[1500:]
    # y = y[1500:]
    y_fit = seno_param(x, A_fit, omega_fit, phi_fit, C_fit)

    y_err = abs(y)*0.029
    mask = y_err != 0
    y_valid = y[mask]
    y_fit_valid = y_fit[mask]
    y_err_valid = y_err[mask]

    chi2 = np.sum(((y_valid - y_fit_valid) / y_err_valid)**2)
    ndof = len(y_valid)/10 - 4  # o -5 se hai 5 parametri
    chi2_reduced = chi2 / ndof

    r.write(f"{name}\t"
        f"{abs(A_fit)}\t{A_err}\t"
        f"{omega_fit}\t{omega_err}\t"
        f"{phi_fit}\t{phi_err}\t"
        f"{C_fit}\t{C_err}\t"
        f"{chi2_reduced}\n")

    plt.plot(x, y_fit, '-', lw=2, alpha=0.8, label="y = A sin(ωt + φ) + C")
    plt.errorbar(x, y, xerr=0, yerr=abs(y)*0.029, fmt='o', alpha=0.8, color='orange', label=f"{name} mHz")

    offset = x[0]
    ax = plt.gca()
    labels = ax.get_xticklabels()
    new_labels = []
    for label in labels:
        text = label.get_text()
        try:
            val = float(text)
            new_val = val - offset
            new_labels.append(f'{new_val:.0f}')
        except ValueError:
            # Se non è convertibile in float, mantieni il testo originale
            new_labels.append(text)
    ax.set_xticklabels(new_labels)
    plt.gca().yaxis.set_major_locator(MultipleLocator(base=np.pi/4))  # ogni pi/8
    plt.gca().yaxis.set_major_formatter(FuncFormatter(pi_fraction_formatter))

    plt.xlabel('tempo [s]', fontsize=24, labelpad=20)
    plt.ylabel('ampiezza [rad]', fontsize=24, labelpad= 20)
    plt.legend(fontsize=18, loc='right')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=28)
    plt.grid(True)
    plt.tight_layout()

    pdf_path_risonanza = os.path.join(pdf_folder_risonanza, f"{name}.pdf")
    plt.savefig(pdf_path_risonanza, format="pdf", dpi=1000, bbox_inches="tight")
    if print_option != 0:
        plt.show()
    plt.close()



def Oscillazione_smorzato(df, t_max, t_min, t_noise):
    # FIT OSCILLAZIONE SMORZATO
    mask = (df['t'] >= t_max) & (df['t'] <= t_noise)
    x_ass = df.loc[mask, 't'].to_numpy()
    t0 = x_ass[0]
    x = x_ass - t0
    y = df.loc[mask, 'A'].to_numpy() * 2 * np.pi

    # x = x[:235]
    # y = y[:235]

    if len(x) > 50:
        fft_vals = np.fft.rfft(y - y.mean())
        freqs = np.fft.rfftfreq(x.size, d=(x[1] - x[0]))
        f0 = freqs[np.argmax(np.abs(fft_vals))]
        p0 = [
            (y.max() - y.min()) / 2,  # A
            0.1,  # lamda
            2 * np.pi * int(name)/1000,  # ω
            0.0,  # φ
            y.mean()  # C
        ]

        popt, pcov = curve_fit(damped_sine, x, y, p0=p0)
        A_fit, lam_fit, omega_fit, phi_fit, C_fit = popt
        perr = np.sqrt(np.diag(pcov))

        print(f"oscillazioni smorzate")
        print(f"A     = {abs(A_fit):.4f} ± {perr[0]:.4f}")
        print(f"λ     = {lam_fit:.4f} ± {perr[1]:.4f}  [1/s]")
        print(f"ω     = {omega_fit:.4f} ± {perr[2]:.4f}  [rad/s]")
        print(f"φ     = {phi_fit:.4f} ± {perr[3]:.4f}  [rad]")
        print(f"C     = {C_fit:.4f} ± {perr[4]:.4f}  [rad]")

        y_fit = damped_sine(x, *popt)

        y_err = abs(y)*0.029
        mask = y_err != 0
        y_valid = y[mask]
        y_fit_valid = y_fit[mask]
        y_err_valid = y_err[mask]

        chi2 = np.sum(((y_valid - y_fit_valid) / y_err_valid)**2)
        ndof = len(y_valid)/10 - 5  # o -5 se hai 5 parametri
        chi2_reduced = chi2 / ndof

        s.write(f"{name}\t{abs(A_fit)}\t{perr[0]}\t{lam_fit}\t{perr[1]}\t{omega_fit}\t{perr[2]}\t{phi_fit}\t{perr[3]}\t{C_fit}\t{perr[4]}\t{chi2_reduced}\n")
        
        plt.figure(figsize=(14, 7))
        # plt.plot(x, y_fit, '-', lw=2, alpha=0.8, label="y = A * exp(-γt) * sin(ωt + φ) + C")
        plt.errorbar(x, y, xerr=0, yerr=y_err, fmt='o', alpha=0.8, color='orange', label=f"{name} mHz")

        plt.gca().yaxis.set_major_locator(MultipleLocator(base=np.pi/4))  # ogni pi/8
        plt.gca().yaxis.set_major_formatter(FuncFormatter(pi_fraction_formatter))

        plt.xlabel('tempo [s]', fontsize=24, labelpad=20)
        plt.ylabel('ampiezza [rad]', fontsize=24, labelpad=20)
        plt.legend(fontsize=18, loc='right')
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=28)
        plt.grid(True)
        plt.tight_layout()

        pdf_path_smorzamento = os.path.join(pdf_folder_smorzamento, f"{name}.pdf")
        plt.savefig(pdf_path_smorzamento, format="pdf", dpi=1000, bbox_inches="tight")
        if print_option != 0:
            plt.show()
        plt.close()

        # LINEARIZZA I PICCHI
        mask = (df['t'] >= t_max) & (df['t'] <= t_noise)
        x_ass = df.loc[mask, 't'].to_numpy()
        t0 = x_ass[0]
        x = x_ass - t0
        y = df.loc[mask, 'A'].to_numpy() * np.pi

        period = 2 * np.pi / omega_fit
        dt = x[1] - x[0]
        # vogliamo almeno l'80% del periodo tra due picchi
        min_distance = max(1, int(0.8 * period / dt))
        # e richiediamo anche una prominenza minima (es. 10% di A_fit)
        prom = abs(A_fit) * 0.05
        peaks, props = find_peaks(y, distance=min_distance, prominence=prom)
        x_peaks = x[peaks]
        # y_peaks = y[peaks]
        y_peaks = np.log(y[peaks] - C_fit)  # shift the peakes to be off-set to 0
        m, q = np.polyfit(x_peaks, y_peaks, 1)

        x_vals = np.linspace(min(x_peaks), max(x_peaks), 500)
        y_interp = exp_fit(x_vals, m, q)
        plt.figure(figsize=(14, 7))
        plt.errorbar(x, y, xerr=0, yerr=abs(y)*0.029, fmt='o', alpha=0.8, color='orange', label=f"{name} mHz")
        plt.plot(x_vals, y_interp, '-', color='blue', label=r'$y =± A e^{-\gamma t} \sin(\omega t + \phi)$')
        plt.plot(x_vals, -y_interp-0.05, '-', color='blue')
        plt.gca().yaxis.set_major_locator(MultipleLocator(base=np.pi/4))  # ogni pi/8
        plt.gca().yaxis.set_major_formatter(FuncFormatter(pi_fraction_formatter))
        plt.xlabel('tempo [s]', fontsize=24, labelpad=20)
        plt.ylabel('ampiezza [rad]', fontsize=24, labelpad=20)
        plt.legend(fontsize=18, loc='upper right')
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=28)
        plt.grid(True)
        plt.tight_layout()
        # plt.show()

        plt.figure(figsize=(10, 6))
        y_fit = q + m * x_peaks
        plt.scatter(x_peaks, y_peaks, color='red', s=30, label='picchi')
        plt.plot(x_peaks, y_fit, '-', lw=2, alpha=0.8, label="oscillazione smorzata")

        plt.xlabel('tempo [s]', fontsize=14)
        plt.ylabel('ampiezza', fontsize=14)
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.tight_layout()

        pdf_path_fitMassimi = os.path.join(pdf_folder_fitMassimi, f"{name}.pdf")
        plt.savefig(pdf_path_fitMassimi, format="pdf", dpi=1000, bbox_inches="tight")
        if print_option != 0 or mode == 0:
            plt.show()
        plt.close()

        residuals = np.abs((y_peaks - y_fit)/y_fit ) * np.pi
        plt.figure(figsize=(10, 6))
        plt.scatter(x_peaks, residuals, color='blue', s=40, label=f'scarto relativo - {name} mHz')
        plt.axhline(0, color='black', lw=1, linestyle='--')

        plt.xlabel('x_peaks', fontsize=14)
        plt.ylabel('modulo distanza relativa - % sul fit', fontsize=14, labelpad=20)
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.tight_layout()

        pdf_path_scarto_smorzamento = os.path.join(pdf_folder_scarto_smorzamento, f"{name}.pdf")
        plt.savefig(pdf_path_scarto_smorzamento, format="pdf", dpi=1000, bbox_inches="tight")
        if print_option != 0 or mode == 0:
            plt.show()
        plt.close()


def God_mode(df):
    # GOD MODE
    x = df['t']
    y = df['A']
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, marker='o', s=15, alpha=0.8, color='orange', label=f"{name} mHz")
    plt.show()

    nonzero_counts_from_i = np.cumsum((df['F'] != 0)[::-1])[::-1]
    zero_start_idx = np.where(nonzero_counts_from_i == 0)[0]
    if zero_start_idx.size > 0:
        idx_max = zero_start_idx[0]
        print(f"La funzione è sempre nulla a partire da indice {idx_max}, tempo t = {x[idx_max]:.3f} s")
    else:
        print("Non ho trovato un tratto tutto-zero.")

    idx_min = float(input("Inserisci t_min [s]: ")) * frequency

    Oscillazione_smorzato(df, idx_max/frequency, idx_min/frequency, x.max())
    idx_noise = float(input("Inserisci t_noise [s]: ")) * frequency + idx_max

    with open(t_range_path, "a") as f:
        f.write(f"{idx_min}\t{idx_max}\t{idx_noise}\n")



for i, fd in enumerate(glob.glob(data_path)):
    df = pd.read_csv(fd, sep='\t', header=None, names=['t', 'F', 'A'], skiprows=2)
    df[['t', 'F', 'A']] = df[['t', 'F', 'A']].apply(pd.to_numeric, errors='coerce')
    name = re.findall(r'\d+', os.path.splitext(os.path.basename(fd))[0])[0]  # togliere l'estensione .txt
    print(f"\n\n\t--- {name} -- \n")

    if mode != 0:
        with open(t_range_path) as t:
            idx_min, idx_max, idx_noise = map(float, t.read().splitlines()[i].split())
        t_min = idx_min / frequency
        t_max = idx_max / frequency
        t_noise = idx_noise / frequency

        Risonanza_attiva(df, t_max, t_min)
        Oscillazione_smorzato(df, t_max, t_min, t_noise)
        Azione_forzante(df, t_max, t_min)

    if mode == 0:
        God_mode(df)