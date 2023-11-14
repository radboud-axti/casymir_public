import argparse
from scipy import integrate
import numpy as np
from openpyxl import Workbook

import casymir.casymir

np.seterr(divide='ignore', invalid='ignore')


def print_intro():
    print("\n")
    print("\t*********************************************************")
    print("\t*\t\t\t\t\t\t\t*")
    print("\t*\t\t\tCASYMIR\t\t\t\t*")
    print("\t*   Cascaded linear system model for x-ray detectors\t*")
    print("\t*\t\t\t\t\t\t\t*")
    print("\t*********************************************************\n")


def save_to_excel(data, excel_path):
    # Create a new workbook
    wb = Workbook()
    # Create a new sheet
    ws = wb.active
    # Write data to the sheet
    for row in data:
        ws.append(row.tolist())
    # Save the workbook to the specified path
    wb.save(excel_path)


def run_model_dc(system, spec_name, kV, mAs, fit='Y'):
    sys = casymir.casymir.System(system)
    material = sys.detector["active_layer"]
    name = sys.system_id

    det = casymir.casymir.Detector(sys.detector["type"], "casymir/data/" + sys.detector["active_layer"] + ".yaml")
    det.fill_from_dict(sys.detector)

    tube = casymir.casymir.Tube()
    tube.fill_from_dict(sys.source)

    print("\nPCM for " + name + '\n')

    spec = casymir.casymir.Spectrum(name=spec_name, kV=kV, mAs=mAs, detector=det, tube=tube)
    energy, spectrum = spec.get_incident_spec()

    # Get probability of K reabsorption (fk) and blurring due to remote absorption (Tk)
    fk = det.calculate_fk()
    f, tk = det.calculate_Tk()
    # Get quantum efficiency of detector
    QE = det.get_QE(energy)

    spec_norm = spectrum / integrate.simps(spectrum, energy)
    # Mean quantum efficiency:
    g1 = integrate.simps(QE * spec_norm, energy)
    # Stage 0: starting signal and its wiener spectrum is q0 = number of photons per unit area
    q0 = spec.get_fluence()
    sig = casymir.casymir.Signal(f, q0 * np.ones(np.size(f), dtype=np.float64), q0 * np.ones(np.size(f),
                                                                                             dtype=np.float64))
    # Stage 1: Selection by detector. Stochastic gain based on detector quantum efficiency, Bernoulli statistics
    sig.stochastic_gain(g1, np.sqrt(g1 * (1 - g1)), weight=1)
    gA, gB, gC, wA, wB, wC = casymir.casymir.parallel_gains_direct(det, energy, spectrum, fk, kV)
    sig2A = casymir.casymir.Signal(sig.freq, sig.signal, sig.wiener)
    sig2B = casymir.casymir.Signal(sig.freq, sig.signal, sig.wiener)
    sig2C = casymir.casymir.Signal(sig.freq, sig.signal, sig.wiener)
    # Path A: stochastic gain gA, Poisson statistics -> var_gA = gA
    # Path B: stochastic gain gB, Poisson statistics -> var_gB = gB
    sig2A.stochastic_gain(gA, gA ** (1 / 2), wA)
    sig2B.stochastic_gain(gB, gB ** (1 / 2), wB)
    # Probability of remote reabsorption is fk and probability of whole path C occurring is wC
    # Path C:
    sig2C.stochastic_blur(tk, wC * fk)
    sig2C.stochastic_gain(gC, gC ** (1 / 2), 1)
    # Cross wiener spectrum for B & C processes:
    wienerBC = 2 * wB * tk * fk * gB * gC * sig.wiener
    # Putting paths A, B and C together:
    sig2 = casymir.casymir.Signal(sig.freq, sig2A.signal + sig2B.signal + sig2C.signal,
                                  sig2A.wiener + sig2B.wiener + sig2C.wiener + wienerBC)
    # Stage 3: stochastic blurring due to charge redistribution in Selenium (Tb)
    tb = casymir.casymir.spread_direct(det, sig2)
    sig2.stochastic_blur(tb, 1)
    # Signal at stage 3, after blurring due to charge redistribution/scintillator blurring
    sig3 = sig2
    # Stage 4: deterministic blurring by pixel aperture function Ta
    ta = np.abs(np.sinc(det.px_size * det.ff * sig2.freq))
    sig3.deterministic_blur(ta, 1)
    # Integrated signal at stage 4.
    sig4 = casymir.casymir.Signal(sig3.freq, sig3.signal * det.pxa, sig3.wiener * (det.pxa ** 2))
    # MTF
    mtf = sig4.signal / sig4.signal[0]
    # Aliased NPS (up to fNY):
    sig4.resample()

    add_noise = det.add_noise

    nps = sig4.wiener[0:int(np.size(sig4.freq) / 2)] + ((add_noise ** 2) * det.pxa)
    f2 = sig4.freq[0:int(np.size(sig4.freq) / 2)]
    mtf = mtf[0:int(np.size(sig4.freq) / 2)]
    # Normalized NPS
    nnps = nps / (sig4.signal[0] ** 2)

    sig4.mtf = mtf
    sig4.nnps = nnps

    if fit == 'Y':
        print("\n*********************************************************")
        print("Fit Results:\n")
        sig4.fit()

    results = np.array([f2, mtf, nnps])
    results = np.transpose(results)

    return results


def run_model_ic(system, spec_name, kV, mAs, fit='Y'):
    sys = casymir.casymir.System(system)
    material = sys.detector["active_layer"]
    name = sys.system_id

    det = casymir.casymir.Detector(sys.detector["type"], "casymir/data/" + sys.detector["active_layer"] + ".yaml")
    det.fill_from_dict(sys.detector)

    tube = casymir.casymir.Tube()
    tube.fill_from_dict(sys.source)

    print("\nPCM for " + name)

    spec = casymir.casymir.Spectrum(name=spec_name, kV=kV, mAs=mAs, detector=det, tube=tube)
    energy, spectrum = spec.get_incident_spec()

    # Get probability of K reabsorption (fk) and blurring due to remote absorption (Tk)
    fk = det.calculate_fk()
    f, tk = det.calculate_Tk()
    # Get quantum efficiency of detector
    QE = det.get_QE(energy)

    spec_norm = spectrum / integrate.simps(spectrum, energy)
    # Mean quantum efficiency:
    g1 = integrate.simps(QE * spec_norm, energy)
    # Stage 0: starting signal and its wiener spectrum is q0 = number of photons per unit area
    q0 = spec.get_fluence()
    sig = casymir.casymir.Signal(f, q0 * np.ones(np.size(f), dtype=np.float64), q0 * np.ones(np.size(f),
                                                                                             dtype=np.float64))
    # Stage 1: Selection by detector. Stochastic gain based on detector quantum efficiency, Bernoulli statistics
    sig.stochastic_gain(g1, np.sqrt(g1 * (1 - g1)), weight=1)
    gA, gB, gC, wA, wB, wC = casymir.casymir.parallel_gains_indirect(det, energy, spectrum, fk, kV)
    sig2A = casymir.casymir.Signal(sig.freq, sig.signal, sig.wiener)
    sig2B = casymir.casymir.Signal(sig.freq, sig.signal, sig.wiener)
    sig2C = casymir.casymir.Signal(sig.freq, sig.signal, sig.wiener)
    # Path A: stochastic gain gA, Poisson statistics -> var_gA = gA
    # Path B: stochastic gain gB, Poisson statistics -> var_gB = gB
    sig2A.stochastic_gain(gA, gA ** (1 / 2), wA)
    sig2B.stochastic_gain(gB, gB ** (1 / 2), wB)
    # Probability of remote reabsorption is fk and probability of whole path C occurring is wC
    # Path C:
    sig2C.stochastic_blur(tk, wC * fk)
    sig2C.stochastic_gain(gC, gC ** (1 / 2), 1)
    # Cross wiener spectrum for B & C processes:
    wienerBC = 2 * wB * tk * fk * gB * gC * sig.wiener
    # Putting paths A, B and C together:
    sig2 = casymir.casymir.Signal(sig.freq, sig2A.signal + sig2B.signal + sig2C.signal,
                                  sig2A.wiener + sig2B.wiener + sig2C.wiener + wienerBC)

    # Stage 3: stochastic blurring by scintillator
    tb = casymir.casymir.spread_indirect(det, sig2)
    sig2.stochastic_blur(tb, 1)
    # Signal at stage 3, after blurring due to charge redistribution/scintillator blurring
    sig3 = sig2
    # Stage 4: optical coupling and deterministic blurring by pixel aperture function Ta
    g4 = det.material["ce"]
    ta = np.abs(np.sinc(det.px_size * det.ff * sig2.freq))
    sig3.stochastic_gain(g4, (g4 * (1 - g4)) ** (1 / 2), 1)
    sig3.deterministic_blur(ta, 1)
    # Integrated signal at stage 4.
    sig4 = casymir.casymir.Signal(sig3.freq, sig3.signal * det.pxa, sig3.wiener * (det.pxa ** 2))
    # MTF
    mtf = sig4.signal / sig4.signal[0]
    # Aliased NPS (up to fNY):
    sig4.resample()
    add_noise = det.add_noise

    nps = sig4.wiener[0:int(np.size(sig4.freq) / 2)] + ((add_noise ** 2) * det.pxa)
    f2 = sig4.freq[0:int(np.size(sig4.freq) / 2)]
    mtf = mtf[0:int(np.size(sig4.freq) / 2)]
    # Normalized NPS
    nnps = nps / (sig4.signal[0] ** 2)

    sig4.mtf = mtf
    sig4.nnps = nnps

    if fit == 'Y':
        print("\n*********************************************************")
        print("Fit Results:\n")
        sig4.fit()

    results = np.array([f2, mtf, nnps])
    results = np.transpose(results)

    return results


def main():
    parser = argparse.ArgumentParser(description='CASYMIR demo script')

    parser.add_argument('yaml_file', type=str, help='Path to System YAML file')
    parser.add_argument('spectrum_name', type=str, help='Name of the spectrum (e.g., LE for Low Energy)')
    parser.add_argument('kV', type=float, help='kV used to generate the spectrum')
    parser.add_argument('mAs', type=float, help='mAs used to generate the spectrum')
    parser.add_argument('--output_file', '-op', type=str, default='ouptut.xlsx',
                        help='Path to output file. Default is "output.xlsx".')

    parser.add_argument('--print_fit', '-pf', choices=['Y', 'N'], default='N', help='Print fit parameters')

    args = parser.parse_args()

    print_fit_params = args.print_fit
    spectrum = args.spectrum_name
    kV = args.kV
    mAs = args.mAs
    system = args.yaml_file
    output_path = args.output_file

    sys = casymir.casymir.System(system)

    print_intro()

    if sys.detector["type"] == "direct":
        results = run_model_dc(system, spectrum, kV, mAs, fit=print_fit_params)

    else:
        results = run_model_ic(system, spectrum, kV, mAs, fit=print_fit_params)

    save_to_excel(results, output_path)

    print("\nProgram successfully ended, results written to: " + output_path + "\n")


if __name__ == "__main__":
    main()
