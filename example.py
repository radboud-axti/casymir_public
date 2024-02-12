import argparse
import numpy as np
import csv
from importlib import resources
import casymir.casymir
import casymir.processes

np.seterr(divide='ignore', invalid='ignore')


def print_intro():
    print("\n")
    print("\t*********************************************************")
    print("\t*\t\t\t\t\t\t\t*")
    print("\t*\t\t\tCASYMIR\t\t\t\t*")
    print("\t*   Cascaded linear system model for x-ray detectors\t*")
    print("\t*\t\t\t\t\t\t\t*")
    print("\t*********************************************************\n")


def save_to_csv(data, csv_path):
    headers = ["Frequency (1/mm)", "MTF", "NNPS"]
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)
        for row in data:
            writer.writerow(row.tolist())


def run_model_dc(system, spec_name, kV, mAs, fit='Y'):
    sys = casymir.casymir.System(system)
    material = sys.detector["active_layer"]
    name = sys.system_id

    with resources.path("casymir.data.detectors", f'{material}.yaml') as yaml_file_path:
        material_path = str(yaml_file_path)

    det = casymir.casymir.Detector(sys.detector["type"], material_path, sys.detector)
    tube = casymir.casymir.Tube(sys.source)

    print("\nPCM for " + name + '\n')

    spec = casymir.casymir.Spectrum(name=spec_name, kV=kV, mAs=mAs, detector=det, tube=tube)

    sig = casymir.processes.initial_signal(det, spec)
    sig2 = casymir.processes.quantum_selection(det, spec, sig)
    sig3 = casymir.processes.parallel_gain_direct(det, spec, sig2)
    sig4 = casymir.processes.spread_direct(det, sig3)
    sig5 = casymir.processes.px_integration(det, sig4)

    if fit == 'Y':
        print("\n*********************************************************")
        print("Fit Results:\n")
        sig5.fit()

    results = np.array([sig5.freq, sig5.mtf, sig5.nnps])
    results = np.transpose(results)

    return results


def run_model_ic(system, spec_name, kV, mAs, fit='Y'):
    sys = casymir.casymir.System(system)
    material = sys.detector["active_layer"]
    name = sys.system_id

    with resources.path("casymir.data.detectors", f'{material}.yaml') as yaml_file_path:
        material_path = str(yaml_file_path)

    det = casymir.casymir.Detector(sys.detector["type"], material_path, sys.detector)
    tube = casymir.casymir.Tube(sys.source)

    print("\nPCM for " + name + '\n')

    spec = casymir.casymir.Spectrum(name=spec_name, kV=kV, mAs=mAs, detector=det, tube=tube)

    sig = casymir.processes.initial_signal(det, spec)
    sig2 = casymir.processes.quantum_selection(det, spec, sig)
    sig3 = casymir.processes.parallel_gain_indirect(det, spec, sig2)
    sig4 = casymir.processes.spread_indirect(det, sig3)
    sig5 = casymir.processes.optical_coupling(det, sig4)
    sig6 = casymir.processes.px_integration(det, sig5)

    if fit == 'Y':
        print("\n*********************************************************")
        print("Fit Results:\n")
        sig6.fit()

    results = np.array([sig6.freq, sig6.mtf, sig6.nnps])
    results = np.transpose(results)

    return results


def main():
    parser = argparse.ArgumentParser(description='CASYMIR standalone demo script')

    parser.add_argument('yaml_file', type=str, help='Path to System YAML file')
    parser.add_argument('spectrum_name', type=str, help='Name of the spectrum (e.g., LE for Low Energy)')
    parser.add_argument('kV', type=float, help='kV used to generate the spectrum')
    parser.add_argument('mAs', type=float, help='mAs used to generate the spectrum')
    parser.add_argument('--output_file', '-of', type=str, default='ouptut.xlsx',
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

    save_to_csv(results, output_path)

    print("\nProgram successfully ended, results written to: " + output_path + "\n")


if __name__ == "__main__":
    main()
