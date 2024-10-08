import argparse
import os
import numpy as np
import csv
from importlib import resources
import casymir.casymir
import casymir.processes

np.seterr(divide='ignore', invalid='ignore')


def positive_float(value):
    f_value = float(value)
    if f_value <= 0:
        raise argparse.ArgumentTypeError(f"Invalid value: {value}. It must be a positive number.")
    return f_value


def valid_file(value):
    if not os.path.isfile(value):
        raise argparse.ArgumentTypeError(f"File not found: {value}")
    return value


def print_intro():
    print("\n")
    print("\t*********************************************************")
    print("\t*                       CASYMIR                         *")
    print("\t*********************************************************")
    print("\t*     Generalized Cascaded Linear Model for X-ray       *")
    print("\t*        Detector Noise & Resolution Simulation         *")
    print("\t*********************************************************\n")


def save_to_csv(data, csv_path):
    headers = ["Frequency (1/mm)", "MTF", "NNPS (1/mm2)"]
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)
        for row in data:
            writer.writerow(row.tolist())


def run_simulation(system, spec_name, kV, mAs, detector_type, fit='Y'):
    sys = casymir.casymir.System(system)
    material = sys.detector["active_layer"]
    name = sys.system_id

    # Define the package where detector materials are stored
    detectors_package = 'casymir.data.detectors'
    material_filename = f'{material}.yaml'

    # Check if the material file exists
    available_materials = [file_name for file_name in resources.contents(detectors_package) if file_name.endswith('.yaml')]

    if material_filename not in available_materials:
        available_materials_list = ', '.join([mat[:-5] for mat in available_materials])  # Remove '.yaml' extension
        raise FileNotFoundError(
            f"Material file '{material_filename}' not found in '{detectors_package}'.\n"
            f"Available detector materials are: {available_materials_list}\n"
            "Please ensure the material exists and the name is correct."
        )

    # If the file exists, proceed to load it
    with resources.path(detectors_package, material_filename) as yaml_file_path:
        material_path = str(yaml_file_path)

    det = casymir.casymir.Detector(detector_type, material_path, sys.detector)
    tube = casymir.casymir.Tube(sys.source)
    spec = casymir.casymir.Spectrum(name=spec_name, kV=kV, mAs=mAs, detector=det, tube=tube)

    sig = casymir.processes.initial_signal(det, spec)
    sig = casymir.processes.quantum_selection(det, spec, sig)
    sig = casymir.processes.k_fluorescence(det, spec, sig)

    if detector_type == "direct":
        sig = casymir.processes.charge_trapping(det, sig)
    else:
        sig = casymir.processes.optical_blur(det, sig)
        sig = casymir.processes.optical_coupling(det, sig)

    sig = casymir.processes.px_integration(det, sig)

    if fit == 'Y':
        print("Fit Results:\n")
        sig.fit()

    results = np.array([sig.freq, sig.mtf, sig.nnps])
    results = np.transpose(results)

    return results


def run_standalone_simulation():
    parser = argparse.ArgumentParser(description='Run CASYMIR standalone')

    parser.add_argument('yaml_file', type=valid_file, help='Path to System YAML file')
    parser.add_argument('spectrum_name', type=str, help='Name of the spectrum (e.g., LE for Low Energy)')
    parser.add_argument('kV', type=positive_float, help='Tube voltage in kV')
    parser.add_argument('mAs', type=positive_float, help='Tube current-time product in mAs')
    parser.add_argument('--output_file', '-of', type=str, default='output.csv',
                        help='Output CSV file path')
    parser.add_argument('--print_fit', '-pf', choices=['Y', 'N'], default='N',
                        help='Print fit parameters for MTF and NNPS')

    args = parser.parse_args()

    print_fit_params = args.print_fit
    spectrum = args.spectrum_name
    kV = args.kV
    mAs = args.mAs
    system = args.yaml_file
    output_path = args.output_file

    sys = casymir.casymir.System(system)
    info = sys.description
    detector_type = sys.detector["type"]

    print_intro()
    print(f"Running cascaded linear system model for {info} \n")
    results = run_simulation(system, spectrum, kV, mAs, detector_type, fit=print_fit_params)

    save_to_csv(results, output_path)

    print("Program successfully ended, results written to: " + output_path + "\n")


if __name__ == "__main__":
    run_standalone_simulation()
