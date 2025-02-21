import numpy as np
from importlib import resources
import casymir.casymir
import casymir.processes
import casymir.parallel
import copy
import matplotlib.pyplot as plt

noise = 100
tfilt = [("Be", 1.4), ("Al", 1.514)]
efilt = [("Al", 10), ("Air", 950)]
dfilt = [("Carbon Fiber", 2.5), ("Silicon Dioxide", 1)]

sys = casymir.casymir.System("example_bct.yaml")
material = sys.detector["active_layer"]
detectors_package = 'casymir.data.detectors'
material_filename = f'{material}.yaml'

with resources.path(detectors_package, material_filename) as yaml_file_path:
    material_path = str(yaml_file_path)

detector = casymir.casymir.Detector("direct", material_path, {
    "type": "indirect",
    "active_layer": "CsI",
    "px_size": 0.1518,
    "ff": 0.85,
    "thickness": 700,
    "trapping_depth": 0,
    "elems": 256,
    "add_noise": noise
})

tube = casymir.casymir.Tube({
    'target_angle': 10,
    'target': 'W',
    'SID': 95,
    'filter': tfilt,
    'external_filter': efilt,
})

detector.extra_materials = dfilt
kV = 51.1
mAs = 80.56 * 0.0065

spec = casymir.casymir.Spectrum(name="spectrum", kV=kV, mAs=mAs, detector=detector, tube=tube)

# CASE 1: Full bCT model

# Create initial signal (source) and initial quanta absorption.
sig_n, q0_n, _ = casymir.processes.initial_signal(detector, spec)
sig_n, g1_n, _ = casymir.processes.quantum_selection(detector, spec, sig_n)

# Parallel path engine
# Create Paths (branches)
PathA = casymir.parallel.Path(processes=[casymir.processes.absorption])  # Direct absorption
PathB = casymir.parallel.Path(processes=[casymir.processes.local_k_absorption])  # Local K-Fluorescence
PathC = casymir.parallel.Path(processes=[casymir.processes.remote_k_absorption])  # Remote K-Fluorescence

# Create Nodes
Node1 = casymir.parallel.Node(node_type="Bernoulli")  # First split: Absorption vs. Fluorescence
Node2 = casymir.parallel.Node(node_type="Fork")       # Second split: Local vs. Remote Fluorescence Absorption

# Attach Paths to Nodes.
Node1.add(PathA, probability=1 - detector.material["xi"] * detector.material["omega"])  # Absorption
Node2.add(PathB, probability=1)  # Local Fluorescence (always occurs if fluorescence occurs)
Node2.add(PathC, probability=1)  # Remote Fluorescence (always occurs if fluorescence occurs)

# Attach second Node to parent Node. The probability of the fork split is equal to xi*omega
Node1.add(Node2, probability=detector.material["xi"] * detector.material["omega"])

# Apply parallel process goes through all Nodes and Paths, returning the combined signal with
# the cross-spectral terms taken into account.
comb_sig = casymir.parallel.apply_parallel_process(Node1, sig_n, detector, spec)

# Remaining processes: optical blur, coupling, and signal integration.
sig_nf, _, _ = casymir.processes.optical_blur(detector, comb_sig)
sig_nf, _, _ = casymir.processes.optical_coupling(detector, sig_nf)

# New: previous integration block was split into: quantum integration, aliasing (with the Dirac comb approach
# and model_output, which adds electronic noise to the Wiener spectrum and
sig_nf, _, _ = casymir.processes.q_integration(detector, sig_nf)
sig_nf = casymir.processes.aliasing(detector, sig_nf)
sig_nf = casymir.processes.model_output(detector, sig_nf)

# Case 2: Simple validation cascade consisting of two paths separated by a Fork:

# Path 1: stochastic gain
# Path 2: quantum selection, stochastic gain, and deterministic blur (filtering)

# We define some simple processes here, for the sake of this example and also for demonstrating
# that the logic works for processes beyond those defined in the processes.py module


def generic_gain(detector_o, spectrum, signal):
    ga = 10
    signal_2 = copy.copy(signal)
    signal_2.stochastic_gain(ga, np.sqrt(ga))
    gain = ga
    spread = np.ones(np.size(signal.freq))

    return signal_2, gain, spread


def generic_selection(detector_o, spectrum, signal):
    g1 = 0.7
    signal_2 = copy.copy(signal)
    signal_2.stochastic_gain(g1, np.sqrt(g1 * (1 - g1)))
    gain = g1
    spread = np.ones(np.size(signal.freq))

    return signal_2, gain, spread


def generic_blur(detector_o, spectrum, signal):
    # Using a pixel aperture function for this example.
    ta = np.abs(np.sinc(detector_o.px_size * detector_o.ff * signal.freq))
    signal_2 = copy.copy(signal)
    signal_2.deterministic_blur(ta)

    return signal_2, 1, ta


sig2, q0_2, _ = casymir.processes.initial_signal(detector, spec)

# Branching paths implementation using the Node/Path logic
Path_1 = casymir.parallel.Path(processes=[generic_gain], name="Path A")
Path_2 = casymir.parallel.Path(processes=[generic_selection, generic_gain, generic_blur], name="Path B")

Node_F = casymir.parallel.Node(node_type="Fork", name="Fork 1")

Node_F.add(Path_1, probability=1)
Node_F.add(Path_2, probability=1)

final_sig = casymir.parallel.apply_parallel_process(Node_F, sig2, detector, spec)

# Handle each path separately:
sigA, gA, tA = generic_gain(detector, spec, sig2)

sigB, gB, tB = generic_selection(detector, spec, sig2)
sigB2, gB2, tB2 = generic_gain(detector, spec, sigB)
sigB3, gB3, tB3 = generic_blur(detector, spec, sigB2)

# Analytic Wiener cross term
npsAB = q0_2 * gA * gB * gB2 * tB3 * np.ones_like(sigB.freq)

plt.plot(final_sig.freq, final_sig.wiener, "or", label="Generalized model", alpha=0.4)
plt.plot(final_sig.freq, sigA.wiener + sigB3.wiener + 2*npsAB, "--k", label="$W_C = W_A + W_B + 2W_{AB}$", alpha=1, linewidth=2)
plt.plot(sigA.freq, sigA.wiener, "--b", label="$W_A$", linewidth=2)
plt.plot(sigA.freq, sigB3.wiener, "--g", label="$W_B$", linewidth=2)
plt.plot(sigA.freq, npsAB, "--m", label="$W_{AB}$", linewidth=2)
plt.legend(loc="best")
plt.title("Wiener Spectrum")
plt.xlabel("Frequency (1/mm)")
plt.show()

