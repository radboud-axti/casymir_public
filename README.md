# CASYMIR

## IMPORTANT NOTICE
This repository is currently in transition from CASYMIR v1 to CASYMIR v2.
The poster presented at IWBI 2026 describes the upcoming v2 implementation. The public v2 code, examples, and documentation are being prepared for release and will be added here after final cleanup and validation.
Expected public release: August 2026.

For the previously reported v1 implementation (projection domain model), see legacy-v1/.

CASYMIR is a generalized cascaded linear systems model for x-ray detector
resolution and noise propagation. It models the detector as a sequence of gain,
blur, sampling, and noise processes, and can now carry those signals from the
original 1D detector chain into 2D projection-domain and 3D reconstruction-domain
frequency models.

The implementation is described in:

Pacheco G, Pautasso JJ, Michielsen K, Sechopoulos I. Software Article: A
generalized cascaded linear system model implementation for x-ray detectors.
Medical Physics. 2025;52:e18079. https://doi.org/10.1002/mp.18079

## Installation

CASYMIR requires Python 3.10 or newer. Install it from the repository root with:

```bash
python -m pip install -e .
```

The core dependencies are:

- numpy
- scipy
- spekpy
- pyyaml
- tqdm
- mucoeff==1.0.0

`mucoeff` is the default attenuation-coefficient source. `xraydb` is no longer a
required dependency; install the optional extra only if you need the legacy
`mu_source="XRAYDB"` comparison path.

```bash
python -m pip install -e ".[xraydb]"
```

## Public System Examples

CASYMIR V2 provides four example systems:

- `example_bct.yaml`: original breast CT example based on a Koning bCT system.
- `example_dbt.yaml`: original DBT example based on the Siemens Mammomat
  Revelation.
- `example_cbct.yaml`: benchtop CBCT example.
- `example_novation_prototype.yaml`: Novation prototype DBT example.

The YAML files describe source and detector hardware. Acquisition choices such
as view count, angular span, reconstruction filter settings, and voxel sampling
are set in the calling script or API workflow.

## Examples

Example scripts live in `examples/` and can be run from the repository root:

```bash
python -m examples.cbct
python -m examples.novation_prototype
python -m examples.bct
python -m examples.dbt
```

The examples do not require plotting libraries. They save compressed `.npz`
outputs in `examples/outputs/` by default. Each example saves 3D
reconstruction-domain arrays, including the signal transfer, Wiener spectrum,
MTF, NNPS, frequency axes, projection angles, and run metadata.

## System YAML Format

Each system file contains a short identifier, a description, a detector block,
and a source block:

```yaml
system_id: example bct
description: Example bCT system based on a Koning bCT

detector:
  type: indirect
  active_layer: CsI
  px_size: 0.1518
  ff: 0.85
  thickness: 700
  trapping_depth: 0
  elems: 256
  add_noise: 100
  extra_materials: [(Carbon Fiber, 2.5), (Silicon Dioxide, 1)]

source:
  target_angle: 10
  target: W
  SID: 95
  filter: [(Be, 1.4), (Al, 1.514)]
  external_filter: [(Al, 10), (Air, 950)]
```

Detector fields:

- `type`: `direct` or `indirect`.
- `active_layer`: detector material YAML name in `casymir/data/detectors`.
- `px_size`: detector pixel pitch in mm.
- `ff`: pixel fill factor.
- `thickness`: active-layer thickness in micrometers.
- `trapping_depth`: direct-conversion charge collection depth in micrometers.
- `elems`: number of frequency samples.
- `add_noise`: additive electronic noise.
- `extra_materials`: detector cover materials as material/thickness pairs.

Source fields:

- `target_angle`: anode angle in degrees.
- `target`: target material, usually `W`, `Mo`, or `Rh`.
- `SID`: source-to-image distance in cm.
- `filter`: internal tube filtration.
- `external_filter`: filtration between tube and detector.

## 2D and 3D Model Blocks

The 3D model follows the same block style as the original detector cascade:

1. Run the 1D detector stages.
2. Lift the result to a 2D detector plane with `pixel_integration_2d`.
3. Apply 2D sampling, log normalization, and reconstruction filters.
4. Collect projection views in a `SignalStack`.
5. Map the stack into a 3D frequency volume.
6. Apply voxel sampling or diagnostic aliasing blocks when needed.

For CBCT cases where each view uses the same projection-domain signal,
`projection_stack_from_signal(signal_2d, angles_rad)` builds the stack directly.
For DBT, build the stack view by view when focal spot blur, beam obliquity, or
other angle-dependent terms are active.

Reconstructed 3D volumes use the CASYMIR convention:

- `x`: detector-u / transverse scan direction.
- `y`: reconstructed depth / rotation direction.
- `z`: detector-v / longitudinal direction.

### DBT Fourier Mapping

`map_dbt_stack_to_volume` supports two deposition modes:

```python
volume = map_dbt_stack_to_volume(stack, f_depth, splat=False)
```

`splat=False` performs pure nearest-bin deposition for each view. This is useful
when inspecting the discrete Fourier mapping directly.

```python
volume = map_dbt_stack_to_volume(
    stack,
    f_depth,
    splat=True,
    kernel="gaussian",
    kernel_width_mode="angular",
)
```

`splat=True` spreads each view with the selected kernel. This produces smoother,
more continuous reconstruction-domain volumes on practical grids. The available
kernels are `gaussian`, `triangular`, and `sinc`.

## Data Files

Detector material files are in `casymir/data/detectors`. They define elemental
composition, density, K-fluorescence parameters, conversion gain, and blur
parameters for active detector layers such as amorphous selenium and CsI.

User-defined filtration materials are in `casymir/data/materials`. Each YAML
file defines one or more SpekPy materials by chemical formula or fractional
elemental composition.
