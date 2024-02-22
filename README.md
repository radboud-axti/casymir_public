# CASYMIR

## Introduction

CASYMIR is a generalized cascaded linear model to describe the spatial resolution and noise propagation characteristics of x-ray detectors. It employs linear systems theory to model the different gain and blurring stages within the detector. The key feature of CASYMIR is its flexibility in terms of detector designs and system geometries.
Technical details of the implementation can be found in [references].

## 1. Requirements

CASYMIR has the following dependencies:
- numpy
- scipy
- spekpy
- xraydb
- pyYAML

With the exception of SpekPy, all dependencies can be installed from the PyPI repository. Ensure that SpekPy is installed before attempting to install the package.

### 1.1 SpekPy installation

Download SpekPy from: https://bitbucket.org/spekpy/spekpy_release/downloads/

Navigate to the SpekPy directory in the command window and install SpekPy using pip:


```
cd path/to/spekpy
python -m pip install .
```

For in-depth installation instructions and documentation for SpekPy, refer to the [SpekPy Wiki](https://bitbucket.org/spekpy/spekpy_release/wiki/Home#markdown-header-how-to-install-spekpy).

### 1.2. CASYMIR installation

Once SpekPy is installed, navigate to the CASYMIR root directory and install it along with its dependencies:

```
cd path/to/casymir
python -m pip install -e
```

This command installs CASYMIR in editable mode and automatically installs dependencies specified in the “setup.py” file.

## 2. Running CASYMR (standalone)
CASYMIR can be executed in standalone mode using the following command:

```
python example.py <system_file.yaml> <spectrum_name> <kV> <mAs> [options]
```

### Required inputs:

- `<system_file.yaml>`:	Path to the YAML file containing the system description (see sect. 3)
- `<spectrum_name>`:	Name of the spectrum
- `<kV>`:	kV of the x-ray tube
- `<mAs>`:	Current-time product of the x-ray tube

### Optional inputs:

- `--output_file, -of <output_file.csv>`:	Path to CSV to store the output of CASYMIR (standalone). The default is “output.csv”.
- `--print_fit, -pf <Y|N>`:	Print polynomial fit parameters for MTF and NNPS curves. Default function for the MTF is f(x)=1+ax+bx^2+cx^3+dx^4, and f(x)=m+ax+bx^2+cx^3+dx^4 for the NNPS.

When executed in standalone mode, CASYMIR creates an CSV file containing frequency vector (up until the detector’s Nyquist frequency), the MTF, and the NNPS. The following command saves the results corresponding to the provided DBT system example for a representative mammography/DBT spectrum. 

```
python example.py example_dbt.yaml ex_spectrum 28 50 -of example_dbt.csv -pf Y
```

## 3. System files
Input to CASYMIR (standalone) is via YAML system files, which define all parameters of the x-ray source and detector.  The first two key-value pairs are used to store the system’s name and description, and these are followed by dictionaries containing the detector and source parameters.

## 3.1 An example
An example is given below. This system file corresponds to a Koning breast CT (bCT) system.
```
system_id: example bct
description: Test bCT system

detector:
  type: indirect
  active_layer: CsI
  px_size: 0.1518
  ff: 0.83
  thick: 700
  layer: 0
  elems: 256
  add_noise: 100
  extra_materials: [(Al, 0.05), (Carbon Fiber, 2.5), (Silica, 1)]

source:
  target_angle: 10
  target: W
  SID: 95
  filter: [(Be, 1.4), (Al, 1.42), (Al, 0.044)]
  external_filter: [(Al, 10), (Air, 950)]
```
Explanations of each of the detector and source parameters are given in the following subsections

## 3.1.1 Detector dictionary
The detector dictionary contains the following key-value pairs:
- `type`: detector type (direct or indirect). This defines the default gain and blurring stages.
- `active_layer`: material of the detector’s conversion layer. The name (value) will be used to reference a material YAML file (see section 4).
- `px_size`: pixel pitch in millimeters.
- `ff`: pixel fill factor. Value goes from 0 to 1.
- `thick`: active layer thickness in micrometers.
- `layer`: (only for direct conversion detectors) thickness (in micrometers) of the charge collection layer within the semiconductor.
- `elems`: number of detector elements.
- `add_noise`: additive electronic noise, expressed in charge quanta per area.
- `extra_materials`: detector cover materials. The value of this key is a list with the following format, with all thicknesses expressed in millimeters:
[(Material1, thickness1), (Material2, thickness2), … , (MaterialN, thicknessN)]

The available material list can be accessed via SpekPy’s  show_matls() function. In the event that materials not included in SpekPy are needed, these can be manually added to the mat_dictionary dictionary within the Spectrum class definition.

## 3.1.2 Source dictionary
This dictionary contains all the parameters of the x-ray source used by the model, namely:
- `target_angle`: target (anode) angle in degrees.
- `target`: target material. Supported materials are W (tungsten), Mo (molybdenum) or Rh (rhodium).
- `SID`: source to image distance in centimeters.
- `filter`: internal tube filtration, including the x-ray tube window. Follows the same format as the extra_materials list.
- `external_filter`: external filtration. Every material in between the x-ray tube and the detector should be listed here, for instance aluminum or PMMA to simulate breast attenuation. Follows the same format as the extra_materials list.


## 4. Data files

### 4.1 Detectors 

Detector material files are located in the “data/detectors” folder. These files define the characteristics of commonly used active layer materials, such as amorphous selenium (aSe) for direct conversion detectors and cesium iodide (CsI) for scintillator detectors.
The parallel cascaded model implementation uses the following information about the detector materials:

- `Components`: elemental composition, expressed in fractional weights.
-	`density`: physical density, expressed in g/cm3
-	`omega`: k-fluorescent yield.
-	`xi`: probability of a k-shell photoelectric interaction.
-	`ek`: K-edge of the material, in keV.
-	`kenergy`: average energy of k-emissions, in keV.
-	`w`: energy required to create an electron-hole pair, in keV. Exclusive to semiconductors.
-	`m0`: optical photons produced per keV of deposited energy. Exclusive to scintillators.
-	`ce`: coupling efficiency. Exclusive to scintillators.
-	`pf`: packing fraction of the material.
-	`mu_mass_abs`: photoelectric mass attenuation coefficient of the material at kenergy, in cm2/g.
-	`mu_mass_tot`: total mass attenuation coefficient of the material at kenergy, in cm2/g.
-	`spread_type`: type of function describing the spread of the quanta after the parallel paths. For scintillator detectors, the “Optical” tag is used; for semiconductors, “Charge redistribution”.
-	`spread_coeff`: value of H in the function T(f)=1⁄(1+Hf^2+H^2 f^3), which describes the spread of optical quanta in scintillator detectors.

Example YAML file for CsI:

```
Components:
  1: Cs, 0.51155
  2: I, 0.48845

density: 4.51
omega: 0.87
xi: 0.83
ek: 35
kenergy: 30
pf: 0.72
ce: 0.59
m0: 55.6

mu_mass_abs: 7.907025736
mu_mass_tot: 8.733446589

spread_type: "Optical"
spread_coeff: 0.15
```

### 4.2 Materials

User-defined materials are stored in the “data/materials” folder. These YAML files contain the necessary information to create SpekPy materials to be used for filtering. Materials can be defined in two ways: by chemical formula, or by fractional weights.
Each YAML file should only contain one material definition, enclosed in a `material` block. Examples for each material definition are given below:

```
material:
  - name: Mg-HA
    density: 2.8
    formula: Mg10P6O24O2H2
    comment: Magnesium-substituted Hydroxyapatite, defined by chemical formula
```

```
material:
  - name: Type I MC
    density: 2.2
    composition:
      20: 0.244188
      6: 0.146358
      8: 0.584889
      1: 0.024565
    comment: Type I breast microcalcification (Calcium Oxalate), defined by material weights
```

The densities are expressed in g/cm3, and the fractional weights are expressed as atomic number – weight pairs.
