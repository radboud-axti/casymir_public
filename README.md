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

- `--output_file, -of <output_file.xlsx>`:	Path to .xlsx to store the output of CASYMIR (standalone). The default is “output.xlsx”.
- `--print_fit, -pf <Y|N>`:	Print polynomial fit parameters for MTF and NNPS curves. Default function for the MTF is f(x)=1+ax+bx^2+cx^3+dx^4, and f(x)=m+ax+bx^2+cx^3+dx^4 for the NNPS.

When executed in standalone mode, CASYMIR creates an .xlsx file containing frequency vector (up until the detector’s Nyquist frequency), the MTF, and the NNPS.

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
