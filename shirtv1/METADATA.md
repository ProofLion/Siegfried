# METADATA.md

**Author**: Tae Ha "Jeff" Park

**Last updated**: 2022/11/29

This document describes the contents of the `metadata.json` files associated with each trajectory of SHIRT. The json file consists of the following data: `pSim`, `pMdl`, `sAbsState`, and `tRelState`. Before diving into these items, basic notations are first introduced.

## Notations

The table below lists the reference frames used in `metadata.json`.

| ref. frame | description |
| --- | --- |
| `eci` | Earth-centered inertial frame |
| `pri` | principal axes of each spacecraft |
| `bdy` | xyz-axes of the spacecraft CAD model |
| `cam` | camera axes, z-axis along boresight |

Some other details:

- `pri` origin coincides with the center of mass (`com`). `bdy` origin is the origin of a CAD model and may not coincide with `com`.
- `r_A2B_C` refers to a position vector from point `A` to `B` expressed in reference frame `C`. If `A` and `B` are reference frames, the vector goes from the origin of `A` to that of `B`.
- `q_A2B` refers to a quaternion representing a rotation aligning the frame `A` with the frame `B`. The quaternion is scalar-first, i.e., `q = [qw, qx, qy, qz]`.


## `pSim`

`pSim` contains information about the simulation parameters.

| name | unit | description |
| ---  | ---  |   ---      |
| `cam_step`  | s | time step between consecutive images |
| `num_images` | - | number of images |
| `sim_duration` | s | total simulation time |

## `pMdl`

`pMdl` contains information about the spacecraft model parameters. Specifically, it consists of two high-level items: `x1` for the servicer (Mango), and `x2` for the target (Tango). Below lists the properties that pertain to either or both spacecraft.

| name | size | unit | description |
| ---  | ---  | ---  |   ---       |
| `I_pri` | 3x3 | kg.m^2 | Inertia matrix about principal axes |
| `r_pri2bdy_pri` | 3x1 | m | Position vector from `com` to model origin expressed in `pri` |
| `q_pri2bdy` | 4x1 | - | Quaternion vector aligning `pri` with `bdy` |
| `r_pri2cam_pri` | 3x1 | m | Position vector from `com` to camera origin expressed in `pri` (servicer only) |
| `q_pri2cam` | 4x1 | - | Quaternion vector aligning `pri` with `cam` |

## `sAbsState`

`sAbsState` contains information about the servicer's measurements of its ground-truth absolute orbital states.

| name | size | unit | description |
| ---  | ---  | ---  |   ---       |
| `rv_eci2com_eci` | Nx6 | m, m/s | Position and velocity expressed in `eci`, arranged as `[r, v]` |
| `oe_osc_kep` | Nx6 | m, rad | Osculating Keplerian orbital elements, arranged as `[a, e, i, Omega, omega, M]` |
| `oe_osc_eq` | Nx6 | m, rad | Osculating equinoctial orbital elements, arranged as `[a, ex, ey, ix, iy, lambda]` |
| `q_eci2pri` | Nx4 | - | Quaternion representing the orientation of the principal axes w.r.t. `eci` |
| `w_pri` | Nx4 | rad/s | Angular velocity about its principal axes |
| `m_pri` | Nx3 | N.m | Control moment about its principal axes |

## `tRelState`

`tRelState` contains information about the target's ground-truth relative state w.r.t. the servicer.

| name | size | unit | description |
| ---  | ---  | ---  |   ---       |
| `rv_scom2tcom_spri` | Nx6 | m, m/s | Position and velocity of the target's `com` in the servicer's `pri` frame |
| `roe_osc_qns` | Nx6 | m | Osculating quasi-nonsingular relative orbital elements of the target scaled by the servicer's semi-major axis, arranged as `[ada, adl, adex, adey, adix, adiy]` |
| `roe_osc_ns` | Nx6 | m | Osculating nonsingular relative orbital elements of the target scaled by the servicer's semi-major axis, arranged as `[ada, adl, adex, adey, adix, adiy]` |
| `q_spri2tpri` | Nx4 | - | Quaternion representing the orientation of the target's principal axes w.r.t. the servicer's principal axes |
| `w_tpri2spri_tpri` | Nx3 | rad/s | Angular velocity of the servicer's principal axes w.r.t. the target's principal axes expressed in the target's principal axes |
