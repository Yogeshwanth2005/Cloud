# Data Issue and Recommended Plan

## 1. Current Data Problem

The current project has **two different types of solar datasets**, and they do not contain the same variables.

This creates a mismatch between the data available for the fleet simulation and the variables required by the Phase 3 causal model.

---

## 2. Real PVDAQ Physical Data

The real PVDAQ data in:

```text
system_50/51_weather.parquet
```

contains physical variables such as:

```text
poa_irradiance
module_temp
ambient_temp
dc_power
ac_power
```

These variables can be used to study real physical relationships in a PV system.

For example:

```text
POA Irradiance
       |
       v
    DC Power
       |
       v
    AC Power
```

and:

```text
Ambient Temperature
       |
       v
Module Temperature
       |
       v
    DC Power
```

Therefore, the PVDAQ data is suitable for establishing and calibrating the **physical part of the causal model**.

---

## 3. The 20-Site Fleet Data

The Phase 2 dataset is:

```text
site_timeline.parquet
```

It represents the 20 synthetic/replayed solar sites.

Its relevant information is essentially:

```text
site_id
time
power_mw
cpu_rate_sum
```

It does **not** contain:

```text
poa_irradiance
module_temp
ambient_temp
dc_power
ac_power
```

The raw Integration Studies CSVs were also checked and contain only:

```text
LocalTime
Power(MW)
```

Therefore, there is no hidden irradiance or temperature dataset within that source that can simply be recovered.

---

## 4. Why This Is a Problem for Phase 3

The proposed Phase 3 causal graph needs physical variables such as:

```text
poa_irradiance
module_temp
ambient_temp
dc_power
ac_power
```

along with the cloud-side variable:

```text
cpu_rate_sum
```

However, the 20-site fleet currently provides only:

```text
power_mw
cpu_rate_sum
```

Therefore, Phase 3 cannot directly learn a physical causal graph such as:

```text
POA Irradiance
       |
       +----------> DC Power ------> AC Power
       |
       +----------> Module Temperature
                         |
                         v
                     DC Power
```

from the 20-site fleet data.

The problem is therefore **not simply a missing column**.

The Integration Studies data was never designed to contain those physical sensor measurements.

---

## 5. The Two Datasets Represent Different Things

### Real PVDAQ

```text
Real PV system
      |
      +-- Real irradiance
      +-- Real temperature
      +-- Real DC power
      +-- Real AC power
      |
      +-- 2011–2023
```

### Integration Studies / Fleet

```text
20 synthetic/replayed plants
      |
      +-- Power(MW)
      +-- 2006 simulation clock
      +-- CPU workload
```

The plants and timelines are different.

Therefore, we cannot simply take:

```text
system_51 irradiance
```

and attach it to:

```text
Arizona site power
```

and claim that the resulting irradiance is measured irradiance for the Arizona site.

---

# 6. Recommended Solution According to the Proposal

The recommended approach is a **two-tier design**.

## Tier 1 — Real Physical Causal Calibration

Use the real PVDAQ data to learn/calibrate the physical relationships.

The primary source should be:

```text
system_51
```

with particular emphasis on:

```text
2015–2023
```

where the irradiance/DC-power relationship is strong.

The purpose of this stage is to establish a reusable physical relationship.

Conceptually:

```text
Real PVDAQ
     |
     v
Physical Measurements
     |
     v
Physical Causal Relationships
     |
     v
Calibrated Physical Model
```

---

# 7. Tier 2 — 20-Site Fleet Simulation

Keep the existing 20-site fleet.

Do **not** remove the 20-site distributed-fleet experiment simply because the Integration Studies data lacks physical sensor variables.

The proposal is explicitly focused on:

> Active Causal Orchestration for Distributed Solar Fleets

and its evaluation includes public multi-plant datasets and a custom simulation layer for synthetic or replayed PV traces.

Therefore, retaining the fleet is important to the overall research framing.

---

# 8. Generate Physical Proxies for the Fleet

For each synthetic fleet site, use its existing:

```text
power_mw
capacity_mw
```

to calculate normalized PV output:

```text
normalized_power = power_mw / capacity_mw
```

This normalized power can be used as an **irradiance-like proxy**.

Conceptually:

```text
Fleet Power
     |
     v
Power / Capacity
     |
     v
Normalized PV Output
     |
     v
Irradiance-like Proxy
```

The physical variables that are required by the causal graph can then be generated through a documented synthetic physical model calibrated from the real PVDAQ data.

For example:

```text
Normalized Fleet Power
          |
          v
Irradiance-like Proxy
          |
          +----------------+
          |                |
          v                v
    Synthetic          Synthetic
  Module Temperature  DC Power
          |                |
          +-------+--------+
                  |
                  v
            Synthetic AC Power
```

These variables must be clearly identified as:

```text
derived / synthetic / proxy variables
```

and **not as real measurements**.

---

# 9. Important Disclosure

The paper should explicitly state that the 20-site fleet does not contain direct physical sensor measurements.

The methodology should distinguish between:

### Observed physical variables

From PVDAQ:

```text
poa_irradiance
module_temp
ambient_temp
dc_power
ac_power
```

and:

### Derived fleet variables

For the 20-site simulation:

```text
irradiance-like proxy
synthetic module temperature
synthetic DC power
synthetic AC power
```

This prevents a reviewer from assuming that the Integration Studies plants contain real irradiance and temperature sensors.

---

# 10. How This Fits the Proposal

The proposal already includes:

```text
Custom simulation layer for safe intervention effects
on synthetic or re-played PV traces.
```

Therefore, the fleet-side physical-variable generation can be treated as part of the simulation layer.

The proposal's main contribution remains:

```text
Active Causal Orchestration
```

rather than the creation of a new physical-weather dataset.

---

# 11. Recommended Treatment of system_50

A second, separate data issue exists in the real PVDAQ data.

The observed relationships are:

| System | Period | Correlation: Irradiance vs DC Power | Interpretation |
|---|---|---:|---|
| system_50 | 2011–2014 | 0.724 | Reasonably strong |
| system_50 | 2015–2023 | 0.121 | Major breakdown |
| system_51 | 2011–2014 | 0.282 | Weak, small data volume |
| system_51 | 2015–2023 | 0.961 | Very strong |

Therefore:

## Primary physical calibration

Use:

```text
system_51
2015–2023
```

as the main real-data physical calibration source.

## system_50

Use:

```text
system_50
2011–2014
```

if its earlier data is useful for additional validation/calibration.

Exclude:

```text
system_50
2015–2023
```

from physical causal calibration because the irradiance/DC-power relationship has broken down substantially.

---

# 12. Why Not Build a Generic Degradation Detector Yet?

A generic operational-validity detector could be developed to automatically identify periods where the physical relationship becomes invalid.

However, this is not necessary at the current stage.

There are currently only two systems and one clearly identified problematic window.

Therefore, a simple per-system exclusion is sufficient.

Conceptually:

```text
system_50
     |
     +-- 2011–2014 → retain
     |
     +-- 2015–2023 → exclude from physical calibration
```

This keeps the engineering effort focused on the main research contribution.

---

# 13. Final Proposed Architecture

The overall data flow should therefore be:

```text
                         REAL PVDAQ
                             |
                             v
                Physical Causal Calibration
                             |
                             v
                  Reusable Physical Model
                             |
                             |
             +---------------+---------------+
             |                               |
             v                               v
       20-Site Fleet                    Real PVDAQ
       Simulation                       Validation
             |
             v
      power_mw / capacity
             |
             v
     Irradiance-like Proxy
             |
             v
    Synthetic Physical State
             |
             +-------------------------+
             |                         |
             v                         v
       Physical Variables         CPU Workload
             |                         |
             +------------+------------+
                          |
                          v
            Active Causal Semantic
                    Event Graph
                          |
                          v
                Causal World Model
                          |
                          v
             Value-of-Information
                          |
                          v
             Intervention Selection
                          |
                          v
          Distributionally Robust
                  Optimizer
                          |
              +-----------+-----------+
              |                       |
              v                       v
       Resource Allocation      Intervention
                                  Selection
              |                       |
              +-----------+-----------+
                          |
                          v
                 Edge / Cloud Execution
                          |
                          v
                 Updated Causal Model
                          |
                          v
                     Next Cycle
```

---

# 14. Final Decision

The recommended approach is:

```text
NODE_SCHEMA issue → A1
Physical degradation issue → B3
```

In practical terms:

1. **Keep the 20-site fleet.**
2. **Do not search for missing irradiance/temperature columns in the Integration Studies data anymore.**
3. **Use PVDAQ to calibrate the real physical relationships.**
4. **Use system_51, especially 2015–2023, as the primary physical calibration source.**
5. **Exclude system_50 after 2015 from physical calibration.**
6. **Create synthetic/derived physical proxies for the 20 fleet sites.**
7. **Clearly label those variables as derived/synthetic rather than measured.**
8. **Use the resulting fleet environment for the Active Causal Orchestration experiments.**
9. **Keep the causal graph → Causal World Model → VOI → intervention → optimization closed loop as the central contribution.**

---

# 15. The Key Idea

The project does **not** need every dataset to contain every variable.

Instead, use the datasets for different roles:

```text
PVDAQ
   ↓
"Learn what physically happens in a PV system."

20-Site Fleet
   ↓
"Test whether the proposed Active Causal Orchestration
works across a distributed fleet."

Cloud Trace
   ↓
"Represent the edge/cloud workload."

Simulation Layer
   ↓
"Connect these components into one controlled
experimental environment."
```

This preserves the proposal's central idea while being honest about what is measured and what is simulated.