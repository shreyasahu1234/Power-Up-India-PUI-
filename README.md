[README.md](https://github.com/user-attachments/files/33074381/README.md)
# Power-Up-India-PUI-# 🔋 REUSEGRID AI — Intelligent Second-Life Battery Allocation Platform

> **Multi-Battery Health Intelligence, Modular Repair & Explainable Second-Life Grid Allocation**

---

## 🎯 Core Operating Principle

> **Safety First → Health Second → Repair/Capability Assessment Third → Energy-Need Analysis → Explainable Matching**

Under the REUSEGRID AI governance protocol, **a good matching score can never override a physical safety failure**. Safety is a strict, non-negotiable hard gate.

---

## 📁 Project Architecture

```text
reusegrid_ai/
│
├── app.py                          # Streamlit Web Platform UI & Dashboard
│
├── backend/                        # Modular Intelligent Cognitive Engines
│   ├── __init__.py                 # Package declaration
│   ├── analysis.py                 # Unified pipeline entrypoint: analyse_battery(battery)
│   ├── safety.py                   # Layer 1: Mandatory Safety Screening Hard Gate
│   ├── similarity.py               # Layer 1.5: Similarity Intelligence Wrapper
│   ├── similarity_matcher.py       # Weighted Euclidean KNN against 500 benchmark battery cases
│   ├── health.py                   # Layer 2: Health Model Wrapper
│   ├── health_model.py             # Random Forest Regressor (SoH % and usable energy)
│   ├── diagnosis.py                # Layer 3: Component Fault Diagnosis & Repair Economics
│   ├── matching.py                 # Layers 4 & 5: Energy Application Matching & OR-Tools ILP
│   └── hardware_bridge.py          # ESP32 USB Serial Bridge + Calibrated Emulation Fallback
│
├── data/                           # Verified Empirical Datasets
│   ├── batteries.csv               # 500-battery benchmark dataset (45 feature columns)
│   ├── battery_dataset.csv         # Direct alias of 500-battery dataset
│   ├── energy_needs.csv            # 9 second-life energy application profiles
│   ├── energy_needs.xlsx           # Excel version with application profiles & summary
│   └── live_hardware_test_results.csv # Export of live evaluations
│
├── hardware/                       # Physical Prototype Firmware
│   └── esp32_code.ino              # 3-channel INA219 (I2C) + DS18B20 (OneWire) telemetry code
│
├── requirements.txt                # Python dependencies
└── README.md                       # Complete platform documentation
```

---

## ⚡ Quickstart Guide

### 1. Install Required Dependencies
Ensure Python 3.10+ is installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Platform
```bash
streamlit run app.py
```
Streamlit will automatically open your default browser at:
```text
http://localhost:8501
```

---

## 🖥️ Platform Features

1. **Flexible Ingestion Modes**:
   - **Manual Parameter Input**: Enter custom voltage, current, temperature, cycles, and visual observations.
   - **Live Hardware Bridge (ESP32)**: Automatically detects connected ESP32 over USB Serial or runs calibrated multi-channel emulation for CH1 (Safe), CH2 (Sensor Fault), and CH3 (Thermal Hazard).
   - **Benchmark Archetypes (Demo Presets)**: Fast one-click loading of representative cases directly from the 500-battery dataset.

2. **5-Layer Intelligence Report**:
   - **Current Battery Condition**: 6-metric physical monitoring grid.
   - **Layer 1 Safety Gate**: Clear visual alert (`PASS` or `FAIL`) with detailed hazard grounds. Unsafe batteries are immediately locked out from matching.
   - **Layer 2 Health Intelligence**: Random Forest predicted State of Health (SoH %) and extracted usable energy in Wh.
   - **Layer 3 Component Diagnosis**: Detection of replaceable modular faults (temperature thermistors, BMS boards, connector oxidation, casing cracks) with cost estimates in INR and post-repair retest verification.
   - **Similarity Intelligence**: Weighted Euclidean nearest-neighbor ranking showing the top 5 closest historical cases in the 500-battery database with historical outcomes.
   - **Layers 4 & 5 Explainable Matching**: Google OR-Tools compatibility scoring and natural-language decision audit cards explaining exactly why the battery was allocated.

3. **Global 500-Battery Grid Optimizer**:
   - Simulates global Integer Linear Programming (OR-Tools CBC/SCIP) across all 500 batteries simultaneously.
   - Tracks fulfillment rates across all 9 second-life energy quotas.

---

## 🔬 Hardware Telemetry Stack

| Sensor / Component | Bus / Protocol | Target Parameter | Measurement Role |
|---|---|---|---|
| **ESP32 DevKit V1** | 240 MHz Dual-Core | System Master | Aggregates and streams JSON telemetry over USB Serial (115200 baud) |
| **3× INA219** | I2C (0x40, 0x41, 0x44) | Voltage & Current | High-side DC bus voltage (0-32V) and current (±3.2A) per channel |
| **3× DS18B20** | OneWire (GPIO 4) | Skin Temperature | Temperature (°C) and thermal rise velocity ($dT/dt$, °C/min) |
| **Safety Watchdog** | Firmware & Python | Hardware Interlock | Flags runaway risks if $T > 48^\circ\text{C}$ or $dT/dt > 1.8^\circ\text{C/min}$ |

---

## 📜 Mentor & Governance Compliance
- **No component frankenstein reconstruction**: Entire battery packs are evaluated and allocated to suitable applications.
- **Safety priority**: Unsafe batteries cannot be assigned to any application under any circumstance.
- **Explainability**: Every match produces an auditable rationale citing capacity, voltage, health, and repair pathways.
