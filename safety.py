"""
REUSEGRID AI — Layer 1: Safety Screening Engine
Mandatory hard gate: zero matching score can ever override a safety failure.
Evaluates physical, thermal, and electrochemical safety thresholds.
"""

import pandas as pd


def screen_battery(row):
    """
    Evaluates thermal, voltage, impedance, and physical hazards.
    Supports both rich benchmark columns and lightweight prototype columns.

    Returns:
        status (str): "PASS" or "FAIL"
        reasons (list): List of detected anomaly strings
    """
    reasons = []

    # 1. Thermal checks: Skin temp and temperature velocity
    temp = float(row.get("Operating_Temperature_C", row.get("temperature", 25.0)))
    temp_rise = float(row.get("Temp_Rise_Rate_C_per_min", 0.2))

    if temp > 48.0 or temp_rise > 1.8:
        reasons.append(f"Thermal instability / Runaway risk ({temp:.1f} deg C, {temp_rise:.2f} deg C/min)")

    # 2. Voltage collapse / deep discharge / cell inversion hazard
    v_meas = float(row.get("Measured_Voltage_V", row.get("voltage", 3.8)))
    v_nom = float(row.get("Nominal_Voltage_V", 3.7))
    pack_config = str(row.get("Pack_Configuration", "1S"))
    cells = 1
    if "S" in pack_config:
        try:
            cells = int(pack_config.replace("S", ""))
        except ValueError:
            cells = 1

    # Check cell-level under-voltage (< 2.5V/cell is destructive for Li-ion/NMC)
    v_per_cell = v_meas / max(1, cells)
    if v_per_cell < 2.5 or v_meas < (v_nom * 0.70):
        reasons.append(f"Severe under-voltage / Dendritic short hazard ({v_meas:.2f}V, {v_per_cell:.2f}V/cell vs nom {v_nom:.1f}V)")

    # 3. Dynamic internal resistance surge (micro-short / internal degradation)
    ir = float(row.get("Internal_Resistance_mOhm", 40.0))
    if ir > 220.0:
        reasons.append(f"Excessive internal impedance / Micro-short risk ({ir:.1f} mOhm)")

    # 4. Mechanical swelling / delamination
    swelling = str(row.get("Swelling_Deformation", "None"))
    if "Severe" in swelling or ">" in swelling or "5%" in swelling or "6%" in swelling:
        reasons.append(f"Mechanical swelling / Delamination hazard ({swelling})")

    # 5. Electrolyte leakage / seal breach
    leakage = str(row.get("Electrolyte_Leakage_Observation", "None"))
    if "Active" in leakage or "Leakage" in leakage or "Breach" in leakage:
        reasons.append("Electrolyte leakage / Chemical seal breach detected")

    # 6. Physical damage flag
    casing = str(row.get("Physical_Casing_Condition", "Intact"))
    damage_flag = int(row.get("physical_damage", 1 if ("Distorted" in casing or "Puffed" in casing or "Punctured" in casing) else 0))
    if damage_flag == 1 and not any("swelling" in r.lower() or "puncture" in r.lower() for r in reasons):
        reasons.append(f"Severe physical casing breach / Puncture hazard ({casing})")

    if reasons:
        return "FAIL", reasons

    return "PASS", ["Passed all mandatory safety screening thresholds"]


def apply_safety_screening(df):
    """
    Applies safety screening across the entire battery pool dataframe.
    """
    results = []
    reasons_list = []
    for _, row in df.iterrows():
        status, reasons = screen_battery(row)
        results.append(status)
        reasons_list.append("; ".join(reasons))

    df_out = df.copy()
    df_out["safety_status"] = results
    df_out["safety_reason"] = reasons_list
    return df_out
