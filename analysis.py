"""
REUSEGRID AI -- Unified Battery Analysis Engine (backend/analysis.py)
Provides a single callable entrypoint: analyse_battery(battery)
Connecting:
  1. Layer 1: Safety Screening Engine
  2. Similarity Engine: Weighted Euclidean KNN against 500 benchmark battery cases
  3. Layer 2: Battery Health Intelligence Engine (Random Forest SoH)
  4. Layer 3: Component Fault Diagnosis & Repairability Assessment
  5. Layer 4 & 5: Energy Application Compatibility & Allocation Matching
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_current_dir)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from backend.safety import screen_battery
from backend.similarity_matcher import BatterySimilarityMatcher
from backend.health_model import BatteryHealthModel
from backend.diagnosis import diagnose_battery
from backend.matching import (
    calculate_compatibility,
    generate_explanation
)

# Global singleton matcher and health model to avoid re-training on every click
_matcher_instance = None
_health_model_instance = None
_energy_needs_df = None
_batteries_df = None


def get_data_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(project_dir, "data")
    bats_path = os.path.join(data_dir, "batteries.csv")
    if not os.path.exists(bats_path):
        bats_path = os.path.join(data_dir, "battery_dataset.csv")
    needs_path = os.path.join(data_dir, "energy_needs.csv")
    return bats_path, needs_path, project_dir


def load_resources():
    global _matcher_instance, _health_model_instance, _energy_needs_df, _batteries_df
    bats_path, needs_path, _ = get_data_paths()
    if _matcher_instance is None:
        _matcher_instance = BatterySimilarityMatcher(bats_path)
    if _health_model_instance is None:
        _health_model_instance = BatteryHealthModel()
    if _energy_needs_df is None:
        _energy_needs_df = pd.read_csv(needs_path)
    if _batteries_df is None:
        _batteries_df = pd.read_csv(bats_path)
    return _matcher_instance, _health_model_instance, _energy_needs_df, _batteries_df


def analyse_battery(battery_input):
    """
    Main callable function for the REUSEGRID AI platform.

    Parameters:
        battery_input (dict):
            Can include:
              - id (str): Battery ID (e.g. 'B001')
              - battery_type (str): Chemistry (e.g. 'Li-ion', 'LiFePO4', etc.)
              - voltage (float): Measured voltage in Volts
              - current (float): Measured current in Amperes
              - temperature (float): Measured temperature in deg C
              - temp_rise_rate (float): Thermal rise velocity in deg C/min
              - internal_resistance_mOhm (float): Internal resistance in mOhm
              - age_years (float): Estimated age in years
              - cycles (int): Charge/discharge cycles
              - capacity_wh (float): Measured capacity in Wh
              - physical_damage (int/str): Physical damage flag or description
              - component_issue (str): Observed component issue
              - swelling_observed (bool/str): Swelling flag or severity
              - leakage_observed (bool/str): Electrolyte leakage flag

    Returns:
        result (dict): Complete structured evaluation containing:
            - safety: {status: 'PASS'/'FAIL', reason: str, reasons: list}
            - health: float (SoH %)
            - remaining_energy_wh: float
            - diagnosis: str (diagnosed problem)
            - repairability: 'YES' / 'LIMITED' / 'NO'
            - repair_action: str
            - repair_cost_inr: int
            - retest_status: str
            - reuse_category: str (eligibility)
            - application: str (best matched application)
            - application_details: dict (score, quota, explanation)
            - best_case: dict (most similar case in dataset)
            - similar_cases: list of dicts (top 3-5 similar cases)
            - enriched_profile: dict
    """
    matcher, health_model, energy_needs, batteries_df = load_resources()

    # Standardize input dictionary
    bat_id = str(battery_input.get("id", battery_input.get("Battery_ID", "TEST_BAT")))
    bat_type_in = str(battery_input.get("battery_type", battery_input.get("Battery_Type", "Li-ion")))
    v = float(battery_input.get("voltage", battery_input.get("Measured_Voltage_V", 3.9)))
    i = float(battery_input.get("current", battery_input.get("Measured_Current_A", 1.0)))
    temp = float(battery_input.get("temperature", battery_input.get("Operating_Temperature_C", 28.0)))
    trise = float(battery_input.get("temp_rise_rate", battery_input.get("Temp_Rise_Rate_C_per_min", 0.2)))
    ir = float(battery_input.get("internal_resistance_mOhm", battery_input.get("Internal_Resistance_mOhm", 55.0)))
    age = float(battery_input.get("age_years", battery_input.get("Estimated_Age_Years", 2.5)))
    cycles = int(battery_input.get("cycles", battery_input.get("Historical_Cycles", 450)))
    cap_wh = float(battery_input.get("capacity_wh", battery_input.get("Measured_Remaining_Capacity_Wh", 35.0)))

    # Physical damage / component issue mapping
    raw_phys = battery_input.get("physical_damage", battery_input.get("Physical_Casing_Condition", 0))
    if isinstance(raw_phys, int):
        casing_desc = "Distorted / Swelling" if raw_phys == 1 else "Intact (Good)"
        phys_flag = raw_phys
    else:
        casing_desc = str(raw_phys)
        phys_flag = 0 if "Intact" in casing_desc or "No visible damage" in casing_desc else 1

    comp_issue = str(battery_input.get("component_issue", battery_input.get("Primary_Diagnosed_Issue", "None")))
    swelling = str(battery_input.get("swelling_observed", "None (0-1%)"))
    if swelling in [True, "True", "Yes", "Swelling"]:
        swelling = "Severe Swelling (>6%)"
    elif swelling in [False, "False", "No"]:
        swelling = "None (0-1%)"

    leakage = str(battery_input.get("leakage_observed", "None"))
    if leakage in [True, "True", "Yes", "Leakage"]:
        leakage = "Electrolyte Leakage / Breach"
    elif leakage in [False, "False", "No"]:
        leakage = "None"

    # Step 2: Similarity Matching against Dataset
    telemetry_for_match = {
        "channel_id": bat_id,
        "measured_voltage": v,
        "measured_current": i,
        "operating_temperature": temp,
        "temp_rise_rate": trise,
        "internal_resistance_mOhm": ir,
        "physical_casing": casing_desc,
        "swelling_observed": swelling,
        "leakage_observed": leakage,
        "source": "REUSEGRID_PLATFORM"
    }
    similarity_result = matcher.find_most_similar_case(telemetry_for_match, top_k=5)
    matched_case = similarity_result["enriched_battery"]
    top_matches = similarity_result["top_k_matches"]

    # Combine user inputs with inferred archetype fields
    # If user selected an explicit chemistry, preserve it; otherwise infer from match
    final_type = matched_case["Battery_Type"] if bat_type_in in ["Auto-Detect", "Other", "", "Li-ion"] else bat_type_in
    
    # Determine nominal voltage and pack config (support single-cell 1S as well as multi-cell packs)
    if "nominal_voltage" in battery_input:
        nom_v = float(battery_input["nominal_voltage"])
    elif v < 5.5:
        nom_v = 3.7
    else:
        nom_v = float(matched_case.get("Nominal_Voltage_V", 11.1))

    if "pack_configuration" in battery_input:
        pack_config = str(battery_input["pack_configuration"])
    elif v < 5.5:
        pack_config = "1S"
    else:
        pack_config = matched_case.get("Pack_Configuration", "3S")

    nom_wh = float(matched_case.get("Nominal_Energy_Wh", max(cap_wh, 40.0)))
    if v < 5.5 and cap_wh <= 25.0:
        nom_wh = max(cap_wh, 15.0)

    # Step 1: Safety Screening
    eval_row = {
        "Battery_ID": bat_id,
        "Operating_Temperature_C": temp,
        "Temp_Rise_Rate_C_per_min": trise,
        "Measured_Voltage_V": v,
        "Nominal_Voltage_V": nom_v,
        "Pack_Configuration": pack_config,
        "Internal_Resistance_mOhm": ir,
        "Swelling_Deformation": swelling,
        "Electrolyte_Leakage_Observation": leakage,
        "Physical_Casing_Condition": casing_desc,
        "physical_damage": phys_flag,
        "Historical_Cycles": cycles,
        "Estimated_Age_Years": age,
        "Primary_Diagnosed_Issue": comp_issue if comp_issue not in ["None", "Unknown", ""] else matched_case.get("Primary_Diagnosed_Issue", "None"),
        "Nominal_Energy_Wh": nom_wh,
        "Measured_Remaining_Capacity_Wh": cap_wh,
        "Battery_Type": final_type
    }

    safety_status, safety_reasons = screen_battery(eval_row)
    eval_row["safety_status"] = safety_status
    eval_row["safety_reason"] = "; ".join(safety_reasons)

    # Step 3: Health Intelligence
    if safety_status == "FAIL":
        predicted_soh = 0.0
        remaining_wh = 0.0
    else:
        predicted_soh = health_model.predict(eval_row)
        remaining_wh = round((predicted_soh / 100.0) * nom_wh, 1)

    eval_row["estimated_health"] = predicted_soh
    eval_row["remaining_energy_wh"] = remaining_wh

    # Step 4: Diagnosis & Repairability Assessment
    diag_res = diagnose_battery(eval_row)
    eval_row["diagnosis"] = diag_res["diagnosis"]
    eval_row["impact_level"] = diag_res["impact"]
    eval_row["repair_action"] = diag_res["action"]
    eval_row["repair_cost_inr"] = diag_res["cost"]
    eval_row["retest_status"] = diag_res["retest"]
    eval_row["effective_health"] = diag_res["effective_soh"]
    eval_row["eligibility"] = diag_res["eligibility"]
    eval_row["effective_capacity_wh"] = round((diag_res["effective_soh"] / 100.0) * nom_wh, 1) if safety_status == "PASS" else 0.0

    # Determine simple repairability indicator: YES / LIMITED / NO
    elig_str = diag_res["eligibility"]
    if safety_status == "FAIL" or "Unfit" in elig_str:
        repairability = "NO"
    elif "Post-Repair" in elig_str:
        repairability = "YES"
    elif "Limited Use" in elig_str:
        repairability = "LIMITED"
    else:
        repairability = "YES"  # Direct reuse - no repair needed

    # Step 5: Application Matching
    best_app_name = "None (Safety Rejected)" if safety_status == "FAIL" else "Secondary Micro-Reserve Buffer"
    best_app_score = 0.0
    best_app_row = None

    if safety_status == "PASS" and not "Unfit" in elig_str:
        # Evaluate compatibility with all available application profiles
        for _, app_row in energy_needs.iterrows():
            score = calculate_compatibility(eval_row, app_row)
            if score > best_app_score:
                best_app_score = score
                best_app_name = app_row["application"]
                best_app_row = app_row

    # Step 6: Explainable rationale
    if safety_status == "FAIL":
        why_text = f"REJECTED AT SAFETY GATE:\n- Grounds: {eval_row['safety_reason']}\n- Under REUSEGRID AI protocol, zero matching score can override a physical safety failure.\n- Routed directly to closed-loop recycling."
    elif best_app_row is not None:
        why_text = generate_explanation(eval_row, best_app_row, best_app_score)
    else:
        why_text = "Allocated to reserve capacity due to marginal health or lack of high-demand quota."

    # Format historical similar cases for presentation
    similar_cases_formatted = []
    for m in top_matches:
        # Lookup historical outcome from dataset
        match_id = m["battery_id"]
        hist_rows = batteries_df[batteries_df["Battery_ID"] == match_id]
        if not hist_rows.empty:
            h = hist_rows.iloc[0]
            reuse_cat = h.get("Pool_Eligibility_Status", "Eligible")
            rec_app = h.get("Optimal_Application_Match", "Community Solar Microgrid")
            soh_hist = h.get("State_of_Health_SoH_Percent", 80.0)
            wh_hist = h.get("Effective_Available_Energy_Wh", 45.0)
        else:
            reuse_cat = "Eligible"
            rec_app = "Community Solar Microgrid"
            soh_hist = 80.0
            wh_hist = 45.0

        similar_cases_formatted.append({
            "Rank": m["rank"],
            "Battery_ID": m["battery_id"],
            "Battery_Type": m["battery_type"],
            "Voltage": m["measured_v"],
            "Temp_C": m["measured_t"],
            "IR_mOhm": m["measured_ir"],
            "Cycles": m["historical_cycles"],
            "Health_SoH": f"{soh_hist}%",
            "Capacity_Wh": wh_hist,
            "Historical_Category": reuse_cat,
            "Recommended_Application": rec_app,
            "Similarity": f"{m['similarity_score_pct']}%",
            "Similarity_Val": m["similarity_score_pct"]
        })

    # Historical outcome of the single best match
    best_sim = similar_cases_formatted[0] if similar_cases_formatted else {}

    return {
        "id": bat_id,
        "safety": {
            "status": safety_status,
            "reason": eval_row["safety_reason"],
            "reasons": safety_reasons
        },
        "health": predicted_soh,
        "remaining_energy_wh": eval_row["effective_capacity_wh"],
        "diagnosis": diag_res["diagnosis"],
        "impact_level": diag_res["impact"],
        "repairability": repairability,
        "repair_action": diag_res["action"],
        "repair_cost_inr": diag_res["cost"],
        "retest_status": diag_res["retest"],
        "reuse_category": diag_res["eligibility"],
        "application": best_app_name,
        "application_score": round(best_app_score * 100, 1),
        "application_explanation": why_text,
        "best_case": {
            "id": best_sim.get("Battery_ID", "B001"),
            "similarity_percent": best_sim.get("Similarity_Val", 95.0),
            "reuse_category": best_sim.get("Historical_Category", "Eligible (Direct)"),
            "recommended_application": best_sim.get("Recommended_Application", "Community Solar Microgrid"),
            "voltage": best_sim.get("Voltage", 12.0),
            "cycles": best_sim.get("Cycles", 400),
            "health": best_sim.get("Health_SoH", "85%")
        },
        "similar_cases": similar_cases_formatted,
        "raw_inputs": {
            "voltage": v,
            "current": i,
            "temperature": temp,
            "age_years": age,
            "cycles": cycles,
            "capacity_wh": cap_wh,
            "ir_mohm": ir,
            "battery_type": final_type,
            "physical_casing": casing_desc,
            "swelling": swelling,
            "leakage": leakage
        }
    }


# Quick test
if __name__ == "__main__":
    test_battery = {
        "id": "B001",
        "battery_type": "Li-ion",
        "voltage": 3.93,
        "current": 1.05,
        "temperature": 30.2,
        "age_years": 3.0,
        "cycles": 570,
        "capacity_wh": 35.0,
        "physical_damage": 0,
        "component_issue": "Temperature Sensor"
    }
    res = analyse_battery(test_battery)
    print("=== ANALYSIS TEST RESULT ===")
    print("Safety:", res["safety"]["status"])
    print("Health:", res["health"], "%")
    print("Diagnosis:", res["diagnosis"])
    print("Repairability:", res["repairability"], f"(Cost: INR {res['repair_cost_inr']})")
    print("Reuse Category:", res["reuse_category"])
    print("Best Application:", res["application"], f"({res['application_score']}%)")
    print("Best Case Match:", res["best_case"]["id"], f"({res['best_case']['similarity_percent']}%)")
    print("Top 3 matches count:", len(res["similar_cases"]))
