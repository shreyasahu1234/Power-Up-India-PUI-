"""
REUSEGRID AI — Modular Pipeline Orchestrator with Hardware-to-Software Bridge
Ingests physical/emulated battery telemetry from ESP32, matches it with the closest case
in the 500-battery dataset, enriches its categories, and runs the full 5-layer allocation pipeline.
"""

import os
import sys
import pandas as pd

from backend.hardware_bridge import HardwareBridge
from backend.similarity_matcher import BatterySimilarityMatcher
from backend.safety import apply_safety_screening
from backend.health_model import apply_health_model, BatteryHealthModel
from backend.diagnosis import apply_diagnosis
from backend.matching import (
    build_compatibility_matrix,
    run_multi_battery_optimizer,
    generate_explanation
)


def load_dataset():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(project_dir, "data")

    bats_path = os.path.join(data_dir, "batteries.csv")
    needs_path = os.path.join(data_dir, "energy_needs.csv")

    batteries = pd.read_csv(bats_path)
    energy_needs = pd.read_csv(needs_path)
    return batteries, energy_needs, project_dir


def run_pipeline(chosen_channel=None):
    print("=" * 80)
    print("  REUSEGRID AI -- HARDWARE-TO-SOFTWARE INTELLIGENT ALLOCATION PIPELINE")
    print("=" * 80)

    # 1. Hardware Bridge Acquisition
    bridge = HardwareBridge()
    matcher = BatterySimilarityMatcher()
    batteries_df, energy_needs, project_dir = load_dataset()

    target_channel = chosen_channel if chosen_channel else "CH1"
    print(f"\n[Hardware Bridge Status] Mode: {bridge.mode}")
    print(f"[Hardware Bridge] Polling physical/emulated telemetry for chosen channel: {target_channel}...")

    live_telemetry = bridge.read_live_telemetry(target_channel)
    print(f"  * Voltage:             {live_telemetry['measured_voltage']} V")
    print(f"  * Current:             {live_telemetry['measured_current']} A")
    print(f"  * Temperature:         {live_telemetry['operating_temperature']} deg C")
    print(f"  * Thermal Velocity:    {live_telemetry['temp_rise_rate']} deg C/min")
    print(f"  * Internal Resistance: {live_telemetry['internal_resistance_mOhm']} mOhm")
    print(f"  * Hardware Source:     {live_telemetry.get('source', 'ESP32')}")

    # 2. Case-Based Reasoning: Similarity Matching against 500-battery Dataset
    print(f"\n[Similarity Engine] Matching {target_channel} against 500 benchmark battery cases...")
    similarity_result = matcher.find_most_similar_case(live_telemetry, top_k=3)
    enriched_battery = similarity_result["enriched_battery"]

    print(f"  [+] Best Dataset Match:    Battery ID '{similarity_result['best_match_id']}'")
    print(f"  [+] Similarity Score:      {similarity_result['similarity_score_pct']}%")
    print(f"  [+] Inferred Chemistry:    {enriched_battery['Battery_Type']} ({enriched_battery['Pack_Configuration']})")
    print(f"  [+] Inferred History:      {enriched_battery['Historical_Cycles']} cycles (~{enriched_battery['Estimated_Age_Years']} years)")
    print(f"  [+] Inferred Defect State: {enriched_battery['Primary_Diagnosed_Issue']}")

    print("\n  Top 3 Nearest Historical Archetypes in Dataset:")
    for n in similarity_result["top_k_matches"]:
        print(f"    - Rank {n['rank']}: ID {n['battery_id']} ({n['similarity_score_pct']}% match) | {n['battery_type']} | {n['historical_cycles']} cyc | {n['measured_v']}V")

    # 3. Insert / Prioritize the chosen hardware battery at the top of the pool
    live_df = pd.DataFrame([enriched_battery])
    full_pool_df = pd.concat([live_df, batteries_df], ignore_index=True)

    # 4. Layer 1: Safety Screening
    print("\n" + "-" * 80)
    print("[Layer 1] Mandatory Safety Screening Hard Gate")
    print("-" * 80)
    full_pool_df = apply_safety_screening(full_pool_df)

    live_safety = full_pool_df.iloc[0]["safety_status"]
    live_reason = full_pool_df.iloc[0]["safety_reason"]
    print(f"  Chosen Battery ({target_channel}) Safety Result: [{live_safety}]")
    print(f"  Reasoning: {live_reason}")

    # 5. Layer 2: Health Intelligence (Random Forest)
    print("\n" + "-" * 80)
    print("[Layer 2] Battery Health Intelligence Engine")
    print("-" * 80)
    health_model = BatteryHealthModel()
    full_pool_df = apply_health_model(full_pool_df, health_model)

    live_soh = full_pool_df.iloc[0]["estimated_health"]
    live_wh = full_pool_df.iloc[0]["remaining_energy_wh"]
    print(f"  Chosen Battery ({target_channel}) Predicted SoH: {live_soh}%")
    print(f"  Usable Extracted Energy: {live_wh} Wh")

    # 6. Layer 3: Component Diagnosis, Repairability & Retest Gate
    print("\n" + "-" * 80)
    print("[Layer 3] Component Fault Diagnosis & Repairability Assessment")
    print("-" * 80)
    full_pool_df = apply_diagnosis(full_pool_df)

    live_diag = full_pool_df.iloc[0]["diagnosis"]
    live_action = full_pool_df.iloc[0]["repair_action"]
    live_cost = full_pool_df.iloc[0]["repair_cost_inr"]
    live_retest = full_pool_df.iloc[0]["retest_status"]
    live_elig = full_pool_df.iloc[0]["eligibility"]

    print(f"  Diagnosed Component Issue: {live_diag}")
    print(f"  Action Pathway:            {live_action} (Est Cost: INR {live_cost})")
    print(f"  Post-Repair Retest Gate:   {live_retest}")
    print(f"  Pool Eligibility Status:   {live_elig}")

    # 7. Layers 4 & 5: Energy Needs Matching & Global OR-Tools Optimization
    print("\n" + "-" * 80)
    print("[Layers 4 & 5] Multi-Battery Matching & Google OR-Tools Global Allocation")
    print("-" * 80)

    eligible = full_pool_df[full_pool_df["eligibility"].str.startswith("Eligible")].copy()
    compat_matrix = build_compatibility_matrix(eligible, energy_needs)

    allocations, total_utility = run_multi_battery_optimizer(eligible, energy_needs, compat_matrix)
    print(f"  Optimization Solver: Optimal. Total System Utility: {total_utility:.3f}")

    # Find the allocation for our chosen battery
    chosen_id = enriched_battery["Battery_ID"]
    chosen_alloc = next((a for a in allocations if a["battery_id"] == chosen_id), None)

    print("\n" + "=" * 80)
    print(f"  FINAL ALLOCATION & EXPLAINABLE AUDIT FOR CHOSEN BATTERY ({target_channel})")
    print("=" * 80)

    if chosen_alloc:
        matched_app_name = chosen_alloc["application"]
        match_score_pct = int(chosen_alloc["score"] * 100)
        app_row = energy_needs[energy_needs["application"] == matched_app_name].iloc[0]

        print(f"\n[*] Chosen Physical Battery ({target_channel}) ----> {matched_app_name}")
        print(f"   System Compatibility Score: {match_score_pct}%\n")
        print("   EXPLAINABLE DECISION AUDIT CARD (Why This Match?):")
        explanation = generate_explanation(full_pool_df.iloc[0], app_row, chosen_alloc["score"])
        print(f"   {explanation}")
    else:
        if live_safety == "FAIL":
            print(f"\n[X] Chosen Battery ({target_channel}) REJECTED AT SAFETY GATE:")
            print(f"   Grounds: {live_reason}")
            print(f"   Under REUSEGRID AI protocol, zero matching score can override a physical safety failure.")
            print(f"   Status: Routed directly to closed-loop recycling.")
        else:
            print(f"\n[!] Chosen Battery ({target_channel}) placed in Reserve / Secondary Micro-Buffer Pool.")

    # Save benchmark run results
    out_csv = os.path.join(project_dir, "data", "live_hardware_test_results.csv")
    full_pool_df.to_csv(out_csv, index=False)
    print(f"\n[+] Enriched battery results saved to: {out_csv}")
    print("=" * 80)


def main():
    args = [a.upper() for a in sys.argv[1:]]

    # --SAMPLE mode: quick 6-battery sanity check (no hardware bridge)
    if "--SAMPLE" in args:
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir)
        sample_path = os.path.join(project_dir, "data", "batteries_sample.csv")
        batteries_s = pd.read_csv(sample_path)
        energy_needs_s = pd.read_csv(os.path.join(project_dir, "data", "energy_needs.csv"))
        from backend.safety import apply_safety_screening
        from backend.health_model import apply_health_model, BatteryHealthModel
        from backend.diagnosis import apply_diagnosis
        from backend.matching import build_compatibility_matrix, run_multi_battery_optimizer
        batteries_s = apply_safety_screening(batteries_s)
        batteries_s = apply_health_model(batteries_s, BatteryHealthModel())
        batteries_s = apply_diagnosis(batteries_s)
        eligible_s = batteries_s[batteries_s["eligibility"].str.startswith("Eligible")].copy()
        compat_s = build_compatibility_matrix(eligible_s, energy_needs_s)
        allocs_s, util_s = run_multi_battery_optimizer(eligible_s, energy_needs_s, compat_s)
        print("=== SAMPLE RUN ===")
        print(batteries_s[["Battery_ID", "safety_status", "estimated_health", "eligibility"]].to_string(index=False))
        print(f"\nOR-Tools Total Utility: {util_s:.3f}")
        for a in allocs_s:
            print(f"  {a['battery_id']:12s} --> {a['application']} (score {a['score']:.2f})")
        return

    # --ALL mode: sequentially run all 3 channels
    if "--ALL" in args:
        for ch in ["CH1", "CH2", "CH3"]:
            print(f"\n{'#'*80}")
            print(f"#  RUNNING CHANNEL: {ch}")
            print(f"{'#'*80}")
            run_pipeline(chosen_channel=ch)
        return

    # Default: single channel mode
    chosen_channel = "CH1"
    for arg in sys.argv[1:]:
        if arg.upper() in ["CH1", "CH2", "CH3"]:
            chosen_channel = arg.upper()
        elif arg.startswith("--channel="):
            chosen_channel = arg.split("=")[1].upper()

    run_pipeline(chosen_channel=chosen_channel)


if __name__ == "__main__":
    main()
