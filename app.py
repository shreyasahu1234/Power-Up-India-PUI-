"""
REUSEGRID AI -- Web-Based Second-Life Battery Intelligence Platform
Built with Streamlit + Python Backend + 500-Battery Benchmark Dataset

Core Pipeline:
  Safety First -> Health Second -> Diagnosis & Repairability Third ->
  Case-Based Similarity Fourth -> Explainable Application Allocation Fifth
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st

# Ensure project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from backend.analysis import analyse_battery, load_resources
from backend.hardware_bridge import HardwareBridge
from backend.matching import (
    build_compatibility_matrix,
    run_multi_battery_optimizer
)
from backend.safety import apply_safety_screening
from backend.health_model import apply_health_model, BatteryHealthModel
from backend.diagnosis import apply_diagnosis
from backend.extension import (
    execute_full_reconstruction_pipeline,
    get_all_reconstruction_targets,
    get_all_components,
    get_inventory_summary
)

# ==============================================================================
# 1. PAGE CONFIG & STYLING
# ==============================================================================
st.set_page_config(
    page_title="REUSEGRID AI — Second-Life Battery Platform",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Clean Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .audit-card {
        background-color: #F0FDF4;
        border-left: 5px solid #16A34A;
        padding: 15px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.92rem;
        line-height: 1.6;
        color: #14532D;
    }
    .audit-card-fail {
        background-color: #FEF2F2;
        border-left: 5px solid #DC2626;
        padding: 15px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.92rem;
        line-height: 1.6;
        color: #7F1D1D;
    }
    .badge-pass {
        background-color: #DCFCE7;
        color: #15803D;
        font-weight: bold;
        padding: 4px 10px;
        border-radius: 4px;
        border: 1px solid #86EFAC;
    }
    .badge-fail {
        background-color: #FEE2E2;
        color: #B91C1C;
        font-weight: bold;
        padding: 4px 10px;
        border-radius: 4px;
        border: 1px solid #FCA5A5;
    }
    .badge-repair {
        background-color: #FEF3C7;
        color: #B45309;
        font-weight: bold;
        padding: 4px 10px;
        border-radius: 4px;
        border: 1px solid #FDE68A;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. CACHED DATA & RESOURCE LOADING
# ==============================================================================
@st.cache_data
def get_benchmark_datasets():
    data_dir = os.path.join(CURRENT_DIR, "data")
    bats_path = os.path.join(data_dir, "batteries.csv")
    if not os.path.exists(bats_path):
        bats_path = os.path.join(data_dir, "battery_dataset.csv")
    needs_path = os.path.join(data_dir, "energy_needs.csv")

    df_bats = pd.read_csv(bats_path)
    df_needs = pd.read_csv(needs_path)
    return df_bats, df_needs


df_batteries_pool, df_energy_needs = get_benchmark_datasets()


# ==============================================================================
# 3. SIDEBAR: NAVIGATION & INPUT MODE
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/battery-charging.png", width=64)
    st.markdown("## **REUSEGRID AI**")
    st.markdown("**Multi-Battery Second-Life Allocation Platform**")
    st.markdown("---")

    mode = st.radio(
        "**Input Source Mode:**",
        [
            "📝 Manual Parameter Input",
            "📡 Live Hardware Bridge (ESP32)",
            "🗂️ Benchmark Archetypes (Demo Presets)"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### 📊 Platform Knowledge Base")
    st.write(f"• **Benchmark Batteries:** {len(df_batteries_pool):,} cases")
    st.write(f"• **Target Energy Applications:** {len(df_energy_needs)} profiles")
    st.write(f"• **Core Architecture:** 5-Layer Cognitive Pipeline")

    st.markdown("---")
    st.caption("Google DeepMind & IIT Mentor Aligned Prototype")


# ==============================================================================
# 4. PRESET ARCHETYPES CONFIGURATION
# ==============================================================================
PRESETS = {
    "Pristine 3S Li-ion Pack (Direct Reuse)": {
        "id": "B104_DEMO",
        "battery_type": "Li-ion 18650 Pack (3S)",
        "voltage": 11.24,
        "current": 1.20,
        "temperature": 27.4,
        "temp_rise_rate": 0.20,
        "internal_resistance_mOhm": 54.0,
        "age_years": 1.4,
        "cycles": 293,
        "capacity_wh": 44.0,
        "physical_damage": "No visible damage",
        "swelling_observed": "None (0-1%)",
        "leakage_observed": "None",
        "component_issue": "None",
        "nominal_voltage": 11.1,
        "pack_configuration": "3S"
    },
    "4S LFP Module with Sensor Lead Detachment (Repairable)": {
        "id": "B028_DEMO",
        "battery_type": "LiFePO4 (LFP) Module (4S)",
        "voltage": 12.89,
        "current": 1.10,
        "temperature": 31.2,
        "temp_rise_rate": 0.32,
        "internal_resistance_mOhm": 62.0,
        "age_years": 2.6,
        "cycles": 747,
        "capacity_wh": 60.0,
        "physical_damage": "No visible damage",
        "swelling_observed": "None (0-1%)",
        "leakage_observed": "None",
        "component_issue": "Temperature Sensor",
        "nominal_voltage": 12.8,
        "pack_configuration": "4S"
    },
    "5S Power Tool Pack with Oxidized Connector (Repairable)": {
        "id": "B089_DEMO",
        "battery_type": "Power Tool Li-ion Pack (5S)",
        "voltage": 17.65,
        "current": 1.40,
        "temperature": 29.5,
        "temp_rise_rate": 0.25,
        "internal_resistance_mOhm": 78.0,
        "age_years": 2.1,
        "cycles": 510,
        "capacity_wh": 48.0,
        "physical_damage": "No visible damage",
        "swelling_observed": "None (0-1%)",
        "leakage_observed": "None",
        "component_issue": "Connector",
        "nominal_voltage": 18.0,
        "pack_configuration": "5S"
    },
    "2S Prismatic Module with Casing Hairline Crack (Limited Use)": {
        "id": "B077_DEMO",
        "battery_type": "Prismatic Li-ion Module (2S)",
        "voltage": 7.35,
        "current": 1.00,
        "temperature": 28.0,
        "temp_rise_rate": 0.18,
        "internal_resistance_mOhm": 75.0,
        "age_years": 3.0,
        "cycles": 620,
        "capacity_wh": 46.0,
        "physical_damage": "Structural enclosure hairline crack",
        "swelling_observed": "None (0-1%)",
        "leakage_observed": "None",
        "component_issue": "None",
        "nominal_voltage": 7.4,
        "pack_configuration": "2S"
    },
    "Thermal Runaway Hazard with Swelling (Unsafe Reject)": {
        "id": "B266_DEMO",
        "battery_type": "Power Tool Li-ion Pack (5S)",
        "voltage": 24.18,
        "current": 2.05,
        "temperature": 52.7,
        "temp_rise_rate": 2.65,
        "internal_resistance_mOhm": 187.0,
        "age_years": 2.1,
        "cycles": 526,
        "capacity_wh": 20.0,
        "physical_damage": "Distorted / Puffed casing",
        "swelling_observed": "Severe Swelling (>6%)",
        "leakage_observed": "None",
        "component_issue": "Uncontrolled Cell Overheating",
        "nominal_voltage": 18.0,
        "pack_configuration": "5S"
    }
}


# ==============================================================================
# 5. MAIN HEADER & APP TABS
# ==============================================================================
st.markdown('<div class="main-header">🔋 REUSEGRID AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Battery Health Intelligence, Modular Repair & Intelligent Second-Life Grid Allocation</div>', unsafe_allow_html=True)

tab_single, tab_fleet, tab_recon, tab_about = st.tabs([
    "🔍 Single Battery Intelligent Evaluation",
    "📊 500-Battery Fleet & Allocation Grid",
    "🔬 Modular Component Recovery & Reconstruction Studio",
    "📖 System Architecture & AI Principles"
])


# ==============================================================================
# TAB 1: SINGLE BATTERY INTELLIGENT EVALUATION
# ==============================================================================
with tab_single:
    st.subheader("Step 1: Ingest or Configure Battery Under Test")

    # Handle preset / hardware bridge defaults
    default_vals = {
        "id": "B001",
        "battery_type": "Li-ion 18650 Pack (3S)",
        "voltage": 11.20,
        "current": 1.15,
        "temperature": 27.5,
        "temp_rise_rate": 0.20,
        "internal_resistance_mOhm": 55.0,
        "age_years": 2.0,
        "cycles": 420,
        "capacity_wh": 42.0,
        "physical_damage": "No visible damage",
        "swelling_observed": "None (0-1%)",
        "leakage_observed": "None",
        "component_issue": "None",
        "nominal_voltage": 11.1,
        "pack_configuration": "3S"
    }

    if "🗂️ Benchmark Archetypes" in mode:
        chosen_preset_name = st.selectbox(
            "Select an Archetype Case from the 500-Battery Dataset:",
            list(PRESETS.keys()),
            index=0
        )
        default_vals = PRESETS[chosen_preset_name]

    elif "📡 Live Hardware Bridge" in mode:
        col_hw1, col_hw2 = st.columns([2, 1])
        with col_hw1:
            channel_choice = st.selectbox(
                "Select Monitored Channel on Hardware Multiplexer:",
                ["CH1 (Low-Voltage Safe Pack)", "CH2 (Sensor Fault Pack)", "CH3 (Thermal Stress Hazard Pack)"],
                index=0
            )
        ch_id = channel_choice.split()[0]
        bridge = HardwareBridge()
        with col_hw2:
            st.info(f"Hardware Bridge Status: **{bridge.mode}**")

        live_data = bridge.read_live_telemetry(ch_id)
        default_vals["id"] = f"{ch_id}_LIVE"
        default_vals["voltage"] = float(live_data["measured_voltage"])
        default_vals["current"] = float(live_data["measured_current"])
        default_vals["temperature"] = float(live_data["operating_temperature"])
        default_vals["temp_rise_rate"] = float(live_data["temp_rise_rate"])
        default_vals["internal_resistance_mOhm"] = float(live_data["internal_resistance_mOhm"])
        default_vals["swelling_observed"] = "Severe Swelling (>6%)" if live_data.get("swelling_observed") else "None (0-1%)"
        default_vals["leakage_observed"] = "Electrolyte Leakage / Breach" if live_data.get("leakage_observed") else "None"
        default_vals["physical_damage"] = "Distorted / Puffed casing" if live_data.get("physical_damage_observed") else "No visible damage"

    # Input Form Layout
    with st.expander("⚡ Battery Parameters & Sensor Measurements", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            bat_id = st.text_input("Battery Identifier (ID)", value=default_vals["id"])
            battery_type = st.selectbox(
                "Battery Chemistry / Form Factor",
                [
                    "Li-ion 18650 Pack (3S)",
                    "Li-ion 18650 Pack (4S)",
                    "LiFePO4 (LFP) Module (4S)",
                    "Power Tool Li-ion Pack (5S)",
                    "NMC Light-EV Pack (10S)",
                    "Prismatic Li-ion Module (2S)",
                    "Li-ion Single Cell (1S)",
                    "Auto-Detect from Telemetry"
                ],
                index=0 if default_vals["battery_type"] not in [
                    "Li-ion 18650 Pack (3S)", "Li-ion 18650 Pack (4S)", "LiFePO4 (LFP) Module (4S)",
                    "Power Tool Li-ion Pack (5S)", "NMC Light-EV Pack (10S)", "Prismatic Li-ion Module (2S)",
                    "Li-ion Single Cell (1S)"
                ] else [
                    "Li-ion 18650 Pack (3S)", "Li-ion 18650 Pack (4S)", "LiFePO4 (LFP) Module (4S)",
                    "Power Tool Li-ion Pack (5S)", "NMC Light-EV Pack (10S)", "Prismatic Li-ion Module (2S)",
                    "Li-ion Single Cell (1S)"
                ].index(default_vals["battery_type"])
            )
            age = st.number_input("Estimated Age (Years)", min_value=0.1, max_value=15.0, value=float(default_vals["age_years"]), step=0.1)
            cycles = st.number_input("Charge / Discharge Cycles", min_value=0, max_value=3000, value=int(default_vals["cycles"]), step=10)

        with col2:
            voltage = st.number_input("Measured Voltage (V)", min_value=0.0, max_value=60.0, value=float(default_vals["voltage"]), step=0.05)
            current = st.number_input("Measured Current (A)", min_value=0.0, max_value=30.0, value=float(default_vals["current"]), step=0.05)
            temperature = st.number_input("Skin Temperature (°C)", min_value=-10.0, max_value=90.0, value=float(default_vals["temperature"]), step=0.5)
            temp_rise = st.number_input("Thermal Rise Velocity (°C/min)", min_value=0.0, max_value=10.0, value=float(default_vals["temp_rise_rate"]), step=0.05)

        with col3:
            ir = st.number_input("Internal Resistance (mΩ)", min_value=10.0, max_value=500.0, value=float(default_vals["internal_resistance_mOhm"]), step=1.0)
            capacity = st.number_input("Measured Remaining Capacity (Wh)", min_value=0.0, max_value=500.0, value=float(default_vals["capacity_wh"]), step=1.0)
            casing_condition = st.selectbox(
                "Physical Enclosure Condition",
                ["No visible damage", "Minor hairline scratches", "Structural enclosure hairline crack", "Distorted / Puffed casing", "Severe mechanical puncture"],
                index=0 if default_vals["physical_damage"] not in ["No visible damage", "Minor hairline scratches", "Structural enclosure hairline crack", "Distorted / Puffed casing", "Severe mechanical puncture"]
                else ["No visible damage", "Minor hairline scratches", "Structural enclosure hairline crack", "Distorted / Puffed casing", "Severe mechanical puncture"].index(default_vals["physical_damage"])
            )
            swelling = st.selectbox(
                "Mechanical Swelling Deformation",
                ["None (0-1%)", "Mild (2-3%)", "Moderate (4-5%)", "Severe Swelling (>6%)"],
                index=["None (0-1%)", "Mild (2-3%)", "Moderate (4-5%)", "Severe Swelling (>6%)"].index(default_vals["swelling_observed"])
                if default_vals["swelling_observed"] in ["None (0-1%)", "Mild (2-3%)", "Moderate (4-5%)", "Severe Swelling (>6%)"] else 0
            )

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            leakage = st.selectbox(
                "Electrolyte Leakage Observation",
                ["None", "Electrolyte Leakage / Breach"],
                index=0 if default_vals["leakage_observed"] == "None" else 1
            )
        with col_b2:
            component_issue = st.selectbox(
                "Observed / Suspected Component Issue",
                [
                    "None",
                    "Temperature Sensor",
                    "Connector",
                    "BMS",
                    "Structural Enclosure Hairline Crack",
                    "Internal Short Circuit",
                    "Uncontrolled Cell Overheating",
                    "Capacity Degradation"
                ],
                index=0 if default_vals["component_issue"] not in [
                    "None", "Temperature Sensor", "Connector", "BMS",
                    "Structural Enclosure Hairline Crack", "Internal Short Circuit",
                    "Uncontrolled Cell Overheating", "Capacity Degradation"
                ] else [
                    "None", "Temperature Sensor", "Connector", "BMS",
                    "Structural Enclosure Hairline Crack", "Internal Short Circuit",
                    "Uncontrolled Cell Overheating", "Capacity Degradation"
                ].index(default_vals["component_issue"])
            )

    # ACTION BUTTON
    analyze_clicked = st.button("🔍 ANALYZE BATTERY THROUGH REUSEGRID AI", type="primary", use_container_width=True)

    if analyze_clicked or "last_analysis_result" in st.session_state:
        if analyze_clicked:
            battery_dict = {
                "id": bat_id,
                "battery_type": battery_type,
                "voltage": voltage,
                "current": current,
                "temperature": temperature,
                "temp_rise_rate": temp_rise,
                "internal_resistance_mOhm": ir,
                "age_years": age,
                "cycles": cycles,
                "capacity_wh": capacity,
                "physical_damage": 0 if casing_condition in ["No visible damage", "Minor hairline scratches"] else 1,
                "physical_casing": casing_condition,
                "swelling_observed": swelling,
                "leakage_observed": leakage,
                "component_issue": component_issue,
                "nominal_voltage": default_vals.get("nominal_voltage", None),
                "pack_configuration": default_vals.get("pack_configuration", None)
            }
            with st.spinner("Executing 5-Layer REUSEGRID AI Cognitive Pipeline..."):
                result = analyse_battery(battery_dict)
                st.session_state["last_analysis_result"] = result
        else:
            result = st.session_state["last_analysis_result"]

        st.markdown("---")
        st.subheader("Step 2: Multi-Layer Intelligence Analysis Report")

        # -------------------------------------------------------------
        # METRIC GRID: Current Condition
        # -------------------------------------------------------------
        c_v, c_i, c_t, c_ir, c_cyc, c_cap = st.columns(6)
        c_v.metric("Voltage", f"{result['raw_inputs']['voltage']:.2f} V")
        c_i.metric("Current", f"{result['raw_inputs']['current']:.2f} A")
        c_t.metric("Temperature", f"{result['raw_inputs']['temperature']:.1f} °C")
        c_ir.metric("Internal Res.", f"{result['raw_inputs']['ir_mohm']:.1f} mΩ")
        c_cyc.metric("Cycles / Age", f"{result['raw_inputs']['cycles']} cyc")
        c_cap.metric("Capacity", f"{result['raw_inputs']['capacity_wh']:.1f} Wh")

        # -------------------------------------------------------------
        # LAYER 1: SAFETY SCREENING (MANDATORY HARD GATE)
        # -------------------------------------------------------------
        st.markdown("### 🛡️ Layer 1 — Mandatory Safety Screening (Hard Gate)")
        safety_status = result["safety"]["status"]

        if safety_status == "PASS":
            st.success("✅ **SAFETY SCREEN PASSED** — Battery meets all thermal, electrochemical, and mechanical structural safety criteria.")
            st.caption(f"**Safety Rationale:** {result['safety']['reason']}")
        else:
            st.error("❌ **UNSAFE — REJECTED AT SAFETY GATE (UNFIT FOR REUSE)**")
            st.markdown(f"""
            <div class="audit-card-fail">
                <b>CRITICAL SAFETY HAZARDS DETECTED:</b><br>
                {result['safety']['reason']}<br><br>
                <i>Core Rule: Under the REUSEGRID AI protocol, a good matching score can never override a physical safety failure.
                This battery is disqualified from second-life allocation and routed immediately to certified closed-loop recycling.</i>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Split into Health & Diagnosis columns
        col_h, col_d = st.columns(2)

        # -------------------------------------------------------------
        # LAYER 2: BATTERY HEALTH INTELLIGENCE
        # -------------------------------------------------------------
        with col_h:
            st.markdown("### ❤️ Layer 2 — Health Intelligence Engine")
            health_pct = result["health"]
            if safety_status == "PASS":
                st.metric("Estimated State of Health (SoH)", f"{health_pct:.1f}%")
                st.progress(min(health_pct / 100.0, 1.0))
                st.write(f"• **Remaining Usable Energy:** **{result['remaining_energy_wh']:.1f} Wh**")
                st.write(f"• **Model:** Trained Random Forest Regressor with cell-normalized voltage & dynamic impedance penalty")
            else:
                st.metric("Estimated State of Health (SoH)", "0.0% (Hazard Lockout)")
                st.progress(0.0)
                st.caption("Health evaluation zeroed out due to Layer 1 safety failure.")

        # -------------------------------------------------------------
        # LAYER 3: COMPONENT DIAGNOSIS & REPAIRABILITY
        # -------------------------------------------------------------
        with col_d:
            st.markdown("### 🔧 Layer 3 — Diagnosis & Repair Assessment")
            st.write(f"**Diagnosed Problem:** `{result['diagnosis']}`")
            st.write(f"**Impact Severity:** `{result['impact_level']}`")

            rep = result["repairability"]
            if rep == "YES" and result["repair_cost_inr"] > 0:
                st.markdown(f'<span class="badge-repair">🔧 REPAIRABLE (Estimated Cost: INR {result["repair_cost_inr"]})</span>', unsafe_allow_html=True)
                st.write(f"• **Action Pathway:** {result['repair_action']}")
                st.write(f"• **Post-Repair Retest:** `{result['retest_status']}`")
            elif rep == "YES" and result["repair_cost_inr"] == 0:
                st.markdown('<span class="badge-pass">✅ PRISTINE (Direct Reuse — No Repair Required)</span>', unsafe_allow_html=True)
                st.write(f"• **Action Pathway:** {result['repair_action']}")
            elif rep == "LIMITED":
                st.markdown('<span class="badge-repair">⚠️ USABLE WITH LIMITATIONS (Derated Operation)</span>', unsafe_allow_html=True)
                st.write(f"• **Action Pathway:** {result['repair_action']}")
            else:
                st.markdown('<span class="badge-fail">❌ NON-REPAIRABLE (Direct to Closed-Loop Recycling)</span>', unsafe_allow_html=True)
                st.write(f"• **Action Pathway:** {result['repair_action']}")

            st.write(f"• **Pool Eligibility Status:** **{result['reuse_category']}**")

        st.markdown("---")

        # -------------------------------------------------------------
        # SIMILARITY INTELLIGENCE: 500-BATTERY DATASET CASES
        # -------------------------------------------------------------
        st.markdown("### 🤖 Case-Based Reasoning: Top Historical Matches in Dataset")
        best_case = result["best_case"]

        c_sim1, c_sim2, c_sim3 = st.columns(3)
        c_sim1.metric("Nearest Dataset Archetype", f"ID {best_case['id']}")
        c_sim2.metric("Similarity Score", f"{best_case['similarity_percent']:.1f}%")
        c_sim3.metric("Historical Second-Life Outcome", f"{best_case['recommended_application']}")

        st.write("#### Nearest 5 Benchmark Cases Identified via Weighted Euclidean Distance:")
        sim_df = pd.DataFrame(result["similar_cases"])
        st.dataframe(
            sim_df[[
                "Rank", "Battery_ID", "Battery_Type", "Voltage", "Temp_C", "IR_mOhm",
                "Cycles", "Health_SoH", "Capacity_Wh", "Historical_Category", "Recommended_Application", "Similarity"
            ]],
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # -------------------------------------------------------------
        # LAYERS 4 & 5: ALLOCATION & EXPLAINABLE DECISION
        # -------------------------------------------------------------
        st.markdown("### ⚡ Layers 4 & 5 — Second-Life Application Matching & Decision Audit")

        if safety_status == "PASS" and not "Unfit" in result["reuse_category"]:
            c_app1, c_app2 = st.columns([3, 1])
            with c_app1:
                st.success(f"🎯 **OPTIMAL ALLOCATION:** **{result['application']}**")
            with c_app2:
                st.metric("System Compatibility Score", f"{result['application_score']}%")

            st.markdown("#### **Explainable Decision Audit Card (Why This Match?):**")
            st.markdown(f"""
            <div class="audit-card">
                <b>REUSEGRID AI AUDIT TRACE — BATTERY {result['id']} -> {result['application']}</b><br><br>
                {result['application_explanation'].replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="audit-card-fail">
                <b>ALLOCATION REJECTED:</b> Battery is classified as <b>{result['reuse_category']}</b>.<br><br>
                Under the REUSEGRID AI safety-first governance, this unit cannot be deployed to any active energy application.
                Recommended Action: <b>{result['repair_action']}</b>.
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# TAB 2: 500-BATTERY FLEET & ENERGY GRID ALLOCATION
# ==============================================================================
with tab_fleet:
    st.subheader("Global Optimization Across the 500-Battery Pool")
    st.write(
        "Demonstrates simultaneous multi-battery evaluation and global integer linear programming (OR-Tools) "
        "allocation across 9 second-life energy applications."
    )

    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        run_opt_clicked = st.button("🚀 RUN GLOBAL MULTI-BATTERY OPTIMIZATION (500 BATTERIES)", type="primary")

    if run_opt_clicked or "fleet_opt_done" in st.session_state:
        if run_opt_clicked:
            with st.spinner("Processing 500 batteries through Layers 1-5 with Google OR-Tools..."):
                # Run full pipeline on df_batteries_pool
                pool_df = apply_safety_screening(df_batteries_pool.copy())
                h_model = BatteryHealthModel()
                pool_df = apply_health_model(pool_df, h_model)
                pool_df = apply_diagnosis(pool_df)

                eligible_df = pool_df[pool_df["eligibility"].str.startswith("Eligible")].copy()
                compat_mat = build_compatibility_matrix(eligible_df, df_energy_needs)
                allocations, total_utility = run_multi_battery_optimizer(eligible_df, df_energy_needs, compat_mat)

                alloc_df = pd.DataFrame(allocations)
                st.session_state["fleet_opt_done"] = True
                st.session_state["pool_evaluated"] = pool_df
                st.session_state["fleet_alloc_df"] = alloc_df
                st.session_state["total_utility"] = total_utility

        pool_df = st.session_state["pool_evaluated"]
        alloc_df = st.session_state["fleet_alloc_df"]
        total_utility = st.session_state["total_utility"]

        # Summary KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        pass_count = (pool_df["safety_status"] == "PASS").sum()
        fail_count = (pool_df["safety_status"] == "FAIL").sum()
        direct_count = (pool_df["eligibility"] == "Eligible (Direct)").sum()
        repair_count = (pool_df["eligibility"] == "Eligible (Post-Repair)").sum()

        kpi1.metric("Safety Passed", f"{pass_count} / {len(pool_df)}")
        kpi2.metric("Safety Rejected", f"{fail_count} ({fail_count/len(pool_df)*100:.1f}%)")
        kpi3.metric("Total Optimal Allocations", f"{len(alloc_df)}")
        kpi4.metric("OR-Tools System Utility", f"{total_utility:.1f}")

        st.markdown("---")
        st.subheader("Application Deployment Quota Fulfillment")

        # Quota fulfillment table
        app_summary = []
        for _, app_row in df_energy_needs.iterrows():
            a_name = app_row["application"]
            q = int(app_row.get("allocation_quota", 0))
            matched = (alloc_df["application"] == a_name).sum() if not alloc_df.empty else 0
            app_summary.append({
                "Application": a_name,
                "Criticality (1-5)": app_row["criticality"],
                "Min Health %": app_row["min_health"],
                "Min Energy Wh": app_row["min_energy_wh"],
                "Quota": q,
                "Batteries Allocated": matched,
                "Fulfillment %": f"{(matched / max(q, 1)) * 100:.1f}%"
            })
        st.dataframe(pd.DataFrame(app_summary), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Detailed Allocation Ledger (First 20 Batteries)")
        st.dataframe(alloc_df.head(20), use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 3: MODULAR COMPONENT RECOVERY & RECONSTRUCTION STUDIO (ADDITIVE EXTENSION)
# ==============================================================================
with tab_recon:
    st.subheader("🔬 Modular Component Recovery, Salvage Grading & Battery Reconstruction")
    st.markdown("""
    **Cognitive Extension Layer**: Harvests tested sub-assemblies (cells, BMS boards, thermistors, busbars, enclosures) 
    from retired/rejected packs, grades their electro-chemical health, and runs combinatorial integer optimization 
    to reconstruct custom second-life battery packs with strict cell balancing constraints.
    """)

    # -------------------------------------------------------------
    # 1. RECOVERED COMPONENT POOL & INVENTORY SUMMARY
    # -------------------------------------------------------------
    st.markdown("### 📦 1. Harvested Component Resource Pool (800 Cases)")
    df_all_comps = get_all_components()
    df_inv_summary = get_inventory_summary()

    # Top KPI row
    if not df_all_comps.empty:
        k1, k2, k3, k4 = st.columns(4)
        total_comps = len(df_all_comps)
        grade_a_cnt = (df_all_comps["Component_Grade"].str.contains("Grade A")).sum()
        total_salvage_val = df_all_comps["Estimated_Salvage_Value_INR"].sum()
        total_co2_sav = df_all_comps["CO2_Emissions_Avoided_kg"].sum()

        k1.metric("Harvested Components", f"{total_comps:,} units")
        k2.metric("Grade A (Premium)", f"{grade_a_cnt} ({grade_a_cnt/total_comps*100:.1f}%)")
        k3.metric("Inventory Economic Value", f"INR {total_salvage_val:,.0f}")
        k4.metric("CO2 Emissions Avoided", f"{total_co2_sav:,.1f} kg")

    # Filter controls
    col_f1, col_f2 = st.columns([2, 2])
    with col_f1:
        comp_type_filter = st.selectbox(
            "Filter by Component Sub-Assembly:",
            ["All Component Types"] + sorted(df_all_comps["Component_Type"].unique().tolist()) if not df_all_comps.empty else ["All"]
        )
    with col_f2:
        grade_filter = st.selectbox(
            "Filter by Quality Tier / Grade:",
            ["All Quality Grades", "Grade A (Premium Reuse, >85% SoH)", "Grade B (Standard Reuse, 70-85% SoH)", "Grade C (Low-Stress Buffer, 55-70% SoH)", "Reject (Unsafe / Recycling Only)"]
        )

    # Filtered dataframe
    df_filtered_comps = df_all_comps.copy()
    if comp_type_filter != "All Component Types":
        df_filtered_comps = df_filtered_comps[df_filtered_comps["Component_Type"] == comp_type_filter]
    if grade_filter != "All Quality Grades":
        df_filtered_comps = df_filtered_comps[df_filtered_comps["Component_Grade"] == grade_filter]

    st.dataframe(
        df_filtered_comps[[
            "Component_ID", "Source_Battery_ID", "Component_Type", "Nominal_Voltage_V", 
            "Measured_Capacity_Ah", "Internal_Resistance_mOhm", "Historical_Cycles",
            "Estimated_Health_SoH_Percent", "Component_Grade", "Electrochemical_Matching_Group",
            "Estimated_Salvage_Value_INR", "CO2_Emissions_Avoided_kg"
        ]].head(25),
        use_container_width=True,
        hide_index=True
    )
    st.caption("Displaying top 25 records matching filter. Full 800-component inventory dataset available for download.")

    st.markdown("---")

    # -------------------------------------------------------------
    # 2. TARGET BATTERY RECONSTRUCTION STUDIO
    # -------------------------------------------------------------
    st.markdown("### 🎯 2. Target Battery Definition & Multi-Battery Reconstruction Optimizer")
    df_targets = get_all_reconstruction_targets()

    col_t1, col_t2 = st.columns([3, 2])
    with col_t1:
        target_choice = st.selectbox(
            "Select Second-Life Target Application to Reconstruct:",
            df_targets["Target_Application_Name"].tolist() if not df_targets.empty else ["Custom Target"],
            index=0
        )
    
    target_row = df_targets[df_targets["Target_Application_Name"] == target_choice].iloc[0].to_dict() if not df_targets.empty else {}

    with col_t2:
        st.markdown(f"**Target Architecture:** `{target_row.get('Required_Series_Parallel', '16S')}` ({target_row.get('Required_Cell_Count', 16)} cells)")
        st.markdown(f"**Required Chemistry:** `{target_row.get('Required_Cell_Chemistry', 'LiFePO4')}`")
        st.markdown(f"**Max Budget:** `INR {target_row.get('Max_Target_Budget_INR', 20000):,}` | **Min Cell SoH:** `{target_row.get('Min_Cell_SoH_Percent', 80)}%`")

    st.info(f"📋 **Application Scope:** {target_row.get('Reconstruction_Target_Description', '')}")

    # Reconstruction Action Button
    run_recon_clicked = st.button("⚡ OPTIMIZE & VALIDATE RECONSTRUCTED BATTERY", type="primary", use_container_width=True)

    if run_recon_clicked or "last_recon_result" in st.session_state:
        if run_recon_clicked:
            with st.spinner(f"Harvesting, binning, and validating components for '{target_choice}'..."):
                recon_res = execute_full_reconstruction_pipeline(target_row)
                st.session_state["last_recon_result"] = recon_res
        else:
            recon_res = st.session_state["last_recon_result"]

        st.markdown("---")
        st.subheader("Step 3: Reconstructed Candidate Battery Validation & Traceability")

        reconstruction = recon_res.get("reconstruction", {})
        validation = recon_res.get("validation", {})
        feedback_df = recon_res.get("feedback", pd.DataFrame())

        # Validation status badge
        cert_status = validation.get("certification_status", "PENDING")
        if "CERTIFIED" in cert_status:
            st.success(f"🏆 **PRE-COMMISSIONING CERTIFICATION:** **{cert_status}**")
        elif "PROVISIONAL" in cert_status:
            st.warning(f"⚠️ **PRE-COMMISSIONING CERTIFICATION:** **{cert_status}**")
        else:
            st.error(f"❌ **PRE-COMMISSIONING CERTIFICATION:** **{cert_status}**")

        st.write(f"**Quality Gate Rationale:** {validation.get('recommendation', '')}")

        # Summary KPIs
        rk1, rk2, rk3, rk4, rk5 = st.columns(5)
        rk1.metric("Pack Reconstructed SoH", f"{reconstruction.get('reconstructed_soh_percent', 0.0):.1f}%")
        rk2.metric("Effective Energy", f"{reconstruction.get('effective_energy_wh', 0.0):.1f} Wh")
        rk3.metric("Cell Capacity Delta", f"{reconstruction.get('cell_capacity_delta_percent', 0.0):.2f}%", help="Must be <= tolerance threshold for active string balancing")
        rk4.metric("Reconstruction Cost", f"INR {reconstruction.get('grand_total_cost_inr', 0):,.0f}")
        rk5.metric("Cost Savings vs New", f"{reconstruction.get('financial_savings_percent', 0.0):.1f}%", f"{reconstruction.get('co2_emissions_avoided_kg', 0.0)} kg CO2 saved")

        # -------------------------------------------------------------
        # QUALITY GATES TABLE
        # -------------------------------------------------------------
        st.markdown("#### 🛡️ Pre-Commissioning Electrochemical Safety Gates")
        gates_list = validation.get("gate_evaluations", [])
        if gates_list:
            df_gates = pd.DataFrame(gates_list)
            st.dataframe(df_gates, use_container_width=True, hide_index=True)

        # -------------------------------------------------------------
        # BILL OF MATERIALS & DONOR TRACEABILITY
        # -------------------------------------------------------------
        col_bom, col_trace = st.columns([3, 2])
        with col_bom:
            st.markdown("#### 📋 Candidate Bill of Materials (BOM)")
            bom_list = reconstruction.get("bill_of_materials", [])
            if bom_list:
                df_bom = pd.DataFrame(bom_list)
                st.dataframe(df_bom, use_container_width=True, hide_index=True)

        with col_trace:
            st.markdown("#### 🧬 Donor Battery Traceability Map")
            donor_list = reconstruction.get("donor_battery_list", [])
            st.write(f"• **Donor Battery Packs Harvested:** **{reconstruction.get('donor_batteries_count', 0)} packs**")
            st.write(f"• **Sample Donor Sources:** `{', '.join(donor_list)}`")
            st.write(f"• **Candidate Battery Identifier:** `{reconstruction.get('candidate_pack_id', 'REBUILT_001')}`")
            st.write(f"• **Assembly & QC Labor:** INR 850 included in total cost")

        st.markdown("---")

        # -------------------------------------------------------------
        # DEGRADATION & CYCLE LIFE FEEDBACK SIMULATOR
        # -------------------------------------------------------------
        st.markdown("#### 📈 Projected Degradation & Predictive Lifecycle Feedback (500 Operating Cycles)")
        if not feedback_df.empty:
            c_chart1, c_chart2 = st.columns([3, 2])
            with c_chart1:
                st.line_chart(
                    feedback_df.set_index("Cycle")[["Projected_SoH_Percent", "Capacity_Retention_Percent"]],
                    use_container_width=True
                )
            with c_chart2:
                st.dataframe(
                    feedback_df[["Cycle", "Projected_SoH_Percent", "Cell_Delta_V_mV", "Operational_State"]].iloc[[0, 2, 4, 6, 8, 10]],
                    use_container_width=True,
                    hide_index=True
                )


# ==============================================================================
# TAB 4: SYSTEM ARCHITECTURE & AI PRINCIPLES
# ==============================================================================
with tab_about:

    st.subheader("REUSEGRID AI — Core Cognitive Architecture")
    st.markdown("""
    ### **Core Principle**
    > **Safety first → Health second → Repair/Capability assessment third → Energy-need analysis → Explainable matching**

    ---

    ### **The 3 AI Intelligence Engines**
    1. **AI 1 — Case-Based Similarity Intelligence (KNN / Weighted Distance):**
       Matches any incoming live battery telemetry against 500 empirically validated benchmark battery cases, inferring degradation history and archetype baselines.
    2. **AI 2 — Battery Health & Repair Intelligence (Random Forest + Modular Rules):**
       Predicts State of Health (SoH %) and diagnoses specific replaceable components (temperature sensors, BMS boards, connector oxidation, casing cracks).
    3. **AI 3 — Multi-Battery Energy Grid Optimization (Integer Linear Programming via Google OR-Tools):**
       Solves global assignment across multiple battery types and multiple second-life energy demands while strictly obeying quotas, safety hard gates, and voltage compatibility windows.

    ---

    ### **Physical Prototype Hardware Stack**
    * **Microcontroller:** ESP32 DevKit V1 (Dual-core 240 MHz)
    * **Electrical Telemetry:** 3× INA219 High-Side DC Voltage & Current Sensors (I2C at 0x40, 0x41, 0x44)
    * **Thermal Telemetry:** 3× DS18B20 Digital Temperature Probes (OneWire bus on GPIO 4)
    * **Safety Watchdog:** Hardware bridge enforces thermal velocity limits (dT/dt ≤ 1.8°C/min) and cell-level under-voltage lockouts.
    """)
