"""
REUSEGRID AI -- Additive Extension Master Pipeline
Orchestrates:
  1. Component Harvesting & Inventory Management
  2. Component Grading & Health Intelligence
  3. Multi-Battery Reconstruction Optimization
  4. Pre-Commissioning Quality & Safety Validation
  5. Degradation & Lifecycle Feedback Loop
"""

import os
import pandas as pd

from backend.extension.component_extractor import ComponentExtractor
from backend.extension.component_grading import ComponentGrader
from backend.extension.component_compatibility import ComponentCompatibilityEngine
from backend.extension.reconstruction_optimizer import ReconstructionOptimizer
from backend.extension.candidate_validator import CandidateBatteryValidator
from backend.extension.feedback_monitor import FeedbackMonitor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_DIR = os.path.join(PROJ_DIR, "data")


def get_all_reconstruction_targets():
    target_path = os.path.join(DATA_DIR, "reconstruction_targets.csv")
    if os.path.exists(target_path):
        return pd.read_csv(target_path)
    return pd.DataFrame()


def get_all_components(filter_grade=None):
    comp_path = os.path.join(DATA_DIR, "component_dataset.csv")
    if os.path.exists(comp_path):
        df = pd.read_csv(comp_path)
        if filter_grade:
            df = df[df["Component_Grade"].str.contains(filter_grade, case=False, na=False)]
        return df
    return pd.DataFrame()


def get_inventory_summary():
    inv_path = os.path.join(DATA_DIR, "battery_component_inventory.csv")
    if os.path.exists(inv_path):
        return pd.read_csv(inv_path)
    return pd.DataFrame()


def execute_full_reconstruction_pipeline(target_id_or_spec, donor_filter=None):
    """
    Runs the complete additive reconstruction workflow for a selected target specification.
    """
    if isinstance(target_id_or_spec, str):
        df_t = get_all_reconstruction_targets()
        match = df_t[df_t["Target_ID"] == target_id_or_spec]
        if match.empty:
            return {"status": "ERROR", "message": f"Target ID '{target_id_or_spec}' not found."}
        target_spec = match.iloc[0].to_dict()
    else:
        target_spec = target_id_or_spec

    # 1. Optimize Component Selection & Assembly
    optimizer = ReconstructionOptimizer()
    reconstruction_result = optimizer.optimize_reconstruction(target_spec, custom_donor_filter=donor_filter)
    
    if reconstruction_result.get("status") != "OPTIMAL_MATCH":
        return {
            "reconstruction": reconstruction_result,
            "validation": {"certification_status": "FAILED", "passed_all_gates": False},
            "feedback": pd.DataFrame()
        }

    # 2. Validate Candidate Battery Quality & Safety Gates
    validation_report = CandidateBatteryValidator.validate_candidate_pack(reconstruction_result, target_spec)
    
    # 3. Simulate Degradation & Monitoring Feedback
    pack_soh = reconstruction_result["reconstructed_soh_percent"]
    cap_delta = reconstruction_result["cell_capacity_delta_percent"]
    feedback_df = FeedbackMonitor.simulate_degradation_trajectory(pack_soh, cap_delta, operating_cycles=500)
    
    return {
        "status": "SUCCESS",
        "target_spec": target_spec,
        "reconstruction": reconstruction_result,
        "validation": validation_report,
        "feedback": feedback_df
    }
