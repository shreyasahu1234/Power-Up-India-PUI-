"""
REUSEGRID AI Extension -- Component Grading & Health Intelligence Engine
Applies strict multi-dimensional grading standards to recovered components:
  - Grade A: Premium Tier (>85% SoH, low IR, pristine visual)
  - Grade B: Standard Tier (70-85% SoH, normal IR, minor cosmetic blemishes)
  - Grade C: Secondary Buffer Tier (55-70% SoH, derated current envelope)
  - Reject: Disqualified (<55% SoH, swelling, leakage, high IR, or internal short)
"""

import numpy as np


class ComponentGrader:
    @staticmethod
    def grade_component(comp_row):
        """
        Evaluates a single component and returns its grade, operational envelope,
        economic salvage valuation, and environmental footprint metrics.
        """
        soh = float(comp_row.get("Estimated_Health_SoH_Percent", 75.0))
        defect = str(comp_row.get("Visual_Defect_Inspection", "Pristine"))
        ir = float(comp_row.get("Internal_Resistance_mOhm", 20.0))
        comp_type = str(comp_row.get("Component_Type", ""))
        
        # Hard rejection criteria
        is_reject = False
        rejection_reasons = []
        
        if soh < 55.0:
            is_reject = True
            rejection_reasons.append(f"SoH below minimum recovery threshold ({soh:.1f}% < 55%)")
        if any(w in defect.lower() for w in ["swelling", "leakage", "burnt", "short", "puncture"]):
            is_reject = True
            rejection_reasons.append(f"Physical/chemical failure observed ({defect})")
        if ir > 150.0 and "Cell" in comp_type:
            is_reject = True
            rejection_reasons.append(f"Severe impedance degradation ({ir:.1f} mOhm)")
            
        if is_reject:
            return {
                "Grade": "Reject (Unsafe / Recycling Only)",
                "Tier": "Reject",
                "Max_Continuous_C_Rate": 0.0,
                "Recommended_Application_Tier": "Closed-Loop Hydrometallurgical Recycling",
                "Safety_Status": "FAIL",
                "Rejection_Reasons": rejection_reasons,
                "Salvage_Value_INR": 15,
                "Economic_Viability": "Negative (Scrap Only)"
            }
            
        # Grade classification
        if soh >= 85.0 and ir <= 45.0:
            grade_name = "Grade A (Premium Reuse, >85% SoH)"
            tier = "A"
            c_rate = 1.0 if "Cell" in comp_type else 0.0
            app_tier = "Critical Infrastructure, e-Mobility & Fast-Response ESS"
            salvage_val = float(comp_row.get("Estimated_Salvage_Value_INR", 150))
            econ_state = "High Economic Margin (>65% savings vs new)"
        elif soh >= 70.0:
            grade_name = "Grade B (Standard Reuse, 70-85% SoH)"
            tier = "B"
            c_rate = 0.5 if "Cell" in comp_type else 0.0
            app_tier = "Stationary Solar ESS, Rural Microgrids & Telecom Buffers"
            salvage_val = float(comp_row.get("Estimated_Salvage_Value_INR", 100))
            econ_state = "Moderate Economic Margin (~50% savings vs new)"
        else:
            grade_name = "Grade C (Low-Stress Buffer, 55-70% SoH)"
            tier = "C"
            c_rate = 0.2 if "Cell" in comp_type else 0.0
            app_tier = "Low C-Rate IoT Nodes, Solar Lanterns & Micro-Buffers"
            salvage_val = float(comp_row.get("Estimated_Salvage_Value_INR", 60))
            econ_state = "Low Margin (Derated operation necessary)"
            
        return {
            "Grade": grade_name,
            "Tier": tier,
            "Max_Continuous_C_Rate": c_rate,
            "Recommended_Application_Tier": app_tier,
            "Safety_Status": "PASS",
            "Rejection_Reasons": [],
            "Salvage_Value_INR": salvage_val,
            "Economic_Viability": econ_state
        }
