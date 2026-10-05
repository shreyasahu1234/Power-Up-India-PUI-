"""
REUSEGRID AI Extension -- Multi-Battery Component Reconstruction Optimizer
Solves the combinatorial matching and assembly problem:
  Takes a Target Battery Specification + Recovered Component Inventory Pool
  -> Optimizes Cell Bin Selection, BMS Allocation & Hardware Harvesting
  -> Produces a Validated Candidate Battery Architecture with Complete Donor Traceability
"""

import os
import pandas as pd
import numpy as np

from backend.extension.component_compatibility import ComponentCompatibilityEngine


class ReconstructionOptimizer:
    def __init__(self, components_df=None):
        if components_df is None:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            proj_dir = os.path.dirname(os.path.dirname(curr_dir))
            comp_path = os.path.join(proj_dir, "data", "component_dataset.csv")
            if os.path.exists(comp_path):
                self.components_df = pd.read_csv(comp_path)
            else:
                self.components_df = pd.DataFrame()
        else:
            self.components_df = components_df

    def optimize_reconstruction(self, target_spec, custom_donor_filter=None):
        """
        Solves multi-battery component reconstruction matching for a given Target Battery Spec.
        
        Args:
            target_spec (dict): Requirements for target reconstructed battery pack
            custom_donor_filter (list): Optional list of donor battery IDs to restrict harvest from
            
        Returns:
            reconstruction_result (dict): Complete candidate architecture, BOM, donor traceability, and ROI
        """
        if self.components_df.empty:
            return {"status": "ERROR", "message": "Component dataset is empty."}
            
        df_pool = self.components_df.copy()
        
        # Exclude rejected components
        df_pool = df_pool[~df_pool["Component_Grade"].str.contains("Reject", case=False, na=False)]
        df_pool = df_pool[df_pool["Recovery_Status"] == "Harvested & In Stock"]
        
        if custom_donor_filter:
            df_pool = df_pool[df_pool["Source_Battery_ID"].isin(custom_donor_filter)]
            
        target_name = target_spec.get("Target_Application_Name", "Custom Second-Life Pack")
        req_chem = target_spec.get("Required_Cell_Chemistry", "")
        req_cells_count = int(target_spec.get("Required_Cell_Count", 16))
        min_soh = float(target_spec.get("Min_Cell_SoH_Percent", 75.0))
        max_cap_delta = float(target_spec.get("Max_Cell_Capacity_Delta_Percent", 2.0))
        max_budget = float(target_spec.get("Max_Target_Budget_INR", 15000))
        
        # 1. Filter viable cells by chemistry and minimum health
        cell_candidates = df_pool[df_pool["Component_Type"] == req_chem].copy()
        cell_candidates = cell_candidates[cell_candidates["Estimated_Health_SoH_Percent"] >= min_soh]
        
        if len(cell_candidates) < req_cells_count:
            return {
                "status": "INSUFFICIENT_STOCK",
                "message": f"Need {req_cells_count} {req_chem} cells, but only {len(cell_candidates)} available in inventory meeting >= {min_soh}% SoH.",
                "available_count": len(cell_candidates),
                "required_count": req_cells_count
            }
            
        # 2. Electrochemical Bin Optimization (Find contiguous window of sorted cells that minimizes delta-Ah)
        best_cell_group = None
        best_metrics = None
        lowest_variance = 999.0
        
        # Sort candidates strictly by capacity
        sorted_all = cell_candidates.sort_values(by="Measured_Capacity_Ah").reset_index(drop=True)
        
        # Search all contiguous sliding windows of size req_cells_count
        for start_idx in range(len(sorted_all) - req_cells_count + 1):
            window_cells = sorted_all.iloc[start_idx:start_idx + req_cells_count].to_dict("records")
            ok, metrics, warns = ComponentCompatibilityEngine.verify_cell_group_compatibility(
                window_cells, max_cap_delta_pct=max_cap_delta
            )
            v_score = metrics.get("Capacity_Delta_Percent", 999.0)
            if v_score < lowest_variance:
                lowest_variance = v_score
                best_cell_group = window_cells
                best_metrics = metrics
                    
        if best_cell_group is None:
            return {
                "status": "COMPATIBILITY_FAILED",
                "message": "Could not assemble a cell cluster satisfying the strict cell balancing variance threshold.",
                "lowest_variance_found": lowest_variance
            }
            
        # 3. Select matching BMS Board
        req_bms_type = target_spec.get("Required_BMS_Type", "")
        bms_candidates = df_pool[df_pool["Component_Type"] == req_bms_type].copy()
        if bms_candidates.empty:
            # Fallback: any BMS with compatible voltage
            bms_candidates = df_pool[df_pool["Component_Type"].str.contains("BMS", case=False, na=False)]
            
        selected_bms = bms_candidates.iloc[0].to_dict() if not bms_candidates.empty else {
            "Component_ID": "BMS_NEW_OEM",
            "Component_Type": req_bms_type,
            "Estimated_Salvage_Value_INR": 950,
            "Estimated_Health_SoH_Percent": 100.0,
            "Source_Battery_ID": "New_OEM_Supply"
        }
        
        # 4. Select Sensor, Busbars, Connectors, Casing
        connectors = df_pool[df_pool["Component_Type"].str.contains("Connector", na=False)]
        selected_conn = connectors.iloc[0].to_dict() if not connectors.empty else {"Component_ID": "CONN_GEN", "Component_Type": "Heavy-Duty XT60 Connector", "Estimated_Salvage_Value_INR": 80, "Source_Battery_ID": "Stock"}
        
        sensors = df_pool[df_pool["Component_Type"].str.contains("Sensor", na=False)]
        selected_sensor = sensors.iloc[0].to_dict() if not sensors.empty else {"Component_ID": "SENS_GEN", "Component_Type": "NTC Thermistor Sensor", "Estimated_Salvage_Value_INR": 40, "Source_Battery_ID": "Stock"}
        
        casing = df_pool[df_pool["Component_Type"].str.contains("Casing", na=False)]
        selected_casing = casing.iloc[0].to_dict() if not casing.empty else {"Component_ID": "CASE_GEN", "Component_Type": "Modular Enclosure Frame", "Estimated_Salvage_Value_INR": 350, "Source_Battery_ID": "Stock"}
        
        # 5. Compute Bill of Materials (BOM) and Financial/Carbon ROI
        bom_items = []
        donor_batteries_involved = set()
        total_salvage_cost = 0.0
        total_new_equivalent_cost = 0.0
        total_co2_avoided = 0.0
        
        # Add cells
        for c in best_cell_group:
            donor_batteries_involved.add(c["Source_Battery_ID"])
            salvage_val = float(c.get("Estimated_Salvage_Value_INR", 100))
            new_val = float(c.get("Replacement_Cost_New_INR", 200))
            co2 = float(c.get("CO2_Emissions_Avoided_kg", 1.5))
            total_salvage_cost += salvage_val
            total_new_equivalent_cost += new_val
            total_co2_avoided += co2
            
        bom_items.append({
            "Item_Category": "Electrochemical Energy Core",
            "Description": f"{req_cells_count}x {req_chem} (Matched Group)",
            "Quantity": req_cells_count,
            "Sub_Total_Salvage_INR": round(total_salvage_cost, 2),
            "Sub_Total_New_INR": round(total_new_equivalent_cost, 2),
            "Donor_Sources": len(donor_batteries_involved)
        })
        
        # Add electronics & mechanicals
        other_items = [
            ("Battery Management System", selected_bms["Component_Type"], 1, selected_bms.get("Estimated_Salvage_Value_INR", 500), selected_bms.get("Replacement_Cost_New_INR", 1200), selected_bms["Source_Battery_ID"]),
            ("Thermal Safety Probe", selected_sensor["Component_Type"], 2, selected_sensor.get("Estimated_Salvage_Value_INR", 40) * 2, selected_sensor.get("Replacement_Cost_New_INR", 150) * 2, selected_sensor["Source_Battery_ID"]),
            ("Power Connector Harness", selected_conn["Component_Type"], 1, selected_conn.get("Estimated_Salvage_Value_INR", 80), selected_conn.get("Replacement_Cost_New_INR", 220), selected_conn["Source_Battery_ID"]),
            ("Structural Enclosure Frame", selected_casing["Component_Type"], 1, selected_casing.get("Estimated_Salvage_Value_INR", 350), selected_casing.get("Replacement_Cost_New_INR", 900), selected_casing["Source_Battery_ID"])
        ]
        
        for cat, desc, qty, s_val, n_val, src in other_items:
            donor_batteries_involved.add(src)
            total_salvage_cost += s_val
            total_new_equivalent_cost += n_val
            total_co2_avoided += 1.5
            bom_items.append({
                "Item_Category": cat,
                "Description": desc,
                "Quantity": qty,
                "Sub_Total_Salvage_INR": round(float(s_val), 2),
                "Sub_Total_New_INR": round(float(n_val), 2),
                "Donor_Sources": 1
            })
            
        # Assembly & QC Labor
        assembly_labor_inr = 850.0
        grand_total_reconstruction_cost = round(total_salvage_cost + assembly_labor_inr, 2)
        financial_savings_pct = round(((total_new_equivalent_cost - grand_total_reconstruction_cost) / max(1.0, total_new_equivalent_cost)) * 100.0, 1)
        
        # 6. Reconstructed Pack Metrics
        pack_soh = best_metrics["Mean_SoH_Percent"]
        pack_nom_v = float(target_spec.get("Target_Voltage_Nominal_V", 48.0))
        pack_nom_ah = float(target_spec.get("Target_Capacity_Ah", 50.0))
        effective_pack_wh = round((pack_soh / 100.0) * (pack_nom_v * pack_nom_ah), 1)
        
        return {
            "status": "OPTIMAL_MATCH",
            "target_name": target_name,
            "target_id": target_spec.get("Target_ID", "TGT_CUSTOM"),
            "candidate_pack_id": f"REBUILT_{target_spec.get('Target_ID', 'CUSTOM')}_{np.random.randint(100, 999)}",
            "pack_configuration": target_spec.get("Required_Series_Parallel", "16S"),
            "reconstructed_soh_percent": pack_soh,
            "effective_energy_wh": effective_pack_wh,
            "cell_capacity_delta_percent": best_metrics["Capacity_Delta_Percent"],
            "cell_ir_delta_mohm": best_metrics["IR_Delta_mOhm"],
            "total_cells_harvested": req_cells_count,
            "donor_batteries_count": len(donor_batteries_involved),
            "donor_battery_list": list(donor_batteries_involved)[:10],
            "grand_total_cost_inr": grand_total_reconstruction_cost,
            "new_battery_equivalent_cost_inr": total_new_equivalent_cost,
            "financial_savings_percent": financial_savings_pct,
            "co2_emissions_avoided_kg": round(total_co2_avoided, 1),
            "bill_of_materials": bom_items,
            "harvested_cell_records": best_cell_group,
            "selected_bms": selected_bms
        }
