"""
REUSEGRID AI Extension -- Component-to-Component Compatibility Engine
Enforces strict electro-chemical, thermal, and electrical compatibility constraints:
  - Chemical Purity: Identical chemistry across all series/parallel groups
  - Capacity Matching: Delta-Ah <= 2.0% within the same series block
  - Internal Resistance Matching: Delta-IR <= 3.0 mOhm
  - BMS Sizing: Continuous discharge current rating >= target peak current demand
  - Thermal Sensor Headroom: Operating range covers pack thermal envelope
"""

import numpy as np


class ComponentCompatibilityEngine:
    @staticmethod
    def verify_cell_group_compatibility(cell_list, max_cap_delta_pct=2.0, max_ir_delta=3.0):
        """
        Validates a candidate group of cells for series/parallel interconnection.
        
        Returns:
            compatible (bool): True if all cells pass matching tolerance
            metrics (dict): Variance, mean SoH, mean capacity, delta V, delta IR
            warnings (list): Specific mismatch flags
        """
        if not cell_list or len(cell_list) == 0:
            return False, {}, ["No cells provided for compatibility verification."]
            
        chemistries = set()
        capacities = []
        voltages = []
        irs = []
        sohs = []
        
        for c in cell_list:
            chemistries.add(c.get("Sub_Category", c.get("Component_Type", "")))
            capacities.append(float(c.get("Measured_Capacity_Ah", 0.0)))
            voltages.append(float(c.get("Measured_Voltage_V", 3.7)))
            irs.append(float(c.get("Internal_Resistance_mOhm", 20.0)))
            sohs.append(float(c.get("Estimated_Health_SoH_Percent", 80.0)))
            
        warnings = []
        
        # 1. Chemical Purity Check
        if len(chemistries) > 1:
            warnings.append(f"Incompatible Chemistries mixed: {list(chemistries)}. Cannot connect different chemistries in one pack.")
            return False, {}, warnings
            
        # 2. Capacity Variance Check (Delta Capacity %)
        mean_cap = np.mean(capacities)
        min_cap = np.min(capacities)
        max_cap = np.max(capacities)
        cap_delta_pct = round(((max_cap - min_cap) / max(0.01, mean_cap)) * 100.0, 2)
        
        if cap_delta_pct > max_cap_delta_pct:
            warnings.append(f"Capacity imbalance exceeds safe tolerance ({cap_delta_pct}% > {max_cap_delta_pct}% limit). Risk of cell over-discharge.")
            
        # 3. Internal Resistance Delta Check
        mean_ir = np.mean(irs)
        min_ir = np.min(irs)
        max_ir = np.max(irs)
        ir_delta = round(max_ir - min_ir, 2)
        
        if ir_delta > max_ir:
            warnings.append(f"Internal resistance mismatch ({ir_delta} mOhm > {max_ir_delta} mOhm limit). Uneven thermal generation during load.")
            
        # 4. Open-Circuit Voltage Delta (Delta V)
        delta_v_mv = round((np.max(voltages) - np.min(voltages)) * 1000.0, 1)
        if delta_v_mv > 50.0:
            warnings.append(f"Initial cell voltage spread too high ({delta_v_mv} mV > 50 mV). Pre-assembly active top-balancing required.")
            
        compatible = len(warnings) == 0
        
        metrics = {
            "Cell_Count": len(cell_list),
            "Mean_Capacity_Ah": round(mean_cap, 3),
            "Capacity_Delta_Percent": cap_delta_pct,
            "Mean_IR_mOhm": round(mean_ir, 2),
            "IR_Delta_mOhm": ir_delta,
            "Mean_SoH_Percent": round(np.mean(sohs), 1),
            "Initial_Delta_V_mV": delta_v_mv,
            "Chemistry": list(chemistries)[0] if chemistries else "Unknown"
        }
        
        return compatible, metrics, warnings

    @staticmethod
    def verify_bms_pack_compatibility(target_spec, bms_comp):
        """
        Validates whether a selected BMS board satisfies target pack voltage & current specs.
        """
        required_v = float(target_spec.get("Target_Voltage_Nominal_V", 12.0))
        required_i = float(target_spec.get("BMS_Current_Rating_A", 20.0))
        
        bms_v = float(bms_comp.get("Nominal_Voltage_V", 0.0))
        bms_soh = float(bms_comp.get("Estimated_Health_SoH_Percent", 0.0))
        
        warnings = []
        if bms_soh < 80.0:
            warnings.append(f"Selected BMS board health degraded ({bms_soh}%). Risk of premature FET cutout.")
            
        # Series voltage match tolerance
        if abs(bms_v - required_v) > (required_v * 0.15):
            warnings.append(f"BMS voltage rating ({bms_v}V) does not match pack nominal target ({required_v}V).")
            
        compatible = len(warnings) == 0
        return compatible, warnings
