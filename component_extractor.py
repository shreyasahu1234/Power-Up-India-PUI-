"""
REUSEGRID AI Extension -- Component Extraction & Harvesting Engine
Decomposes retired or damaged battery packs into salvageable modular sub-assemblies:
  - Electrochemical Cells / Modules
  - BMS Protection Boards
  - Thermal Sensors (NTC Thermistors)
  - Connectors & Terminal Harnesses
  - Copper/Nickel Busbars
  - Enclosure Frames & Spacers
"""

import os
import pandas as pd
import numpy as np


class ComponentExtractor:
    def __init__(self, dataset_path=None):
        if dataset_path is None:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            proj_dir = os.path.dirname(os.path.dirname(curr_dir))
            dataset_path = os.path.join(proj_dir, "data", "component_dataset.csv")
            
        self.dataset_path = dataset_path
        if os.path.exists(self.dataset_path):
            self.df_components = pd.read_csv(self.dataset_path)
        else:
            self.df_components = pd.DataFrame()

    def get_salvageable_pool(self, min_soh=55.0, exclude_rejects=True):
        """Returns all in-stock components meeting minimum health threshold."""
        if self.df_components.empty:
            return pd.DataFrame()
        
        df = self.df_components.copy()
        if exclude_rejects:
            df = df[~df["Component_Grade"].str.contains("Reject", case=False, na=False)]
            df = df[df["Recovery_Status"] == "Harvested & In Stock"]
            
        df = df[df["Estimated_Health_SoH_Percent"] >= min_soh]
        return df

    def extract_from_battery_record(self, bat_row):
        """
        Simulates on-demand component extraction from a specific donor battery pack.
        Determines which sub-components are salvageable based on diagnosed defect state.
        """
        bat_id = bat_row.get("Battery_ID", "DONOR_BAT")
        bat_type = bat_row.get("Battery_Type", "Li-ion 18650 Pack (3S)")
        soh = float(bat_row.get("State_of_Health_SoH_Percent", bat_row.get("estimated_health", 75.0)))
        primary_issue = str(bat_row.get("Primary_Diagnosed_Issue", bat_row.get("component_issue", "None")))
        casing_cond = str(bat_row.get("Physical_Casing_Condition", "Intact"))
        
        extracted = []
        
        # 1. Extract electrochemical cells
        cell_defect = "None"
        if "Capacity Degradation" in primary_issue or "Over-discharge" in primary_issue:
            cell_defect = "Degraded Active Material"
            cell_soh = max(45.0, soh - 6.0)
        elif "Short" in primary_issue or "Overheating" in primary_issue:
            cell_defect = "Thermal Stress / Micro-short Risk"
            cell_soh = 0.0
        else:
            cell_soh = soh
            
        extracted.append({
            "Source_Battery_ID": bat_id,
            "Component_Type": f"Cells ({bat_type})",
            "Estimated_Health_SoH_Percent": cell_soh,
            "Salvageable": cell_soh >= 55.0,
            "Defect_Observation": cell_defect,
            "Salvage_Pathway": "Electrochemical Binning & Re-matching" if cell_soh >= 55.0 else "Raw Material Hydrometallurgical Recycling"
        })
        
        # 2. Extract BMS Board
        bms_ok = not ("BMS" in primary_issue or "Port" in primary_issue or "MOSFET" in primary_issue or cell_soh == 0.0)
        extracted.append({
            "Source_Battery_ID": bat_id,
            "Component_Type": "BMS Protection Circuit",
            "Estimated_Health_SoH_Percent": 95.0 if bms_ok else 20.0,
            "Salvageable": bms_ok,
            "Defect_Observation": "BMS Discharge Port Fault" if not bms_ok else "Pristine Logic & Switching",
            "Salvage_Pathway": "Benchtop Calibration & Direct Reuse" if bms_ok else "Component-level Repair or E-waste Scrap"
        })
        
        # 3. Extract Thermistor Temperature Sensor
        sensor_ok = not ("Sensor" in primary_issue or "Thermistor" in primary_issue)
        extracted.append({
            "Source_Battery_ID": bat_id,
            "Component_Type": "NTC Thermistor Sensor",
            "Estimated_Health_SoH_Percent": 98.0 if sensor_ok else 15.0,
            "Salvageable": sensor_ok,
            "Defect_Observation": "Detached Wire / Open Circuit" if not sensor_ok else "Calibrated Resistance Response",
            "Salvage_Pathway": "Direct Harness Re-use" if sensor_ok else "Recalibrate Lead or Scrap"
        })
        
        # 4. Extract Physical Enclosure / Casing
        casing_ok = not ("Crack" in casing_cond or "Distorted" in casing_cond or "Puffed" in casing_cond)
        extracted.append({
            "Source_Battery_ID": bat_id,
            "Component_Type": "Structural Enclosure Frame",
            "Estimated_Health_SoH_Percent": 95.0 if casing_ok else 40.0,
            "Salvageable": casing_ok,
            "Defect_Observation": casing_cond if not casing_ok else "Intact Mechanical Barrier",
            "Salvage_Pathway": "Direct Secondary Packaging" if casing_ok else "Aluminum/Polymer Melting Recycle"
        })
        
        return pd.DataFrame(extracted)
