"""
REUSEGRID AI — Case-Based Reasoning & Similarity Matching Engine
Bridges physical hardware readings (INA219 + DS18B20) with the comprehensive 500-battery dataset.
Identifies the most similar historical archetype, infers degradation history, and enriches the live battery profile.
"""

import os
import numpy as np
import pandas as pd


class BatterySimilarityMatcher:
    """
    Finds the most similar battery case in the benchmark dataset for any live hardware reading.
    Enriches raw electrical/thermal sensor data with historical, chemical, and diagnostic categories.
    """

    def __init__(self, dataset_path=None):
        if dataset_path is None:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            proj_dir = os.path.dirname(curr_dir)
            dataset_path = os.path.join(proj_dir, "data", "batteries.csv")
            if not os.path.exists(dataset_path):
                dataset_path = os.path.join(proj_dir, "data", "REUSEGRID_AI_500_Battery_Test_Data.csv")

        self.dataset_path = dataset_path
        self.df_bench = pd.read_csv(self.dataset_path)
        self._prepare_feature_space()

    def _prepare_feature_space(self):
        """Precomputes normalized feature vectors for all 500 benchmark battery records."""
        df = self.df_bench

        # Numerical telemetry features to compare
        self.v_vals = df["Measured_Voltage_V"].values.astype(float)
        self.i_vals = df["Measured_Current_A"].values.astype(float)
        self.t_vals = df["Operating_Temperature_C"].values.astype(float)
        self.trise_vals = df["Temp_Rise_Rate_C_per_min"].values.astype(float)
        self.ir_vals = df["Internal_Resistance_mOhm"].values.astype(float)
        self.vnom_vals = df["Nominal_Voltage_V"].values.astype(float)

        # Min-Max ranges for fair distance normalization
        self.v_range = max(1.0, np.ptp(self.v_vals))
        self.i_range = max(0.5, np.ptp(self.i_vals))
        self.t_range = max(10.0, np.ptp(self.t_vals))
        self.trise_range = max(1.0, np.ptp(self.trise_vals))
        self.ir_range = max(20.0, np.ptp(self.ir_vals))
        self.vnom_range = max(1.0, np.ptp(self.vnom_vals))

    def find_most_similar_case(self, live_telemetry, top_k=3):
        """
        Matches a live battery hardware reading against the 500-battery database.

        Args:
            live_telemetry (dict): Must contain 'measured_voltage', 'measured_current',
                                   'operating_temperature', 'temp_rise_rate', 'internal_resistance_mOhm'
            top_k (int): Number of nearest neighbors to return.

        Returns:
            dict containing best_match, similarity_score_pct, top_k_list, and enriched_battery profile.
        """
        v_live = float(live_telemetry.get("measured_voltage", 12.0))
        i_live = float(live_telemetry.get("measured_current", 1.2))
        t_live = float(live_telemetry.get("operating_temperature", 28.0))
        trise_live = float(live_telemetry.get("temp_rise_rate", 0.2))
        ir_live = float(live_telemetry.get("internal_resistance_mOhm", 50.0))

        # Estimate closest nominal voltage tier if not explicitly supplied
        candidate_tiers = [7.4, 11.1, 12.8, 14.8, 18.0, 36.0]
        closest_vnom = min(candidate_tiers, key=lambda nom: abs(nom - v_live))
        vnom_live = float(live_telemetry.get("nominal_voltage", closest_vnom))

        # Weighted Normalized Euclidean Distance
        # Weights: Voltage (0.35), Internal Resistance (0.25), Temp & Rise (0.25), Current (0.15)
        d_v = ((self.v_vals - v_live) / self.v_range) ** 2
        d_vnom = ((self.vnom_vals - vnom_live) / self.vnom_range) ** 2
        d_ir = ((self.ir_vals - ir_live) / self.ir_range) ** 2
        d_t = ((self.t_vals - t_live) / self.t_range) ** 2
        d_trise = ((self.trise_vals - trise_live) / self.trise_range) ** 2
        d_i = ((self.i_vals - i_live) / self.i_range) ** 2

        # Combined weighted distance
        total_dist_sq = (
            (0.20 * d_v) +
            (0.20 * d_vnom) +
            (0.25 * d_ir) +
            (0.15 * d_t) +
            (0.10 * d_trise) +
            (0.10 * d_i)
        )
        distances = np.sqrt(total_dist_sq)

        # Rank cases
        sorted_indices = np.argsort(distances)
        best_idx = sorted_indices[0]
        best_row = self.df_bench.iloc[best_idx].to_dict()

        # Convert distance to similarity percentage (100% = identical)
        best_dist = distances[best_idx]
        best_similarity_pct = round(float(np.clip(100.0 * (1.0 - (best_dist / 1.414)), 50.0, 99.8)), 1)

        # Top K Neighbors
        top_k_list = []
        for rank, idx in enumerate(sorted_indices[:top_k], start=1):
            row_k = self.df_bench.iloc[idx]
            sim_k = round(float(np.clip(100.0 * (1.0 - (distances[idx] / 1.414)), 50.0, 99.8)), 1)
            top_k_list.append({
                "rank": rank,
                "battery_id": row_k["Battery_ID"],
                "battery_type": row_k["Battery_Type"],
                "pack_config": row_k["Pack_Configuration"],
                "nominal_voltage": row_k["Nominal_Voltage_V"],
                "historical_cycles": int(row_k["Historical_Cycles"]),
                "measured_v": row_k["Measured_Voltage_V"],
                "measured_t": row_k["Operating_Temperature_C"],
                "measured_ir": row_k["Internal_Resistance_mOhm"],
                "similarity_score_pct": sim_k,
                "primary_diagnosed_issue": row_k.get("Primary_Diagnosed_Issue", "None")
            })

        # Enrich the live physical battery with matched categories
        channel_name = live_telemetry.get("channel_id", "LIVE_PROBE")
        enriched_battery = {
            "Battery_ID": f"{channel_name}_{best_row['Battery_ID']}",
            "Physical_Channel": channel_name,
            "Battery_Type": best_row["Battery_Type"],
            "Source_Origin": best_row.get("Source_Origin", "Field Retrieval Fleet"),
            "Pack_Configuration": best_row["Pack_Configuration"],
            "Nominal_Voltage_V": best_row["Nominal_Voltage_V"],
            "Nominal_Capacity_Ah": best_row.get("Nominal_Capacity_Ah", 5.0),
            "Nominal_Energy_Wh": best_row.get("Nominal_Energy_Wh", 60.0),
            "Historical_Cycles": int(best_row.get("Historical_Cycles", 500)),
            "Estimated_Age_Years": float(best_row.get("Estimated_Age_Years", 2.5)),
            # Use the live physical sensor readings for evaluation
            "Measured_Voltage_V": v_live,
            "Measured_Current_A": i_live,
            "Operating_Temperature_C": t_live,
            "Temp_Rise_Rate_C_per_min": trise_live,
            "Internal_Resistance_mOhm": ir_live,
            "Physical_Casing_Condition": live_telemetry.get("physical_casing", best_row.get("Physical_Casing_Condition", "Intact (Good)")),
            "Swelling_Deformation": live_telemetry.get("swelling_observed", best_row.get("Swelling_Deformation", "None (0-1%)")),
            "Electrolyte_Leakage_Observation": live_telemetry.get("leakage_observed", best_row.get("Electrolyte_Leakage_Observation", "None")),
            "Primary_Diagnosed_Issue": best_row.get("Primary_Diagnosed_Issue", "None (Fully Functional)"),
            "Matched_Benchmark_ID": best_row["Battery_ID"],
            "Case_Similarity_Percent": best_similarity_pct,
            "Hardware_Source": live_telemetry.get("source", "ESP32_INA219_DS18B20")
        }

        return {
            "best_match_id": best_row["Battery_ID"],
            "similarity_score_pct": best_similarity_pct,
            "top_k_matches": top_k_list,
            "enriched_battery": enriched_battery
        }
