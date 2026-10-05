"""
REUSEGRID AI -- Similarity Engine Wrapper (backend/similarity.py)
"""

from backend.similarity_matcher import BatterySimilarityMatcher

_matcher = None

def get_matcher():
    global _matcher
    if _matcher is None:
        _matcher = BatterySimilarityMatcher()
    return _matcher

def find_similar_cases(battery, top_k=3):
    matcher = get_matcher()
    telemetry = {
        "measured_voltage": battery.get("voltage", battery.get("Measured_Voltage_V", 12.0)),
        "measured_current": battery.get("current", battery.get("Measured_Current_A", 1.2)),
        "operating_temperature": battery.get("temperature", battery.get("Operating_Temperature_C", 28.0)),
        "temp_rise_rate": battery.get("temp_rise_rate", battery.get("Temp_Rise_Rate_C_per_min", 0.2)),
        "internal_resistance_mOhm": battery.get("internal_resistance_mOhm", battery.get("Internal_Resistance_mOhm", 50.0)),
        "channel_id": battery.get("id", battery.get("Battery_ID", "PROBE"))
    }
    return matcher.find_most_similar_case(telemetry, top_k=top_k)
