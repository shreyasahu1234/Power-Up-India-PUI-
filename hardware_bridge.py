"""
REUSEGRID AI — Hardware-to-Software Bridge
Interfaces with the physical ESP32 monitoring unit (INA219 voltage/current + DS18B20 temperature).
Supports live USB Serial streaming and automatic fallback to live hardware emulation.
"""

import os
import json
import time
import random
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class HardwareBridge:
    """
    Bridge connecting physical sensor hardware (ESP32 + INA219 + DS18B20)
    to the REUSEGRID AI analytics engine.
    """

    def __init__(self, port=None, baud_rate=115200, timeout=1.0):
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.serial_conn = None
        self.mode = "SIMULATION"
        self._init_connection()

    def _init_connection(self):
        """Attempts to auto-detect and open an ESP32 USB serial connection."""
        if not SERIAL_AVAILABLE:
            self.mode = "EMULATION"
            return

        target_port = self.port
        if target_port is None:
            # Auto-detect available COM ports
            ports = list(serial.tools.list_ports.comports())
            for p in ports:
                desc = (p.description or "").lower()
                hwid = (p.hwid or "").lower()
                if "cp210" in desc or "ch340" in desc or "usb-serial" in desc or "esp32" in desc or "ch340" in hwid:
                    target_port = p.device
                    break

        if target_port:
            try:
                self.serial_conn = serial.Serial(target_port, self.baud_rate, timeout=self.timeout)
                time.sleep(1.5)  # Allow ESP32 reboot upon DTR trigger
                self.port = target_port
                self.mode = "PHYSICAL_SERIAL"
                print(f"[HardwareBridge] Connected to physical ESP32 on {target_port} @ {self.baud_rate} baud.")
                return
            except Exception as e:
                print(f"[HardwareBridge] Notice: Could not open {target_port} ({e}). Falling back to Live Emulation.")

        self.mode = "EMULATION"

    def list_channels(self):
        """Returns the list of monitored battery channels."""
        return ["CH1", "CH2", "CH3"]

    def read_live_telemetry(self, channel="CH1"):
        """
        Reads instantaneous telemetry for a chosen battery channel.
        Returns:
            dict containing measured voltage, current, temperature, thermal rate, and impedance.
        """
        if self.mode == "PHYSICAL_SERIAL" and self.serial_conn and self.serial_conn.is_open:
            try:
                # Send polling command to ESP32
                cmd = f"READ {channel}\n"
                self.serial_conn.write(cmd.encode("utf-8"))
                line = self.serial_conn.readline().decode("utf-8").strip()
                if line.startswith("{") and line.endswith("}"):
                    data = json.loads(line)
                    data["source"] = f"ESP32_HARDWARE ({self.port})"
                    data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    return data
            except Exception as e:
                print(f"[HardwareBridge] Serial read warning: {e}. Defaulting to calibrated channel sensor.")

        # High-Fidelity Emulation of the 3 Physical Hardware Demo Channels
        return self._get_calibrated_channel_reading(channel)

    def _get_calibrated_channel_reading(self, channel):
        """
        Generates grounded, calibrated live telemetry matching the physical ideathon prototype setup:
        - CH1: Healthy Li-ion 3S pack (Nominal ~11.1V, Intact, Low IR)
        - CH2: Repaired candidate LiFePO4 4S module (Nominal ~12.8V, Sensor Drift / BMS issue)
        - CH3: Thermal hazard NMC 10S pack (Nominal ~36V, Over-temperature & swelling hazard)
        """
        # Small dynamic sensor noise emulating INA219 ADC / DS18B20 12-bit jitter
        v_jitter = random.gauss(0, 0.02)
        i_jitter = random.gauss(0, 0.01)
        t_jitter = random.gauss(0, 0.1)

        if channel == "CH1":
            return {
                "channel_id": "CH1",
                "hardware_device": "INA219_ADDR_0x40 + DS18B20_PIN4",
                "measured_voltage": round(11.22 + v_jitter, 2),
                "measured_current": round(1.24 + i_jitter, 2),
                "operating_temperature": round(27.4 + t_jitter, 1),
                "temp_rise_rate": round(0.22 + random.uniform(-0.02, 0.02), 2),
                "internal_resistance_mOhm": round(54.5 + random.uniform(-1.0, 1.0), 1),
                "physical_damage_observed": 0,
                "swelling_observed": "None (0-1%)",
                "leakage_observed": "None",
                "source": "EMULATED_ESP32_CH1 (Calibrated Safe Pack)",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

        elif channel == "CH2":
            return {
                "channel_id": "CH2",
                "hardware_device": "INA219_ADDR_0x41 + DS18B20_PIN5",
                "measured_voltage": round(12.88 + v_jitter, 2),
                "measured_current": round(1.12 + i_jitter, 2),
                "operating_temperature": round(31.2 + t_jitter, 1),
                "temp_rise_rate": round(0.32 + random.uniform(-0.02, 0.02), 2),
                "internal_resistance_mOhm": round(63.2 + random.uniform(-1.0, 1.0), 1),
                "physical_damage_observed": 0,
                "swelling_observed": "None (0-1%)",
                "leakage_observed": "None",
                "source": "EMULATED_ESP32_CH2 (Calibrated Sensor/BMS Issue Pack)",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

        elif channel == "CH3":
            return {
                "channel_id": "CH3",
                "hardware_device": "INA219_ADDR_0x44 + DS18B20_PIN6",
                "measured_voltage": round(24.20 + v_jitter, 2),  # Under-voltage for 10S
                "measured_current": round(2.05 + i_jitter, 2),
                "operating_temperature": round(52.8 + t_jitter, 1),  # Over-temp (>48°C)
                "temp_rise_rate": round(2.65 + random.uniform(-0.05, 0.05), 2),
                "internal_resistance_mOhm": round(186.0 + random.uniform(-2.0, 2.0), 1),
                "physical_damage_observed": 1,
                "swelling_observed": "Severe Swelling (>6%)",
                "leakage_observed": "None",
                "source": "EMULATED_ESP32_CH3 (Calibrated Thermal Hazard Pack)",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

        else:
            # Custom / Arbitrary probe channel
            return {
                "channel_id": channel,
                "hardware_device": "INA219_GENERIC + DS18B20",
                "measured_voltage": round(7.45 + v_jitter, 2),
                "measured_current": round(1.0 + i_jitter, 2),
                "operating_temperature": round(26.8 + t_jitter, 1),
                "temp_rise_rate": round(0.18 + random.uniform(-0.02, 0.02), 2),
                "internal_resistance_mOhm": round(44.0 + random.uniform(-1.0, 1.0), 1),
                "physical_damage_observed": 0,
                "swelling_observed": "None (0-1%)",
                "leakage_observed": "None",
                "source": "EMULATED_ESP32_CUSTOM",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

    def close(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
