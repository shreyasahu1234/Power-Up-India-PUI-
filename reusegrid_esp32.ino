/*
 * REUSEGRID AI -- ESP32 Hardware Telemetry Firmware
 * =====================================================
 * Reads 3 battery channels simultaneously:
 *   CH1 -> INA219 @ 0x40  + DS18B20 sensor index 0
 *   CH2 -> INA219 @ 0x41  + DS18B20 sensor index 1
 *   CH3 -> INA219 @ 0x44  + DS18B20 sensor index 2
 *
 * Streams JSON telemetry over USB Serial at 115200 baud.
 * Python HardwareBridge reads this stream for live pipeline input.
 *
 * Wiring:
 *   INA219 SDA  -> GPIO 21 (ESP32 I2C SDA)
 *   INA219 SCL  -> GPIO 22 (ESP32 I2C SCL)
 *   DS18B20 DQ  -> GPIO 4  (OneWire bus, single wire, 4.7k pull-up to 3.3V)
 *   DS18B20 VCC -> 3.3V
 *   DS18B20 GND -> GND
 *
 * Libraries Required (install via Arduino Library Manager):
 *   - Adafruit INA219        (by Adafruit)
 *   - DallasTemperature      (by Miles Burton)
 *   - OneWire                (by Jim Studt)
 *
 * Upload Settings:
 *   Board: ESP32 Dev Module
 *   Upload Speed: 921600
 *   CPU Frequency: 240 MHz
 *   Partition Scheme: Default 4MB with spiffs
 */

#include <Wire.h>
#include <Adafruit_INA219.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <ArduinoJson.h>

// ============================================================
// CONFIGURATION
// ============================================================

#define ONE_WIRE_BUS   4        // DS18B20 data pin
#define TELEMETRY_INTERVAL_MS 2000   // Publish every 2 seconds
#define SERIAL_BAUD    115200

// Internal resistance estimation window (in milli-ohms)
// INA219 cannot directly measure IR; we estimate via delta-V / delta-I
// This requires two consecutive readings. For prototype: static fallback.
#define IR_FALLBACK_MOHM 60.0

// ============================================================
// HARDWARE OBJECTS
// ============================================================

Adafruit_INA219 ina_ch1(0x40);
Adafruit_INA219 ina_ch2(0x41);
Adafruit_INA219 ina_ch3(0x44);

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature tempSensors(&oneWire);

// Per-channel previous readings for temperature velocity calculation
float prev_temp[3] = {25.0, 25.0, 25.0};
unsigned long prev_time_ms = 0;

// ============================================================
// SETUP
// ============================================================

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(500);

  Wire.begin();

  // Initialize INA219 sensors
  if (!ina_ch1.begin()) {
    Serial.println("{\"error\": \"INA219 CH1 (0x40) not found. Check wiring.\"}");
  }
  if (!ina_ch2.begin()) {
    Serial.println("{\"error\": \"INA219 CH2 (0x41) not found. Check wiring.\"}");
  }
  if (!ina_ch3.begin()) {
    Serial.println("{\"error\": \"INA219 CH3 (0x44) not found. Check wiring.\"}");
  }

  // Initialize DS18B20 temperature sensors
  tempSensors.begin();
  int numSensors = tempSensors.getDeviceCount();
  Serial.print("{\"status\": \"boot\", \"ds18b20_count\": ");
  Serial.print(numSensors);
  Serial.println("}");

  // Set INA219 calibration for 32V / 2A range (suitable for 3S Li-ion packs)
  // For higher voltage packs: use setCalibration_32V_2A() or custom calibration
  ina_ch1.setCalibration_32V_2A();
  ina_ch2.setCalibration_32V_2A();
  ina_ch3.setCalibration_32V_2A();

  prev_time_ms = millis();
}

// ============================================================
// HELPER: Safe temperature read with fallback
// ============================================================

float readTemp(int sensorIndex) {
  DeviceAddress addr;
  if (!tempSensors.getAddress(addr, sensorIndex)) {
    return 25.0;  // Fallback: ambient if sensor missing
  }
  float t = tempSensors.getTempC(addr);
  if (t == DEVICE_DISCONNECTED_C) return 25.0;
  return t;
}

// ============================================================
// MAIN LOOP
// ============================================================

void loop() {
  unsigned long now_ms = millis();

  // Trigger DS18B20 conversion (blocking 750ms at 12-bit resolution)
  tempSensors.requestTemperatures();

  // Read all 3 INA219 channels
  float v1 = ina_ch1.getBusVoltage_V() + (ina_ch1.getShuntVoltage_mV() / 1000.0);
  float i1 = ina_ch1.getCurrent_mA() / 1000.0;  // Convert to Amperes

  float v2 = ina_ch2.getBusVoltage_V() + (ina_ch2.getShuntVoltage_mV() / 1000.0);
  float i2 = ina_ch2.getCurrent_mA() / 1000.0;

  float v3 = ina_ch3.getBusVoltage_V() + (ina_ch3.getShuntVoltage_mV() / 1000.0);
  float i3 = ina_ch3.getCurrent_mA() / 1000.0;

  // Read temperatures
  float t1 = readTemp(0);
  float t2 = readTemp(1);
  float t3 = readTemp(2);

  // Compute thermal rise velocity (deg C / min)
  float dt_min = (now_ms - prev_time_ms) / 60000.0;
  if (dt_min < 0.001) dt_min = 0.001;  // Avoid division by zero

  float tr1 = (t1 - prev_temp[0]) / dt_min;
  float tr2 = (t2 - prev_temp[1]) / dt_min;
  float tr3 = (t3 - prev_temp[2]) / dt_min;

  // Clamp negative thermal velocities to 0 (cooling is not a hazard trigger)
  tr1 = max(tr1, 0.0f);
  tr2 = max(tr2, 0.0f);
  tr3 = max(tr3, 0.0f);

  // Update history
  prev_temp[0] = t1;
  prev_temp[1] = t2;
  prev_temp[2] = t3;
  prev_time_ms = now_ms;

  // ============================================================
  // PUBLISH JSON TELEMETRY FRAME
  // Each JSON object is one line — HardwareBridge reads line-by-line.
  // Format matches exactly what HardwareBridge.read_live_telemetry() expects.
  // ============================================================

  // CH1
  Serial.print("{");
  Serial.print("\"channel\":\"CH1\",");
  Serial.print("\"measured_voltage\":"); Serial.print(v1, 3); Serial.print(",");
  Serial.print("\"measured_current\":"); Serial.print(i1, 3); Serial.print(",");
  Serial.print("\"operating_temperature\":"); Serial.print(t1, 2); Serial.print(",");
  Serial.print("\"temp_rise_rate\":"); Serial.print(tr1, 3); Serial.print(",");
  Serial.print("\"internal_resistance_mOhm\":"); Serial.print(IR_FALLBACK_MOHM); Serial.print(",");
  Serial.print("\"swelling_observed\":false,");
  Serial.print("\"leakage_observed\":false,");
  Serial.print("\"physical_damage_observed\":false,");
  Serial.print("\"source\":\"ESP32_CH1\"");
  Serial.println("}");

  // CH2
  Serial.print("{");
  Serial.print("\"channel\":\"CH2\",");
  Serial.print("\"measured_voltage\":"); Serial.print(v2, 3); Serial.print(",");
  Serial.print("\"measured_current\":"); Serial.print(i2, 3); Serial.print(",");
  Serial.print("\"operating_temperature\":"); Serial.print(t2, 2); Serial.print(",");
  Serial.print("\"temp_rise_rate\":"); Serial.print(tr2, 3); Serial.print(",");
  Serial.print("\"internal_resistance_mOhm\":"); Serial.print(IR_FALLBACK_MOHM); Serial.print(",");
  Serial.print("\"swelling_observed\":false,");
  Serial.print("\"leakage_observed\":false,");
  Serial.print("\"physical_damage_observed\":false,");
  Serial.print("\"source\":\"ESP32_CH2\"");
  Serial.println("}");

  // CH3
  Serial.print("{");
  Serial.print("\"channel\":\"CH3\",");
  Serial.print("\"measured_voltage\":"); Serial.print(v3, 3); Serial.print(",");
  Serial.print("\"measured_current\":"); Serial.print(i3, 3); Serial.print(",");
  Serial.print("\"operating_temperature\":"); Serial.print(t3, 2); Serial.print(",");
  Serial.print("\"temp_rise_rate\":"); Serial.print(tr3, 3); Serial.print(",");
  Serial.print("\"internal_resistance_mOhm\":"); Serial.print(IR_FALLBACK_MOHM); Serial.print(",");
  Serial.print("\"swelling_observed\":false,");
  Serial.print("\"leakage_observed\":false,");
  Serial.print("\"physical_damage_observed\":false,");
  Serial.print("\"source\":\"ESP32_CH3\"");
  Serial.println("}");

  delay(TELEMETRY_INTERVAL_MS);
}

/*
 * ============================================================
 * HOW TO CONNECT REAL HARDWARE TO REUSEGRID AI PYTHON BACKEND
 * ============================================================
 *
 * 1. Upload this firmware to your ESP32.
 * 2. Connect ESP32 via USB to the PC running the Python backend.
 * 3. The HardwareBridge will auto-detect the COM port (looks for
 *    "CP210x", "CH340", "USB-SERIAL", or "ESP32" in Windows Device Manager).
 * 4. Run:
 *      python -m backend.main CH1
 *      python -m backend.main CH2
 *      python -m backend.main CH3
 *      python -m backend.main --all
 *
 * 5. HardwareBridge reads the latest JSON line matching the requested channel
 *    and feeds voltage, current, temperature, and thermal velocity into the
 *    similarity engine and safety pipeline.
 *
 * NOTES ON INTERNAL RESISTANCE:
 *   INA219 does not directly measure AC impedance. For production:
 *   - Use EIS (Electrochemical Impedance Spectroscopy) hardware, OR
 *   - Use pulse discharge method: dV/dI during load pulse gives DC-IR.
 *   This firmware uses a fixed fallback (IR_FALLBACK_MOHM = 60.0).
 *   Replace with measured value if you add a dedicated IR measurement circuit.
 *
 * SAFETY NOTE:
 *   This prototype assumes low-voltage (< 25V), protected battery packs.
 *   Do NOT use unprotected Li-ion cells with this prototype without adding
 *   a proper hardware safety cutoff (fuse, relay, BMS watchdog).
 * ============================================================
 */
