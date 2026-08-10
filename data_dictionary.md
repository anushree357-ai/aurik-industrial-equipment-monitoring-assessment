# Aurik Technologies - Assessment Dataset Data Dictionary

This dataset is created for the Data Scientist / AI-ML Founding Employee assessment. API responses and data samples are dummy samples created for assessment purposes.

## Files

| File | Rows | Purpose |
|---|---:|---|
| asset_master.csv | 12 | Machine reference data used for joining and contextual interpretation. |
| vendor_a_sensor_events.csv | 740 | Sample MechaPulse automated sensor/event data. |
| vendor_b_alert_events.csv | 742 | Sample ThermoTrack alert/reading data. |
| vendor_c_inspection_maintenance.csv | 493 | Sample MaintaLogix inspection, maintenance, calibration, and operator-note data. |
| maintenance_records.csv | 160 | Work-order style maintenance records. |
| ground_truth_outcomes.csv | 100 | Dummy outcome records that may be used for future ML discussion. |
| alert_type_dictionary.csv | 16 | Raw vendor type dictionary. It is not a complete normalized mapping. |

## Notes for candidates

- Treat the data as external vendor data.
- Preserve raw identifiers when designing normalized schemas.
- Make assumptions explicit.
- Do not assume all rows are clean, complete, unique, or directly comparable.

## Column summary

### asset_master.csv
- machine_id: Machine identifier.
- plant_id: Plant/site identifier.
- line_id: Production line identifier.
- machine_type: Machine category.
- criticality: Business/production criticality.
- installed_date: Date machine was installed.
- oem: Equipment manufacturer label.
- asset_status: Current asset status.
- service_interval_days: Nominal service interval.
- rated_max_temp_c: Reference max temperature in Celsius.
- rated_max_vibration_mm_s: Reference max vibration in mm/s.
- baseline_power_kw: Approximate baseline power draw.

### vendor_a_sensor_events.csv
Key fields include event_id, machine_id, line, event_time, ingestion_time, event_type, severity, vibration_mm_s, temperature_c, machine_status, sensor_health, confidence, operating_hours_today.

### vendor_b_alert_events.csv
Key fields include readingId, assetCode, productionLine, timestampMs, receivedAtEpochMs, alertCode, level, vibration_g, temperature_f, power_kw, is_active, signal_quality, vendorConfidencePct.

### vendor_c_inspection_maintenance.csv
Key fields include record_id, machine_ref, line_id, recorded_at, record_type, inspection_result, maintenance_status, days_since_last_service, technician_note, manual_confidence.

### maintenance_records.csv
Key fields include work_order_id, machine_id, opened_at, closed_at, maintenance_type, trigger_source, status, downtime_minutes, parts_replaced, technician_confidence, post_maintenance_check.

### ground_truth_outcomes.csv
Key fields include outcome_id, machine_id, outcome_start_at, outcome_end_at, outcome_type, downtime_minutes, defect_rate_pct, confirmed_root_cause, label_window_days.
