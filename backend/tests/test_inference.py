"""Run from the project root: python -m unittest discover -s backend/tests -v."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inference


SENSORS = ["volt", "rotate", "pressure", "vibration"]


class InferenceTests(unittest.TestCase):
    def test_runtime_features_match_training_rolling_and_shift_for_every_machine(self):
        # Independently reproduce training's grouped rolling/shift operations.
        current = inference.telemetry.groupby("machineID", sort=False).tail(1).copy()
        current["datetime"] += pd.Timedelta(hours=1)
        frame = pd.concat([inference.telemetry, current], ignore_index=True)
        frame = frame.sort_values(["machineID", "datetime"]).reset_index(drop=True)
        for sensor in SENSORS:
            grouped = frame.groupby("machineID")[sensor]
            frame[f"{sensor}_24h_mean"] = grouped.rolling(24).mean().droplevel(0)
            frame[f"{sensor}_24h_std"] = grouped.rolling(24).std().droplevel(0)
            frame[f"{sensor}_6h_change"] = frame[sensor] - grouped.shift(6)
            frame[f"{sensor}_vs_24h_mean"] = frame[sensor] - frame[f"{sensor}_24h_mean"]

        expected_rows = frame.groupby("machineID", sort=False).tail(1)
        expected_probabilities = []
        actual_probabilities = []
        for _, row in expected_rows.iterrows():
            machine_id = int(row["machineID"])
            with self.subTest(machine_id=machine_id):
                actual = inference.build_machine_features(machine_id, **row[SENSORS].to_dict())
                self.assertEqual(list(actual.columns), list(inference.feature_columns))
                self.assertEqual(actual.shape, (1, 23))
                row["age"] = inference.machines.set_index("machineID").loc[machine_id, "age"]
                error_times = inference.errors.loc[
                    inference.errors["machineID"] == machine_id, "datetime"
                ]
                row["errors_last_24h"] = error_times.searchsorted(row["datetime"], side="right") - error_times.searchsorted(
                    row["datetime"] - pd.Timedelta(hours=24), side="right"
                )
                maintenance_times = inference.maint.loc[
                    inference.maint["machineID"] == machine_id, "datetime"
                ]
                previous = maintenance_times.searchsorted(row["datetime"], side="right") - 1
                row["hours_since_last_maintenance"] = (
                    (row["datetime"] - maintenance_times.iloc[previous]).total_seconds() / 3600
                    if previous >= 0 else np.nan
                )
                expected = pd.DataFrame([row.loc[inference.feature_columns].to_dict()])
                pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-10, atol=1e-10)
                for features, probabilities in [(actual, actual_probabilities), (expected, expected_probabilities)]:
                    prepared = pd.DataFrame(
                        inference.feature_imputer.transform(features), columns=inference.feature_columns
                    ).astype("float32")
                    probabilities.append(inference.smote_xgb.predict_proba(prepared)[0, 1])
        np.testing.assert_array_equal(actual_probabilities, expected_probabilities)

    def test_short_history_is_rejected_instead_of_fabricating_features(self):
        for rows in [1, 5, 6, 22]:
            with self.subTest(rows=rows), patch.object(
                inference, "telemetry", inference.telemetry.query("machineID == 1").tail(rows)
            ):
                with self.assertRaisesRegex(ValueError, "23 stored hourly"):
                    inference.build_machine_features(1, 170, 450, 100, 40)

    def test_history_gaps_and_duplicates_are_rejected(self):
        history = inference.telemetry.query("machineID == 1").copy().reset_index(drop=True)
        for offset in [-1, 1]:
            broken = history.copy()
            broken.loc[0, "datetime"] += pd.Timedelta(hours=offset)
            with self.subTest(offset=offset), patch.object(inference, "telemetry", broken):
                with self.assertRaisesRegex(ValueError, "consecutive hourly"):
                    inference.build_machine_features(1, 170, 450, 100, 40)

    def test_other_machines_and_input_row_order_do_not_change_features(self):
        expected = inference.build_machine_features(1, 170, 450, 100, 40)
        changed = inference.telemetry.copy()
        other = changed["machineID"] != 1
        changed.loc[other, SENSORS] = 99999
        changed.loc[other, "datetime"] += pd.Timedelta(days=100)
        with patch.object(inference, "telemetry", changed.sample(frac=1, random_state=42)):
            actual = inference.build_machine_features(1, 170, 450, 100, 40)
        pd.testing.assert_frame_equal(actual, expected)

    def test_event_boundaries_and_machine_isolation(self):
        timestamp = inference.telemetry.query("machineID == 1")["datetime"].max() + pd.Timedelta(hours=1)
        errors = pd.DataFrame({
            "machineID": [1, 1, 1, 1, 1, 2],
            "datetime": [timestamp - pd.Timedelta(hours=24), timestamp - pd.Timedelta(hours=23),
                         timestamp, timestamp, timestamp + pd.Timedelta(hours=1), timestamp],
        })
        maintenance = pd.DataFrame({
            "machineID": [1, 1, 2],
            "datetime": [timestamp - pd.Timedelta(hours=7), timestamp + pd.Timedelta(hours=1), timestamp],
        })
        with patch.object(inference, "errors", errors), patch.object(inference, "maint", maintenance):
            features = inference.build_machine_features(1, 170, 450, 100, 40).iloc[0]
        self.assertEqual(features["errors_last_24h"], 3)
        self.assertEqual(features["hours_since_last_maintenance"], 7)
        maintenance.loc[0, "datetime"] = timestamp
        with patch.object(inference, "maint", maintenance):
            features = inference.build_machine_features(1, 170, 450, 100, 40).iloc[0]
        self.assertEqual(features["hours_since_last_maintenance"], 0)

    def test_missing_maintenance_uses_saved_imputer(self):
        with patch.object(inference, "maint", inference.maint.iloc[:0]):
            features = inference.build_machine_features(1, 170, 450, 100, 40)
        self.assertTrue(pd.isna(features.loc[0, "hours_since_last_maintenance"]))
        position = list(inference.feature_columns).index("hours_since_last_maintenance")
        prepared = inference.feature_imputer.transform(features)
        self.assertEqual(prepared[0, position], inference.feature_imputer.statistics_[position])

    def test_saved_schema_mismatch_is_rejected(self):
        inference.validate_failure_schema()
        with patch.object(inference, "feature_columns", list(reversed(inference.feature_columns))):
            with self.assertRaisesRegex(RuntimeError, "same order"):
                inference.validate_failure_schema()

    def test_missing_engineered_feature_is_not_silently_imputed(self):
        columns = list(inference.feature_columns)
        columns[-1] = "unknown_feature"
        with patch.object(inference, "feature_columns", columns):
            with self.assertRaisesRegex(RuntimeError, "feature schema"):
                inference.build_machine_features(1, 170, 450, 100, 40)

    def test_notebook_positive_example_and_diagnostic_logs(self):
        with self.assertLogs(inference.logger, level="INFO") as logs:
            result = inference.predict_machine_health(
                machine_id=58, vibration=2.25, rotate=268.74, volt=218.06, pressure=230.66
            )
        self.assertEqual(result["failure"]["prediction"], "Yes")
        self.assertGreater(result["failure"]["risk"], result["failure"]["threshold"])
        self.assertEqual(set(result), {"machine_id", "failure", "anomaly", "rul", "explanation"})
        output = "\n".join(logs.output)
        for field in ["prediction_time=", "history_rows=23", "engineered_features=", "feature_order=",
                      "final_model_features=", "predicted_failure_probability="]:
            self.assertIn(field, output)
        for feature in inference.feature_columns:
            self.assertIn(feature, output)


if __name__ == "__main__":
    unittest.main()
