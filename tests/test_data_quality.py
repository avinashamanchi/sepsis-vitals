import numpy as np
import pandas as pd

from sepsis_vitals.data_quality import summarize_vitals_quality


def test_summarize_vitals_quality_reports_missingness_and_ranges():
    df = pd.DataFrame(
        {
            "patient_id": ["A", "A", "B"],
            "timestamp": pd.to_datetime(
                ["2026-01-01 08:00", "2026-01-01 10:00", "2026-01-01 11:00"]
            ),
            "temperature": [37.0, np.nan, 39.0],
            "heart_rate": [80, 120, 300],
            "resp_rate": [18, 24, 30],
            "sbp": [120, 95, 88],
            "spo2": [98, np.nan, 91],
            "gcs": [15, 14, 13],
        }
    )

    report = summarize_vitals_quality(df)

    assert report["n_rows"] == 3
    assert report["n_patients"] == 2
    assert report["vitals"]["temperature"]["missing_rate"] == 1 / 3
    assert report["vitals"]["spo2"]["missing_rate"] == 1 / 3
    assert report["vitals"]["heart_rate"]["implausible_count"] == 1
    assert report["rows_with_all_six_vitals_rate"] == 2 / 3
    assert report["rows_with_all_six_plausible_vitals_rate"] == 1 / 3
