import numpy as np
import pandas as pd

from sepsis_vitals import build_feature_set
from sepsis_vitals.data_quality import summarize_vitals_quality
from sepsis_vitals.features import get_feature_inventory


def main() -> None:
    raw = pd.DataFrame(
        {
            "patient_id": ["P001", "P001", "P001", "P002", "P002"],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 08:00",
                    "2026-01-01 10:00",
                    "2026-01-01 12:00",
                    "2026-01-01 09:00",
                    "2026-01-01 11:00",
                ]
            ),
            "temperature": [37.1, 38.4, 39.1, 36.8, np.nan],
            "heart_rate": [88, 95, 124, 76, 102],
            "resp_rate": [18, 23, 28, 16, 22],
            "sbp": [118, 99, 88, 130, 105],
            "spo2": [97, 94, 91, 98, 95],
            "gcs": [15, 14, 13, 15, 15],
            "age_years": [45, 45, 45, 72, 72],
        }
    )

    quality = summarize_vitals_quality(raw)
    features = build_feature_set(raw)
    inventory = get_feature_inventory(features)

    print(f"Input shape: {raw.shape}")
    print(f"Rows with all six vitals: {quality['rows_with_all_six_vitals_rate']:.0%}")
    print(f"Feature shape: {features.shape}")
    print(features[["patient_id", "qsofa_score", "shock_index", "n_vitals_missing"]])
    print(f"Catalogued engineered features: {len(inventory)}")


if __name__ == "__main__":
    main()
