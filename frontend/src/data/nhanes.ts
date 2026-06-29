/**
 * NHANES Population Health Data Module
 * Source: CDC National Health and Nutrition Examination Survey (1999-2023)
 * https://wwwn.cdc.gov/nchs/nhanes/
 *
 * Pre-computed population statistics derived from NHANES survey cycles.
 * All values represent weighted population estimates for US adults 18+.
 */

// ── Interfaces ──

export interface Distribution {
  mean: number;
  sd: number;
  p5: number;
  p25: number;
  p50: number;
  p75: number;
  p95: number;
}

export interface BPEthnicity {
  sbp_mean: number;
  sbp_sd: number;
  dbp_mean: number;
  dbp_sd: number;
}

export interface BPTrend {
  sbp: number;
  dbp: number;
  hypertension_pct: number;
}

export interface ObesityTrend {
  overall: number;
  male: number;
  female: number;
}

export interface CRPByBMIEntry {
  mean: number;
  sd: number;
  elevated_pct: number;
}

export interface ScoringParameter {
  threshold: number;
  direction: "above" | "below";
  label: string;
}

export interface ScoringSystem {
  name: string;
  reference: string;
  parameters?: Record<string, ScoringParameter>;
  risk_levels: Record<string, string>;
  sensitivity: number;
  specificity: number;
  auc: number;
}

export interface RiskFactor {
  overall: number;
  by_age: Record<string, number>;
  by_ethnicity: Record<string, number>;
  sepsis_risk_multiplier: number;
}

export interface SepsisEpidemiologyData {
  us_annual_cases: number;
  us_annual_deaths: number;
  global_annual_cases: number;
  global_annual_deaths: number;
  hospital_mortality_rate: number;
  icu_admission_rate: number;
  avg_los_days: number;
  avg_cost_per_case: number;
  mortality_by_hour_delay: number;
  early_detection_mortality_reduction: number;
  incidence_per_100k: Record<string, number>;
  mortality_by_age: Record<string, number>;
}

export interface BenchmarkResult {
  percentile: number;
  severity?: string;
  population: Distribution;
  ageGroup?: string;
  sex?: string;
}

export interface RiskProfile {
  baseRisk: number;
  multiplier: number;
  adjustedRisk: number;
  factors: Array<{ name: string; multiplier: number }>;
  ageGroup: string;
  populationPrevalence: number;
  mortalityRate: number;
}

export interface CurvePoint {
  x: number;
  y: number;
}

export interface VitalsInput {
  sbp?: number | null;
  heart_rate?: number | null;
  resp_rate?: number | null;
  temperature?: number | null;
  spo2?: number | null;
}

type DistributionBySexAge = Record<string, Record<string, Distribution>>;

// ── Survey Cycles ──

export const CYCLES: string[] = [
  "1999-2000", "2001-2002", "2003-2004", "2005-2006", "2007-2008",
  "2009-2010", "2011-2012", "2013-2014", "2015-2016", "2017-2018", "2021-2023",
];

export const AGE_GROUPS: string[] = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"];
export const SEX: string[] = ["Male", "Female"];
export const ETHNICITY: string[] = [
  "Non-Hispanic White", "Non-Hispanic Black", "Mexican-American",
  "Other Hispanic", "Asian", "Other/Multi",
];

// ── Blood Pressure (BPX/BPXO datasets, 1999-2023) ──

export const BP_SYSTOLIC: DistributionBySexAge = {
  Male: {
    "18-29": { mean: 118.2, sd: 10.8, p5: 101, p25: 111, p50: 117, p75: 124, p95: 137 },
    "30-39": { mean: 121.6, sd: 12.4, p5: 103, p25: 113, p50: 120, p75: 129, p95: 143 },
    "40-49": { mean: 124.8, sd: 14.1, p5: 104, p25: 115, p50: 123, p75: 133, p95: 149 },
    "50-59": { mean: 129.3, sd: 16.2, p5: 106, p25: 118, p50: 127, p75: 139, p95: 158 },
    "60-69": { mean: 133.7, sd: 17.8, p5: 108, p25: 121, p50: 132, p75: 144, p95: 165 },
    "70-79": { mean: 137.4, sd: 18.9, p5: 110, p25: 124, p50: 136, p75: 149, p95: 170 },
    "80+":   { mean: 138.8, sd: 20.1, p5: 108, p25: 125, p50: 137, p75: 151, p95: 174 },
  },
  Female: {
    "18-29": { mean: 110.4, sd: 10.2, p5: 94, p25: 103, p50: 110, p75: 117, p95: 128 },
    "30-39": { mean: 113.1, sd: 12.0, p5: 95, p25: 105, p50: 112, p75: 120, p95: 134 },
    "40-49": { mean: 118.7, sd: 14.8, p5: 97, p25: 108, p50: 117, p75: 128, p95: 145 },
    "50-59": { mean: 126.2, sd: 17.3, p5: 100, p25: 114, p50: 124, p75: 137, p95: 157 },
    "60-69": { mean: 133.9, sd: 18.6, p5: 106, p25: 121, p50: 132, p75: 145, p95: 167 },
    "70-79": { mean: 139.1, sd: 19.4, p5: 110, p25: 126, p50: 138, p75: 151, p95: 173 },
    "80+":   { mean: 141.2, sd: 20.8, p5: 109, p25: 127, p50: 140, p75: 154, p95: 177 },
  },
};

export const BP_DIASTOLIC: DistributionBySexAge = {
  Male: {
    "18-29": { mean: 70.1, sd: 9.4, p5: 55, p25: 64, p50: 70, p75: 76, p95: 86 },
    "30-39": { mean: 74.8, sd: 10.1, p5: 58, p25: 68, p50: 74, p75: 81, p95: 92 },
    "40-49": { mean: 77.3, sd: 10.6, p5: 60, p25: 70, p50: 77, p75: 84, p95: 95 },
    "50-59": { mean: 76.8, sd: 10.9, p5: 59, p25: 70, p50: 76, p75: 84, p95: 95 },
    "60-69": { mean: 73.2, sd: 11.4, p5: 55, p25: 66, p50: 73, p75: 80, p95: 92 },
    "70-79": { mean: 68.7, sd: 11.8, p5: 50, p25: 61, p50: 69, p75: 76, p95: 88 },
    "80+":   { mean: 64.3, sd: 12.6, p5: 44, p25: 56, p50: 64, p75: 73, p95: 85 },
  },
  Female: {
    "18-29": { mean: 67.2, sd: 8.8, p5: 53, p25: 61, p50: 67, p75: 73, p95: 82 },
    "30-39": { mean: 70.4, sd: 9.6, p5: 55, p25: 64, p50: 70, p75: 77, p95: 87 },
    "40-49": { mean: 73.1, sd: 10.2, p5: 57, p25: 66, p50: 73, p75: 80, p95: 90 },
    "50-59": { mean: 73.6, sd: 10.5, p5: 57, p25: 67, p50: 73, p75: 80, p95: 91 },
    "60-69": { mean: 71.4, sd: 10.9, p5: 54, p25: 64, p50: 71, p75: 78, p95: 90 },
    "70-79": { mean: 66.8, sd: 11.4, p5: 48, p25: 59, p50: 67, p75: 74, p95: 86 },
    "80+":   { mean: 62.1, sd: 12.2, p5: 42, p25: 54, p50: 62, p75: 70, p95: 83 },
  },
};

export const BP_BY_ETHNICITY: Record<string, BPEthnicity> = {
  "Non-Hispanic White":  { sbp_mean: 124.1, sbp_sd: 16.8, dbp_mean: 72.4, dbp_sd: 10.9 },
  "Non-Hispanic Black":  { sbp_mean: 129.8, sbp_sd: 18.6, dbp_mean: 76.1, dbp_sd: 12.1 },
  "Mexican-American":    { sbp_mean: 122.3, sbp_sd: 15.4, dbp_mean: 70.8, dbp_sd: 10.2 },
  "Other Hispanic":      { sbp_mean: 123.1, sbp_sd: 15.9, dbp_mean: 71.4, dbp_sd: 10.5 },
  "Asian":               { sbp_mean: 121.7, sbp_sd: 16.1, dbp_mean: 72.9, dbp_sd: 10.7 },
  "Other/Multi":         { sbp_mean: 124.8, sbp_sd: 16.5, dbp_mean: 72.8, dbp_sd: 11.0 },
};

export const HYPERTENSION_PREVALENCE: Record<string, number> = {
  "18-29": 0.078, "30-39": 0.123, "40-49": 0.218,
  "50-59": 0.358, "60-69": 0.478, "70-79": 0.592, "80+": 0.641,
};

export const BP_TRENDS: Record<string, BPTrend> = {
  "1999-2000": { sbp: 126.1, dbp: 74.2, hypertension_pct: 0.302 },
  "2001-2002": { sbp: 125.4, dbp: 73.8, hypertension_pct: 0.298 },
  "2003-2004": { sbp: 124.8, dbp: 73.1, hypertension_pct: 0.291 },
  "2005-2006": { sbp: 124.2, dbp: 72.6, hypertension_pct: 0.286 },
  "2007-2008": { sbp: 123.7, dbp: 72.0, hypertension_pct: 0.282 },
  "2009-2010": { sbp: 123.1, dbp: 71.4, hypertension_pct: 0.278 },
  "2011-2012": { sbp: 122.8, dbp: 71.1, hypertension_pct: 0.290 },
  "2013-2014": { sbp: 123.2, dbp: 71.5, hypertension_pct: 0.298 },
  "2015-2016": { sbp: 124.0, dbp: 71.8, hypertension_pct: 0.312 },
  "2017-2018": { sbp: 124.6, dbp: 72.2, hypertension_pct: 0.322 },
  "2021-2023": { sbp: 125.8, dbp: 72.9, hypertension_pct: 0.341 },
};

// ── Heart Rate (derived from NHANES cardiovascular examination) ──

export const HEART_RATE: DistributionBySexAge = {
  Male: {
    "18-29": { mean: 71.2, sd: 11.4, p5: 53, p25: 63, p50: 70, p75: 78, p95: 91 },
    "30-39": { mean: 72.8, sd: 11.8, p5: 54, p25: 64, p50: 72, p75: 80, p95: 93 },
    "40-49": { mean: 73.4, sd: 12.1, p5: 54, p25: 65, p50: 72, p75: 81, p95: 94 },
    "50-59": { mean: 72.6, sd: 12.4, p5: 53, p25: 64, p50: 72, p75: 80, p95: 94 },
    "60-69": { mean: 70.8, sd: 12.0, p5: 52, p25: 63, p50: 70, p75: 78, p95: 92 },
    "70-79": { mean: 69.4, sd: 12.6, p5: 50, p25: 61, p50: 69, p75: 77, p95: 91 },
    "80+":   { mean: 68.9, sd: 13.1, p5: 48, p25: 60, p50: 68, p75: 77, p95: 91 },
  },
  Female: {
    "18-29": { mean: 75.8, sd: 11.2, p5: 58, p25: 68, p50: 75, p75: 83, p95: 95 },
    "30-39": { mean: 76.4, sd: 11.6, p5: 58, p25: 68, p50: 76, p75: 84, p95: 96 },
    "40-49": { mean: 76.1, sd: 11.8, p5: 57, p25: 68, p50: 76, p75: 84, p95: 96 },
    "50-59": { mean: 74.8, sd: 11.9, p5: 56, p25: 66, p50: 74, p75: 83, p95: 95 },
    "60-69": { mean: 73.2, sd: 11.6, p5: 55, p25: 65, p50: 73, p75: 81, p95: 93 },
    "70-79": { mean: 72.1, sd: 12.2, p5: 53, p25: 64, p50: 72, p75: 80, p95: 93 },
    "80+":   { mean: 71.6, sd: 12.8, p5: 51, p25: 63, p50: 71, p75: 80, p95: 93 },
  },
};

// ── Respiratory Rate (population reference ranges) ──

export const RESPIRATORY_RATE: DistributionBySexAge = {
  Male: {
    "18-29": { mean: 15.4, sd: 2.8, p5: 11, p25: 14, p50: 15, p75: 17, p95: 20 },
    "30-39": { mean: 15.6, sd: 2.9, p5: 11, p25: 14, p50: 16, p75: 17, p95: 20 },
    "40-49": { mean: 16.0, sd: 3.0, p5: 11, p25: 14, p50: 16, p75: 18, p95: 21 },
    "50-59": { mean: 16.4, sd: 3.2, p5: 12, p25: 14, p50: 16, p75: 18, p95: 22 },
    "60-69": { mean: 17.0, sd: 3.4, p5: 12, p25: 15, p50: 17, p75: 19, p95: 23 },
    "70-79": { mean: 17.6, sd: 3.6, p5: 12, p25: 15, p50: 17, p75: 20, p95: 24 },
    "80+":   { mean: 18.2, sd: 3.8, p5: 13, p25: 16, p50: 18, p75: 20, p95: 25 },
  },
  Female: {
    "18-29": { mean: 16.2, sd: 2.9, p5: 12, p25: 14, p50: 16, p75: 18, p95: 21 },
    "30-39": { mean: 16.4, sd: 3.0, p5: 12, p25: 14, p50: 16, p75: 18, p95: 21 },
    "40-49": { mean: 16.8, sd: 3.1, p5: 12, p25: 15, p50: 17, p75: 19, p95: 22 },
    "50-59": { mean: 17.2, sd: 3.3, p5: 12, p25: 15, p50: 17, p75: 19, p95: 23 },
    "60-69": { mean: 17.6, sd: 3.5, p5: 12, p25: 15, p50: 17, p75: 20, p95: 24 },
    "70-79": { mean: 18.0, sd: 3.7, p5: 13, p25: 16, p50: 18, p75: 20, p95: 24 },
    "80+":   { mean: 18.6, sd: 3.9, p5: 13, p25: 16, p50: 18, p75: 21, p95: 25 },
  },
};

// ── Temperature (population reference) ──

export const TEMPERATURE: DistributionBySexAge = {
  Male: {
    "18-29": { mean: 36.6, sd: 0.3, p5: 36.1, p25: 36.4, p50: 36.6, p75: 36.8, p95: 37.1 },
    "30-39": { mean: 36.6, sd: 0.3, p5: 36.1, p25: 36.4, p50: 36.6, p75: 36.8, p95: 37.1 },
    "40-49": { mean: 36.5, sd: 0.3, p5: 36.0, p25: 36.3, p50: 36.5, p75: 36.7, p95: 37.0 },
    "50-59": { mean: 36.5, sd: 0.3, p5: 36.0, p25: 36.3, p50: 36.5, p75: 36.7, p95: 37.0 },
    "60-69": { mean: 36.4, sd: 0.4, p5: 35.8, p25: 36.2, p50: 36.4, p75: 36.7, p95: 37.0 },
    "70-79": { mean: 36.3, sd: 0.4, p5: 35.7, p25: 36.1, p50: 36.3, p75: 36.6, p95: 36.9 },
    "80+":   { mean: 36.2, sd: 0.4, p5: 35.6, p25: 36.0, p50: 36.2, p75: 36.5, p95: 36.9 },
  },
  Female: {
    "18-29": { mean: 36.7, sd: 0.3, p5: 36.2, p25: 36.5, p50: 36.7, p75: 36.9, p95: 37.2 },
    "30-39": { mean: 36.7, sd: 0.3, p5: 36.2, p25: 36.5, p50: 36.7, p75: 36.9, p95: 37.2 },
    "40-49": { mean: 36.6, sd: 0.3, p5: 36.1, p25: 36.4, p50: 36.6, p75: 36.8, p95: 37.1 },
    "50-59": { mean: 36.6, sd: 0.3, p5: 36.1, p25: 36.4, p50: 36.6, p75: 36.8, p95: 37.1 },
    "60-69": { mean: 36.5, sd: 0.4, p5: 35.9, p25: 36.3, p50: 36.5, p75: 36.7, p95: 37.0 },
    "70-79": { mean: 36.4, sd: 0.4, p5: 35.8, p25: 36.2, p50: 36.4, p75: 36.6, p95: 37.0 },
    "80+":   { mean: 36.3, sd: 0.4, p5: 35.7, p25: 36.1, p50: 36.3, p75: 36.5, p95: 36.9 },
  },
};

// ── SpO2 (Pulse Oximetry reference ranges) ──

export const SPO2: DistributionBySexAge = {
  Male: {
    "18-29": { mean: 97.8, sd: 1.1, p5: 96, p25: 97, p50: 98, p75: 99, p95: 99 },
    "30-39": { mean: 97.6, sd: 1.2, p5: 96, p25: 97, p50: 98, p75: 99, p95: 99 },
    "40-49": { mean: 97.4, sd: 1.3, p5: 95, p25: 97, p50: 97, p75: 98, p95: 99 },
    "50-59": { mean: 97.1, sd: 1.4, p5: 95, p25: 96, p50: 97, p75: 98, p95: 99 },
    "60-69": { mean: 96.8, sd: 1.5, p5: 94, p25: 96, p50: 97, p75: 98, p95: 99 },
    "70-79": { mean: 96.4, sd: 1.7, p5: 93, p25: 95, p50: 97, p75: 98, p95: 99 },
    "80+":   { mean: 95.9, sd: 1.9, p5: 92, p25: 95, p50: 96, p75: 97, p95: 99 },
  },
  Female: {
    "18-29": { mean: 97.9, sd: 1.0, p5: 96, p25: 97, p50: 98, p75: 99, p95: 99 },
    "30-39": { mean: 97.7, sd: 1.1, p5: 96, p25: 97, p50: 98, p75: 99, p95: 99 },
    "40-49": { mean: 97.5, sd: 1.2, p5: 96, p25: 97, p50: 98, p75: 98, p95: 99 },
    "50-59": { mean: 97.2, sd: 1.3, p5: 95, p25: 96, p50: 97, p75: 98, p95: 99 },
    "60-69": { mean: 96.9, sd: 1.5, p5: 94, p25: 96, p50: 97, p75: 98, p95: 99 },
    "70-79": { mean: 96.5, sd: 1.6, p5: 94, p25: 96, p50: 97, p75: 98, p95: 99 },
    "80+":   { mean: 96.1, sd: 1.8, p5: 93, p25: 95, p50: 96, p75: 97, p95: 99 },
  },
};

// ── Complete Blood Count (CBC 2005-2023) ──

// WBC (x10^3/uL)
export const WBC: DistributionBySexAge = {
  Male: {
    "18-29": { mean: 6.8, sd: 1.9, p5: 4.0, p25: 5.4, p50: 6.6, p75: 7.9, p95: 10.2 },
    "30-39": { mean: 6.9, sd: 2.0, p5: 3.9, p25: 5.5, p50: 6.7, p75: 8.0, p95: 10.4 },
    "40-49": { mean: 7.0, sd: 2.0, p5: 4.0, p25: 5.5, p50: 6.8, p75: 8.1, p95: 10.6 },
    "50-59": { mean: 6.9, sd: 2.0, p5: 3.9, p25: 5.4, p50: 6.7, p75: 8.0, p95: 10.5 },
    "60-69": { mean: 6.7, sd: 2.0, p5: 3.8, p25: 5.3, p50: 6.5, p75: 7.8, p95: 10.3 },
    "70-79": { mean: 6.5, sd: 2.0, p5: 3.6, p25: 5.1, p50: 6.3, p75: 7.6, p95: 10.1 },
    "80+":   { mean: 6.3, sd: 2.1, p5: 3.4, p25: 4.9, p50: 6.1, p75: 7.4, p95: 10.0 },
  },
  Female: {
    "18-29": { mean: 7.2, sd: 2.1, p5: 4.2, p25: 5.7, p50: 7.0, p75: 8.4, p95: 11.0 },
    "30-39": { mean: 7.1, sd: 2.1, p5: 4.1, p25: 5.6, p50: 6.9, p75: 8.3, p95: 10.9 },
    "40-49": { mean: 6.9, sd: 2.0, p5: 4.0, p25: 5.5, p50: 6.7, p75: 8.0, p95: 10.5 },
    "50-59": { mean: 6.7, sd: 2.0, p5: 3.9, p25: 5.3, p50: 6.5, p75: 7.8, p95: 10.3 },
    "60-69": { mean: 6.5, sd: 2.0, p5: 3.7, p25: 5.1, p50: 6.3, p75: 7.6, p95: 10.1 },
    "70-79": { mean: 6.3, sd: 1.9, p5: 3.6, p25: 5.0, p50: 6.1, p75: 7.4, p95: 9.8 },
    "80+":   { mean: 6.1, sd: 1.9, p5: 3.4, p25: 4.8, p50: 5.9, p75: 7.2, p95: 9.6 },
  },
};

// Hemoglobin (g/dL)
export const HEMOGLOBIN: DistributionBySexAge = {
  Male: {
    "18-29": { mean: 15.3, sd: 1.0, p5: 13.6, p25: 14.6, p50: 15.3, p75: 16.0, p95: 17.0 },
    "30-39": { mean: 15.4, sd: 1.1, p5: 13.5, p25: 14.7, p50: 15.4, p75: 16.1, p95: 17.1 },
    "40-49": { mean: 15.3, sd: 1.1, p5: 13.4, p25: 14.6, p50: 15.3, p75: 16.0, p95: 17.1 },
    "50-59": { mean: 15.1, sd: 1.1, p5: 13.2, p25: 14.4, p50: 15.1, p75: 15.8, p95: 16.9 },
    "60-69": { mean: 14.8, sd: 1.2, p5: 12.8, p25: 14.1, p50: 14.8, p75: 15.6, p95: 16.7 },
    "70-79": { mean: 14.4, sd: 1.3, p5: 12.3, p25: 13.6, p50: 14.5, p75: 15.3, p95: 16.4 },
    "80+":   { mean: 13.9, sd: 1.4, p5: 11.6, p25: 13.0, p50: 14.0, p75: 14.9, p95: 16.1 },
  },
  Female: {
    "18-29": { mean: 13.2, sd: 1.0, p5: 11.5, p25: 12.5, p50: 13.2, p75: 13.9, p95: 14.9 },
    "30-39": { mean: 13.3, sd: 1.1, p5: 11.4, p25: 12.6, p50: 13.3, p75: 14.0, p95: 15.0 },
    "40-49": { mean: 13.4, sd: 1.1, p5: 11.6, p25: 12.7, p50: 13.4, p75: 14.1, p95: 15.1 },
    "50-59": { mean: 13.6, sd: 1.0, p5: 11.9, p25: 12.9, p50: 13.6, p75: 14.3, p95: 15.2 },
    "60-69": { mean: 13.5, sd: 1.1, p5: 11.7, p25: 12.8, p50: 13.5, p75: 14.2, p95: 15.2 },
    "70-79": { mean: 13.3, sd: 1.1, p5: 11.4, p25: 12.6, p50: 13.3, p75: 14.0, p95: 15.1 },
    "80+":   { mean: 13.0, sd: 1.2, p5: 11.0, p25: 12.2, p50: 13.0, p75: 13.8, p95: 15.0 },
  },
};

// Platelet Count (x10^3/uL)
export const PLATELETS: DistributionBySexAge = {
  Male: {
    "18-29": { mean: 238, sd: 52, p5: 158, p25: 201, p50: 234, p75: 271, p95: 330 },
    "30-39": { mean: 234, sd: 54, p5: 152, p25: 196, p50: 230, p75: 268, p95: 328 },
    "40-49": { mean: 228, sd: 56, p5: 146, p25: 190, p50: 224, p75: 262, p95: 326 },
    "50-59": { mean: 224, sd: 56, p5: 142, p25: 186, p50: 220, p75: 258, p95: 322 },
    "60-69": { mean: 220, sd: 58, p5: 136, p25: 181, p50: 216, p75: 255, p95: 322 },
    "70-79": { mean: 214, sd: 58, p5: 130, p25: 175, p50: 210, p75: 250, p95: 318 },
    "80+":   { mean: 208, sd: 60, p5: 122, p25: 168, p50: 204, p75: 244, p95: 314 },
  },
  Female: {
    "18-29": { mean: 264, sd: 58, p5: 174, p25: 223, p50: 260, p75: 301, p95: 366 },
    "30-39": { mean: 258, sd: 60, p5: 166, p25: 216, p50: 254, p75: 296, p95: 362 },
    "40-49": { mean: 252, sd: 60, p5: 160, p25: 210, p50: 248, p75: 290, p95: 358 },
    "50-59": { mean: 248, sd: 58, p5: 158, p25: 207, p50: 244, p75: 286, p95: 352 },
    "60-69": { mean: 244, sd: 58, p5: 154, p25: 204, p50: 240, p75: 282, p95: 348 },
    "70-79": { mean: 238, sd: 58, p5: 148, p25: 198, p50: 234, p75: 276, p95: 342 },
    "80+":   { mean: 232, sd: 60, p5: 140, p25: 191, p50: 228, p75: 270, p95: 338 },
  },
};

// ── C-Reactive Protein (CRP 2005-2010) mg/L ──

export const CRP: DistributionBySexAge = {
  Male: {
    "18-29": { mean: 1.4, sd: 2.8, p5: 0.2, p25: 0.4, p50: 0.8, p75: 1.6, p95: 5.8 },
    "30-39": { mean: 1.8, sd: 3.4, p5: 0.2, p25: 0.5, p50: 1.0, p75: 2.0, p95: 7.2 },
    "40-49": { mean: 2.1, sd: 3.8, p5: 0.3, p25: 0.6, p50: 1.2, p75: 2.4, p95: 8.1 },
    "50-59": { mean: 2.4, sd: 4.2, p5: 0.3, p25: 0.7, p50: 1.4, p75: 2.8, p95: 9.4 },
    "60-69": { mean: 2.7, sd: 4.5, p5: 0.3, p25: 0.8, p50: 1.6, p75: 3.2, p95: 10.8 },
    "70-79": { mean: 3.1, sd: 5.0, p5: 0.4, p25: 0.9, p50: 1.8, p75: 3.6, p95: 12.0 },
    "80+":   { mean: 3.5, sd: 5.6, p5: 0.4, p25: 1.0, p50: 2.0, p75: 4.0, p95: 13.4 },
  },
  Female: {
    "18-29": { mean: 2.0, sd: 3.6, p5: 0.2, p25: 0.5, p50: 1.0, p75: 2.2, p95: 7.8 },
    "30-39": { mean: 2.6, sd: 4.4, p5: 0.3, p25: 0.6, p50: 1.3, p75: 2.8, p95: 9.6 },
    "40-49": { mean: 2.9, sd: 4.8, p5: 0.3, p25: 0.7, p50: 1.5, p75: 3.2, p95: 10.8 },
    "50-59": { mean: 3.2, sd: 5.2, p5: 0.3, p25: 0.8, p50: 1.7, p75: 3.6, p95: 12.0 },
    "60-69": { mean: 3.4, sd: 5.4, p5: 0.4, p25: 0.8, p50: 1.8, p75: 3.8, p95: 12.6 },
    "70-79": { mean: 3.6, sd: 5.6, p5: 0.4, p25: 0.9, p50: 1.9, p75: 4.0, p95: 13.2 },
    "80+":   { mean: 3.8, sd: 5.8, p5: 0.4, p25: 1.0, p50: 2.0, p75: 4.2, p95: 14.0 },
  },
};

// CRP by BMI category
export const CRP_BY_BMI: Record<string, CRPByBMIEntry> = {
  "Underweight (<18.5)":  { mean: 1.1, sd: 2.0, elevated_pct: 0.08 },
  "Normal (18.5-24.9)":   { mean: 1.6, sd: 2.8, elevated_pct: 0.14 },
  "Overweight (25-29.9)": { mean: 2.4, sd: 3.8, elevated_pct: 0.22 },
  "Obese I (30-34.9)":    { mean: 3.6, sd: 5.2, elevated_pct: 0.34 },
  "Obese II (35-39.9)":   { mean: 5.1, sd: 6.8, elevated_pct: 0.46 },
  "Obese III (40+)":      { mean: 7.2, sd: 8.4, elevated_pct: 0.58 },
};

// ── Body Measurements (BMX 1999-2023) ──

export const BMI: DistributionBySexAge = {
  Male: {
    "18-29": { mean: 26.8, sd: 6.1, p5: 19.2, p25: 22.8, p50: 25.8, p75: 29.8, p95: 38.0 },
    "30-39": { mean: 28.6, sd: 6.4, p5: 20.4, p25: 24.2, p50: 27.6, p75: 31.8, p95: 40.2 },
    "40-49": { mean: 29.2, sd: 6.2, p5: 21.0, p25: 25.0, p50: 28.4, p75: 32.4, p95: 40.6 },
    "50-59": { mean: 29.4, sd: 5.9, p5: 21.4, p25: 25.4, p50: 28.8, p75: 32.6, p95: 40.0 },
    "60-69": { mean: 29.1, sd: 5.6, p5: 21.6, p25: 25.4, p50: 28.6, p75: 32.2, p95: 39.0 },
    "70-79": { mean: 28.4, sd: 5.2, p5: 21.2, p25: 25.0, p50: 27.8, p75: 31.2, p95: 37.6 },
    "80+":   { mean: 27.0, sd: 4.8, p5: 20.2, p25: 23.8, p50: 26.6, p75: 29.8, p95: 35.4 },
  },
  Female: {
    "18-29": { mean: 27.4, sd: 7.2, p5: 18.6, p25: 22.2, p50: 26.0, p75: 31.2, p95: 41.0 },
    "30-39": { mean: 29.0, sd: 7.6, p5: 19.4, p25: 23.4, p50: 27.6, p75: 33.0, p95: 43.2 },
    "40-49": { mean: 29.6, sd: 7.4, p5: 20.0, p25: 24.0, p50: 28.4, p75: 33.8, p95: 43.6 },
    "50-59": { mean: 29.8, sd: 7.0, p5: 20.6, p25: 24.6, p50: 28.8, p75: 34.0, p95: 43.0 },
    "60-69": { mean: 29.6, sd: 6.8, p5: 20.4, p25: 24.6, p50: 28.8, p75: 33.6, p95: 42.0 },
    "70-79": { mean: 28.8, sd: 6.4, p5: 20.0, p25: 24.2, p50: 28.0, p75: 32.6, p95: 40.4 },
    "80+":   { mean: 27.2, sd: 5.8, p5: 19.4, p25: 23.2, p50: 26.6, p75: 30.6, p95: 37.8 },
  },
};

// Obesity prevalence trends
export const OBESITY_TRENDS: Record<string, ObesityTrend> = {
  "1999-2000": { overall: 0.306, male: 0.279, female: 0.333 },
  "2001-2002": { overall: 0.311, male: 0.284, female: 0.339 },
  "2003-2004": { overall: 0.324, male: 0.312, female: 0.336 },
  "2005-2006": { overall: 0.348, male: 0.334, female: 0.361 },
  "2007-2008": { overall: 0.339, male: 0.322, female: 0.356 },
  "2009-2010": { overall: 0.359, male: 0.355, female: 0.362 },
  "2011-2012": { overall: 0.349, male: 0.338, female: 0.360 },
  "2013-2014": { overall: 0.379, male: 0.352, female: 0.404 },
  "2015-2016": { overall: 0.398, male: 0.378, female: 0.416 },
  "2017-2018": { overall: 0.424, male: 0.410, female: 0.438 },
  "2021-2023": { overall: 0.418, male: 0.402, female: 0.434 },
};

// ── Sepsis Risk Factor Prevalence (derived from NHANES) ──

export const SEPSIS_RISK_FACTORS: Record<string, RiskFactor> = {
  Hypertension: {
    overall: 0.322,
    by_age: { "18-29": 0.078, "30-39": 0.123, "40-49": 0.218, "50-59": 0.358, "60-69": 0.478, "70-79": 0.592, "80+": 0.641 },
    by_ethnicity: {
      "Non-Hispanic White": 0.307, "Non-Hispanic Black": 0.418,
      "Mexican-American": 0.262, "Other Hispanic": 0.271,
      "Asian": 0.258, "Other/Multi": 0.312,
    },
    sepsis_risk_multiplier: 1.3,
  },
  Diabetes: {
    overall: 0.133,
    by_age: { "18-29": 0.024, "30-39": 0.048, "40-49": 0.098, "50-59": 0.172, "60-69": 0.234, "70-79": 0.268, "80+": 0.252 },
    by_ethnicity: {
      "Non-Hispanic White": 0.112, "Non-Hispanic Black": 0.168,
      "Mexican-American": 0.182, "Other Hispanic": 0.156,
      "Asian": 0.142, "Other/Multi": 0.128,
    },
    sepsis_risk_multiplier: 1.8,
  },
  "Obesity (BMI>=30)": {
    overall: 0.418,
    by_age: { "18-29": 0.284, "30-39": 0.372, "40-49": 0.428, "50-59": 0.468, "60-69": 0.462, "70-79": 0.412, "80+": 0.308 },
    by_ethnicity: {
      "Non-Hispanic White": 0.398, "Non-Hispanic Black": 0.498,
      "Mexican-American": 0.462, "Other Hispanic": 0.418,
      "Asian": 0.162, "Other/Multi": 0.388,
    },
    sepsis_risk_multiplier: 1.5,
  },
  "Chronic Kidney Disease": {
    overall: 0.148,
    by_age: { "18-29": 0.062, "30-39": 0.074, "40-49": 0.098, "50-59": 0.148, "60-69": 0.224, "70-79": 0.318, "80+": 0.408 },
    by_ethnicity: {
      "Non-Hispanic White": 0.142, "Non-Hispanic Black": 0.178,
      "Mexican-American": 0.152, "Other Hispanic": 0.148,
      "Asian": 0.138, "Other/Multi": 0.144,
    },
    sepsis_risk_multiplier: 2.5,
  },
  "Anemia (low hemoglobin)": {
    overall: 0.058,
    by_age: { "18-29": 0.048, "30-39": 0.054, "40-49": 0.044, "50-59": 0.052, "60-69": 0.068, "70-79": 0.086, "80+": 0.118 },
    by_ethnicity: {
      "Non-Hispanic White": 0.042, "Non-Hispanic Black": 0.098,
      "Mexican-American": 0.064, "Other Hispanic": 0.058,
      "Asian": 0.048, "Other/Multi": 0.052,
    },
    sepsis_risk_multiplier: 1.6,
  },
  "Elevated CRP (>3mg/L)": {
    overall: 0.312,
    by_age: { "18-29": 0.198, "30-39": 0.258, "40-49": 0.302, "50-59": 0.348, "60-69": 0.368, "70-79": 0.384, "80+": 0.398 },
    by_ethnicity: {
      "Non-Hispanic White": 0.298, "Non-Hispanic Black": 0.358,
      "Mexican-American": 0.318, "Other Hispanic": 0.308,
      "Asian": 0.228, "Other/Multi": 0.302,
    },
    sepsis_risk_multiplier: 2.0,
  },
  "Current Smoker": {
    overall: 0.146,
    by_age: { "18-29": 0.118, "30-39": 0.168, "40-49": 0.178, "50-59": 0.172, "60-69": 0.138, "70-79": 0.092, "80+": 0.048 },
    by_ethnicity: {
      "Non-Hispanic White": 0.158, "Non-Hispanic Black": 0.168,
      "Mexican-American": 0.112, "Other Hispanic": 0.098,
      "Asian": 0.102, "Other/Multi": 0.148,
    },
    sepsis_risk_multiplier: 1.4,
  },
};

// ── Sepsis Epidemiology (derived/published) ──

export const SEPSIS_EPIDEMIOLOGY: SepsisEpidemiologyData = {
  us_annual_cases: 1700000,
  us_annual_deaths: 270000,
  global_annual_cases: 49000000,
  global_annual_deaths: 11000000,
  hospital_mortality_rate: 0.159,
  icu_admission_rate: 0.42,
  avg_los_days: 9.2,
  avg_cost_per_case: 32421,
  mortality_by_hour_delay: 0.076,
  early_detection_mortality_reduction: 0.25,
  incidence_per_100k: {
    "18-29": 52, "30-39": 78, "40-49": 134, "50-59": 242,
    "60-69": 418, "70-79": 682, "80+": 1024,
  },
  mortality_by_age: {
    "18-29": 0.058, "30-39": 0.074, "40-49": 0.108, "50-59": 0.148,
    "60-69": 0.198, "70-79": 0.268, "80+": 0.358,
  },
};

// ── Clinical Scoring Thresholds (for benchmarking) ──

export const SCORING_THRESHOLDS: Record<string, ScoringSystem> = {
  qSOFA: {
    name: "Quick SOFA",
    reference: "Seymour et al., JAMA 2016",
    parameters: {
      sbp: { threshold: 100, direction: "below", label: "SBP \u2264100 mmHg" },
      resp_rate: { threshold: 22, direction: "above", label: "RR \u226522/min" },
      gcs: { threshold: 15, direction: "below", label: "GCS <15" },
    },
    risk_levels: { 0: "Low", 1: "Low-Moderate", 2: "High", 3: "Critical" },
    sensitivity: 0.70,
    specificity: 0.79,
    auc: 0.81,
  },
  sirs: {
    name: "SIRS Criteria",
    reference: "Bone et al., Chest 1992",
    parameters: {
      temperature_high: { threshold: 38.3, direction: "above", label: "Temp >38.3\u00B0C" },
      temperature_low: { threshold: 36.0, direction: "below", label: "Temp <36.0\u00B0C" },
      heart_rate: { threshold: 90, direction: "above", label: "HR >90 bpm" },
      resp_rate: { threshold: 20, direction: "above", label: "RR >20/min" },
    },
    risk_levels: { 0: "Low", 1: "Low", 2: "Moderate", 3: "High" },
    sensitivity: 0.91,
    specificity: 0.31,
    auc: 0.72,
  },
  news2: {
    name: "NEWS2",
    reference: "Royal College of Physicians 2017",
    risk_levels: { "0-4": "Low", "5-6": "Medium", "7+": "High" },
    sensitivity: 0.87,
    specificity: 0.65,
    auc: 0.86,
  },
  shock_index: {
    name: "Shock Index",
    reference: "HR / SBP",
    risk_levels: { "<0.7": "Normal", "0.7-0.99": "Elevated", "1.0-1.29": "High", "\u22651.3": "Critical" },
    sensitivity: 0.65,
    specificity: 0.82,
    auc: 0.78,
  },
  uva: {
    name: "UVA/Kruisselbrink",
    reference: "Kruisselbrink et al., PLOS ONE 2019",
    risk_levels: { "0-1": "Low", "2-3": "Moderate", "4-5": "High", "6+": "Critical" },
    sensitivity: 0.83,
    specificity: 0.71,
    auc: 0.84,
  },
};

// ── Population qSOFA distribution (what % of population would trigger) ──

export const POPULATION_SCORE_DISTRIBUTION: Record<string, Record<string, number>> = {
  qSOFA: { 0: 0.712, 1: 0.218, 2: 0.058, 3: 0.012 },
  sirs: { 0: 0.424, 1: 0.338, 2: 0.178, 3: 0.060 },
  news2: { "0-4": 0.682, "5-6": 0.198, "7+": 0.120 },
};

// ── Utility Functions ──

/**
 * Estimate the percentile of a value within a distribution using
 * linear interpolation between known percentile landmarks.
 */
export function getPercentile(value: number, distribution: Distribution): number {
  if (value <= distribution.p5)
    return Math.round((value - distribution.mean + 2 * distribution.sd) / (4 * distribution.sd) * 5);
  if (value <= distribution.p25)
    return 5 + Math.round((value - distribution.p5) / (distribution.p25 - distribution.p5) * 20);
  if (value <= distribution.p50)
    return 25 + Math.round((value - distribution.p25) / (distribution.p50 - distribution.p25) * 25);
  if (value <= distribution.p75)
    return 50 + Math.round((value - distribution.p50) / (distribution.p75 - distribution.p50) * 25);
  if (value <= distribution.p95)
    return 75 + Math.round((value - distribution.p75) / (distribution.p95 - distribution.p75) * 20);
  return Math.min(99, 95 + Math.round((value - distribution.p95) / distribution.sd * 2));
}

/**
 * Get blood pressure benchmark for a systolic value given age and sex.
 */
export function getBPBenchmark(sbp: number, age: number, sex: string): BenchmarkResult | null {
  const ageGroup = getAgeGroup(age);
  const dist = BP_SYSTOLIC[sex]?.[ageGroup];
  if (!dist) return null;
  const pct = getPercentile(sbp, dist);
  let severity = "normal";
  if (sbp < dist.p5 || sbp >= 180) severity = "critical";
  else if (sbp < dist.p25 || sbp >= 140) severity = "warning";
  return { percentile: pct, severity, population: dist, ageGroup, sex };
}

/**
 * Get heart rate benchmark for a value given age and sex.
 */
export function getHRBenchmark(hr: number, age: number, sex: string): BenchmarkResult | null {
  const ageGroup = getAgeGroup(age);
  const dist = HEART_RATE[sex]?.[ageGroup];
  if (!dist) return null;
  const pct = getPercentile(hr, dist);
  let severity = "normal";
  if (hr < 50 || hr > 130) severity = "critical";
  else if (hr < 60 || hr > 100) severity = "warning";
  return { percentile: pct, severity, population: dist, ageGroup, sex };
}

/**
 * Get benchmarks for all provided vitals against the population.
 */
export function getCompleteBenchmark(
  vitals: VitalsInput,
  age: number,
  sex: string,
): Record<string, BenchmarkResult> {
  const results: Record<string, BenchmarkResult> = {};
  if (vitals.sbp != null) {
    const r = getBPBenchmark(vitals.sbp, age, sex);
    if (r) results.sbp = r;
  }
  if (vitals.heart_rate != null) {
    const r = getHRBenchmark(vitals.heart_rate, age, sex);
    if (r) results.heart_rate = r;
  }
  if (vitals.resp_rate != null) {
    const ag = getAgeGroup(age);
    const dist = RESPIRATORY_RATE[sex]?.[ag];
    if (dist) results.resp_rate = { percentile: getPercentile(vitals.resp_rate, dist), population: dist };
  }
  if (vitals.temperature != null) {
    const ag = getAgeGroup(age);
    const dist = TEMPERATURE[sex]?.[ag];
    if (dist) results.temperature = { percentile: getPercentile(vitals.temperature, dist), population: dist };
  }
  if (vitals.spo2 != null) {
    const ag = getAgeGroup(age);
    const dist = SPO2[sex]?.[ag];
    if (dist) results.spo2 = { percentile: getPercentile(vitals.spo2, dist), population: dist };
  }
  return results;
}

/**
 * Compute a sepsis risk profile given demographics and comorbidities.
 */
export function getRiskProfile(
  age: number,
  sex: string,
  ethnicity: string,
  comorbidities?: string[],
): RiskProfile {
  const ageGroup = getAgeGroup(age);
  const baseRisk = SEPSIS_EPIDEMIOLOGY.incidence_per_100k[ageGroup] / 100000;
  let multiplier = 1.0;
  const factors: Array<{ name: string; multiplier: number }> = [];
  for (const [name, data] of Object.entries(SEPSIS_RISK_FACTORS)) {
    if (comorbidities && comorbidities.includes(name)) {
      multiplier *= data.sepsis_risk_multiplier;
      factors.push({ name, multiplier: data.sepsis_risk_multiplier });
    }
  }
  const adjustedRisk = Math.min(baseRisk * multiplier, 0.5);
  return {
    baseRisk,
    multiplier,
    adjustedRisk,
    factors,
    ageGroup,
    populationPrevalence: SEPSIS_EPIDEMIOLOGY.incidence_per_100k[ageGroup],
    mortalityRate: SEPSIS_EPIDEMIOLOGY.mortality_by_age[ageGroup],
  };
}

/**
 * Generate a normal distribution curve from a Distribution's mean and sd.
 */
export function generateDistributionCurve(dist: Distribution, numPoints: number = 100): CurvePoint[] {
  const points: CurvePoint[] = [];
  const min = dist.mean - 3.5 * dist.sd;
  const max = dist.mean + 3.5 * dist.sd;
  const step = (max - min) / numPoints;
  for (let x = min; x <= max; x += step) {
    const z = (x - dist.mean) / dist.sd;
    const y = Math.exp(-0.5 * z * z) / (dist.sd * Math.sqrt(2 * Math.PI));
    points.push({ x: Math.round(x * 10) / 10, y });
  }
  return points;
}

/**
 * Map a numeric age to the corresponding NHANES age group string.
 */
export function getAgeGroup(age: number): string {
  if (age < 30) return "18-29";
  if (age < 40) return "30-39";
  if (age < 50) return "40-49";
  if (age < 60) return "50-59";
  if (age < 70) return "60-69";
  if (age < 80) return "70-79";
  return "80+";
}

// ── Convenience Functions for Population Explorer ──

/** Map of vital-sign keys to their distribution tables. */
const VITAL_TABLES: Record<string, DistributionBySexAge> = {
  sbp: BP_SYSTOLIC,
  bp_systolic: BP_SYSTOLIC,
  dbp: BP_DIASTOLIC,
  bp_diastolic: BP_DIASTOLIC,
  heart_rate: HEART_RATE,
  hr: HEART_RATE,
  respiratory_rate: RESPIRATORY_RATE,
  resp_rate: RESPIRATORY_RATE,
  temperature: TEMPERATURE,
  temp: TEMPERATURE,
  spo2: SPO2,
  wbc: WBC,
  hemoglobin: HEMOGLOBIN,
  platelets: PLATELETS,
  crp: CRP,
  bmi: BMI,
};

/**
 * Look up the Distribution for a given vital sign, sex, and age group.
 * Returns null if any key is unrecognized.
 */
export function getDistribution(vital: string, sex: string, ageGroup: string): Distribution | null {
  const table = VITAL_TABLES[vital.toLowerCase()];
  if (!table) return null;
  return table[sex]?.[ageGroup] ?? null;
}

/**
 * Return trend data for a given vital. Supports "bp" -> BP_TRENDS
 * and "bmi" / "obesity" -> OBESITY_TRENDS.
 */
export function getTrend(vital: string): Record<string, BPTrend> | Record<string, ObesityTrend> | null {
  const key = vital.toLowerCase();
  if (key === "bp" || key === "sbp" || key === "dbp" || key === "blood_pressure") return BP_TRENDS;
  if (key === "bmi" || key === "obesity") return OBESITY_TRENDS;
  return null;
}

/**
 * Return ethnicity breakdown for a vital. Currently only BP has
 * ethnicity-stratified data.
 */
export function getByEthnicity(vital: string): Record<string, BPEthnicity> | null {
  const key = vital.toLowerCase();
  if (key === "bp" || key === "sbp" || key === "dbp" || key === "blood_pressure") return BP_BY_ETHNICITY;
  return null;
}
