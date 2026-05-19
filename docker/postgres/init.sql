-- Sepsis Vitals — Initial database schema
-- Applied automatically on first docker-compose up

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Patients
CREATE TABLE IF NOT EXISTS patients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id VARCHAR(64) UNIQUE NOT NULL,
    site_id VARCHAR(32) NOT NULL,
    age_years SMALLINT,
    sex CHAR(1) CHECK (sex IN ('M', 'F', 'U')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Vital sign observations
CREATE TABLE IF NOT EXISTS vitals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID NOT NULL REFERENCES patients(id),
    recorded_at TIMESTAMPTZ NOT NULL,
    temperature REAL,
    heart_rate SMALLINT,
    resp_rate SMALLINT,
    sbp SMALLINT,
    spo2 SMALLINT,
    gcs SMALLINT CHECK (gcs BETWEEN 3 AND 15),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_vitals_patient_time ON vitals(patient_id, recorded_at DESC);

-- Score results
CREATE TABLE IF NOT EXISTS scores (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vital_id UUID NOT NULL REFERENCES vitals(id),
    qsofa SMALLINT,
    sirs_count SMALLINT,
    shock_index REAL,
    news2_style SMALLINT,
    uva_style SMALLINT,
    risk_level VARCHAR(16) NOT NULL,
    alert_flag BOOLEAN DEFAULT FALSE,
    component_flags JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_scores_risk ON scores(risk_level, alert_flag);

-- Alerts
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    score_id UUID NOT NULL REFERENCES scores(id),
    patient_id UUID NOT NULL REFERENCES patients(id),
    risk_level VARCHAR(16) NOT NULL,
    status VARCHAR(16) DEFAULT 'active' CHECK (status IN ('active', 'acknowledged', 'dismissed', 'escalated')),
    action_by UUID,
    action_reason TEXT,
    time_to_action_s REAL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    actioned_at TIMESTAMPTZ
);

CREATE INDEX idx_alerts_status ON alerts(status, created_at DESC);

-- Users
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(32) NOT NULL CHECK (role IN ('nurse', 'researcher', 'system_admin')),
    site_id VARCHAR(32),
    totp_secret VARCHAR(64),
    mfa_enabled BOOLEAN DEFAULT FALSE,
    failed_attempts SMALLINT DEFAULT 0,
    locked_until TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ
);

-- Audit log
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(64) NOT NULL,
    resource_type VARCHAR(32),
    resource_id UUID,
    details JSONB,
    ip_address INET,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_audit_user_time ON audit_log(user_id, created_at DESC);
