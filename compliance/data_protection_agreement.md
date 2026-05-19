# Data Protection Agreement — Sepsis Vitals

## Combined HIPAA Business Associate Agreement + GDPR Data Processing Agreement + Africa Data Protection Clauses

---

## 1. Parties

- **Data Controller / Covered Entity:** [Hospital / Research Institution Name]
- **Data Processor / Business Associate:** Sepsis Vitals [Operating Entity]

## 2. Scope of Processing

### Data Categories
| Category | Examples | Encryption |
|----------|----------|------------|
| Patient Identifiers | MRN, name, DOB | AES-256-GCM (SEPSIS_PII_KEY) |
| Clinical Vitals | Temperature, HR, RR, BP, SpO2, GCS | TLS 1.3 in transit, KMS at rest |
| Derived Scores | qSOFA, SIRS, NEWS2, risk levels | Standard encryption |
| Audit Logs | User actions, alert responses | CloudWatch with KMS |

### Processing Purposes
- Real-time sepsis risk scoring and alerting
- Clinical decision support (advisory only)
- Retrospective quality improvement analysis
- Model performance monitoring and fairness auditing

## 3. HIPAA Business Associate Provisions

### 3.1 Permitted Uses
Business Associate shall use PHI only for: treatment support, healthcare operations, and research (with IRB approval or de-identification per 45 CFR 164.514).

### 3.2 Safeguards
- Administrative: Role-based access control (nurse, researcher, system_admin)
- Technical: AES-256 encryption, JWT RS256 authentication, MFA (TOTP)
- Physical: AWS infrastructure with SOC 2 Type II compliance

### 3.3 Breach Notification
- Discovery within 24 hours of detection
- Notification to Covered Entity within 48 hours
- Notification to HHS within 60 days (if >= 500 individuals)

### 3.4 Minimum Necessary
System enforces minimum necessary through RBAC:
- Nurses: vital signs read, alert escalation
- Researchers: de-identified aggregate data only
- System admins: full access with audit logging

## 4. GDPR Data Processing Agreement

### 4.1 Legal Basis
- Legitimate interest (vital interests of data subject — Article 6(1)(d))
- Explicit consent where required by local law

### 4.2 Data Subject Rights
| Right | Implementation |
|-------|---------------|
| Access | Export endpoint in API |
| Rectification | Edit via authorized clinical staff |
| Erasure | Soft-delete with 30-day retention, then hard purge |
| Portability | FHIR-compatible JSON export |
| Restriction | Per-patient processing flags |

### 4.3 Sub-processors
| Sub-processor | Purpose | Location |
|---------------|---------|----------|
| AWS (Amazon) | Cloud infrastructure | Region-specific (configurable) |
| Anthropic | AI analysis (optional) | US-based, no PHI transmitted |

### 4.4 International Transfers
- Standard Contractual Clauses (SCCs) for EU -> non-EU transfers
- Data residency configurable per Terraform deployment region

## 5. Africa-Specific Data Protection Clauses

### 5.1 Kenya Data Protection Act (2019)
- Registration with Office of the Data Protection Commissioner
- Data Protection Impact Assessment completed
- Consent obtained in English and Kiswahili

### 5.2 African Union Convention on Cyber Security (Malabo Convention)
- Cross-border transfer only to countries with adequate protection
- Local data processing preferred where infrastructure permits

### 5.3 South Africa POPIA
- Processing limited to specific, lawful purpose
- Responsible party registered with Information Regulator

## 6. Technical Security Controls

### 6.1 Encryption
| Layer | Method | Key Management |
|-------|--------|---------------|
| PII at rest | AES-256-GCM | SEPSIS_PII_KEY (AWS Secrets Manager) |
| Config values | Fernet | SEPSIS_CONFIG_KEY (AWS Secrets Manager) |
| Database | RDS encryption | AWS KMS with auto-rotation |
| In transit | TLS 1.3 | ACM certificates |
| Redis | TLS + auth token | ElastiCache managed |

### 6.2 Access Control
- JWT RS256 tokens (15-minute access, 7-day refresh)
- Multi-factor authentication (TOTP)
- Account lockout with exponential backoff
- Role-based permissions matrix

### 6.3 Monitoring
- All API access logged (CloudWatch, 90-day retention)
- Webhook integrity (HMAC-SHA256 + replay protection)
- Rate limiting (5 req/s API, 0.5 req/s LLM, WAF 2000/5min/IP)
- Distribution drift detection (PSI monitoring)

## 7. Data Retention

| Data Type | Retention | Deletion Method |
|-----------|-----------|-----------------|
| Active patient data | Duration of care | Soft-delete on discharge |
| Audit logs | 90 days minimum | CloudWatch auto-expiry |
| Model artifacts | Versioned indefinitely | S3 lifecycle policies |
| De-identified research data | Per IRB protocol | Secure wipe |

## 8. Incident Response

1. **Detection** — Automated monitoring alerts (CloudWatch, WAF)
2. **Containment** — Rate limiting escalation, account lockout
3. **Investigation** — Audit log review, forensic analysis
4. **Notification** — Per HIPAA (48h), GDPR (72h), local law
5. **Remediation** — Root cause fix, post-incident review

## 9. Termination

Upon termination of this agreement:
- All PHI returned or destroyed within 30 days
- Certificate of destruction provided
- Audit logs retained per legal requirements
- De-identified aggregate data may be retained for research
