# eICU Demo SQLite Schema Analysis

> This document contains database-catalog metadata only. No patient rows or field values were queried or reproduced.

## Summary

- **SQLite archives scanned:** 1
- **Generation date:** 2026-06-07
- **Raw data directory:** `data/raw/eicu_demo`
- **Inspection method:** Each gzip archive was streamed to a temporary file inside the ignored raw-data directory, opened read-only, inspected through `sqlite_master` and `PRAGMA` metadata, and deleted immediately afterward.
- **Patient-data access:** No `SELECT` query was issued against a clinical table; no row counts or sample values were read.

## Archive: `physionet.org/files/eicu-crd-demo/2.0.1/sqlite/eicu_v2_0_1.sqlite3.gz`

- **Compressed size:** 82,213,572 bytes (78.4 MiB)
- **Uncompressed SQLite size:** 296,071,168 bytes (282.4 MiB)
- **Tables:** 31
- **Columns:** 391
- **Views:** 0
- **Triggers:** 0
- **Declared primary-key columns:** 0
- **Declared foreign-key constraints:** 0
- **Indexes reported by table metadata:** 0
- **Standalone user indexes:** 0
- **Columns declared `NOT NULL`:** 115 of 391

### What This File Is

The archive is a SQLite distribution of the eICU demo tables. It contains the same 31 logical table schemas and 391 columns as the CSV/CSV.GZ distribution. It is useful for SQL-based local exploration because it packages the tables into one database file and preserves declared SQL types and nullability metadata.

It is not a strongly constrained relational model: identifiers such as `patientunitstayid` and table-specific `...id` columns are ordinary columns rather than declared keys. Relationships therefore have to be applied from eICU conventions, especially joins from event tables to `patient.patientunitstayid`.

The absence of indexes means joins and filters may require full table scans. Any indexes needed for development should be created only on a separate local working copy, never by modifying the compressed source archive.

### CSV Comparison

- **Case-insensitive table matches:** 31 of 31
- **Structural schema mismatches:** 0
- **Tables with column-name casing differences only:** 7
- Column names and order match the CSV headers when compared case-insensitively.
- SQLite uses different letter casing for some otherwise matching headers in: `careplangoal` (5 columns), `note` (7 columns), `nurseassessment` (6 columns), `nursecare` (6 columns), `nursecharting` (6 columns), `respiratorycare` (34 columns), `respiratorycharting` (7 columns). SQLite identifiers are case-insensitive, but Python/dataframe code should normalize deliberately when comparing with exact CSV headers.

### Declared Type Families

| Declared Type Family | Columns |
|---|---:|
| `VARCHAR` | 136 |
| `INT` | 131 |
| `SMALLINT` | 49 |
| `DOUBLE PRECISION` | 46 |
| `NUMERIC` | 26 |
| `BIGINT` | 2 |
| `BOOLEAN` | 1 |

### Development Implications

- Use `patientunitstayid` as the principal ICU-stay join and event-grouping field, but enforce uniqueness and referential assumptions in application code because SQLite does not enforce them.
- Preserve CSV header casing in CSV-oriented code; use case-insensitive matching when reconciling SQLite metadata with CSV schemas.
- Treat declared SQL types as useful hints rather than guarantees about every stored value. SQLite uses dynamic typing, and several clinically numeric-looking fields are intentionally declared as text.
- Prefer the CSV files for the planned streaming event builder unless SQL predicates materially simplify a task. The SQLite copy is convenient for metadata discovery and bounded SQL exploration, but it does not add relationships or indexes.
- Do not infer clinical units from SQL types. Units and sentinel-value rules remain table/column-specific.

### Table Overview

| SQLite Table | Columns | Matching CSV | `NOT NULL` Columns | Primary Key | Foreign Keys | Indexes |
|---|---:|---|---:|---|---:|---:|
| `admissiondrug` | 14 | `admissiondrug.csv.gz` | 7 | None declared | 0 | 0 |
| `admissiondx` | 6 | `admissionDx.csv.gz` | 4 | None declared | 0 | 0 |
| `allergy` | 13 | `allergy.csv.gz` | 6 | None declared | 0 | 0 |
| `apacheapsvar` | 26 | `apacheApsVar.csv.gz` | 0 | None declared | 0 | 0 |
| `apachepatientresult` | 23 | `apachePatientResult.csv.gz` | 3 | None declared | 0 | 0 |
| `apachepredvar` | 51 | `apachePredVar.csv.gz` | 0 | None declared | 0 | 0 |
| `careplancareprovider` | 8 | `carePlanCareProvider.csv.gz` | 4 | None declared | 0 | 0 |
| `careplaneol` | 5 | `carePlanEOL.csv.gz` | 3 | None declared | 0 | 0 |
| `careplangeneral` | 6 | `carePlanGeneral.csv.gz` | 5 | None declared | 0 | 0 |
| `careplangoal` | 7 | `carePlanGoal.csv.gz` | 4 | None declared | 0 | 0 |
| `careplaninfectiousdisease` | 8 | `carePlanInfectiousDisease.csv.gz` | 4 | None declared | 0 | 0 |
| `customlab` | 7 | `customLab.csv.gz` | 4 | None declared | 0 | 0 |
| `diagnosis` | 7 | `diagnosis.csv.gz` | 5 | None declared | 0 | 0 |
| `hospital` | 4 | `hospital.csv.gz` | 1 | None declared | 0 | 0 |
| `infusiondrug` | 9 | `infusiondrug.csv.gz` | 4 | None declared | 0 | 0 |
| `intakeoutput` | 12 | `intakeOutput.csv.gz` | 6 | None declared | 0 | 0 |
| `lab` | 10 | `lab.csv.gz` | 4 | None declared | 0 | 0 |
| `medication` | 15 | `medication.csv.gz` | 10 | None declared | 0 | 0 |
| `microlab` | 7 | `microLab.csv.gz` | 5 | None declared | 0 | 0 |
| `note` | 8 | `note.csv.gz` | 6 | None declared | 0 | 0 |
| `nurseassessment` | 8 | `nurseAssessment.csv.gz` | 7 | None declared | 0 | 0 |
| `nursecare` | 8 | `nurseCare.csv.gz` | 7 | None declared | 0 | 0 |
| `nursecharting` | 8 | `nurseCharting.csv.gz` | 5 | None declared | 0 | 0 |
| `pasthistory` | 8 | `pastHistory.csv.gz` | 5 | None declared | 0 | 0 |
| `patient` | 29 | `patient.csv.gz` | 0 | None declared | 0 | 0 |
| `physicalexam` | 6 | `physicalExam.csv.gz` | 3 | None declared | 0 | 0 |
| `respiratorycare` | 34 | `respiratoryCare.csv.gz` | 0 | None declared | 0 | 0 |
| `respiratorycharting` | 7 | `respiratoryCharting.csv.gz` | 0 | None declared | 0 | 0 |
| `treatment` | 5 | `treatment.csv.gz` | 0 | None declared | 0 | 0 |
| `vitalaperiodic` | 13 | `vitalAperiodic.csv.gz` | 3 | None declared | 0 | 0 |
| `vitalperiodic` | 19 | `vitalPeriodic.csv.gz` | 0 | None declared | 0 | 0 |

## Table Schemas

### `admissiondrug`

- **Matching CSV:** `admissiondrug.csv.gz`
- **Columns:** 14
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `admissiondrugid` | `INT` | No | None declared | No key constraint declared |
| `patientunitstayid` | `INT` | No | None declared | No key constraint declared |
| `drugoffset` | `INT` | No | None declared | No key constraint declared |
| `drugenteredoffset` | `INT` | No | None declared | No key constraint declared |
| `drugnotetype` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |
| `specialtytype` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |
| `usertype` | `VARCHAR(255)` | No | None declared | No key constraint declared |
| `rxincluded` | `VARCHAR(5)` | Yes | None declared | No key constraint declared |
| `writtenineicu` | `VARCHAR(5)` | Yes | None declared | No key constraint declared |
| `drugname` | `VARCHAR(255)` | No | None declared | No key constraint declared |
| `drugdosage` | `NUMERIC(11,4)` | Yes | None declared | No key constraint declared |
| `drugunit` | `VARCHAR(1000)` | Yes | None declared | No key constraint declared |
| `drugadmitfrequency` | `VARCHAR(1000)` | No | None declared | No key constraint declared |
| `drughiclseqno` | `INT` | Yes | None declared | No key constraint declared |

### `admissiondx`

- **Matching CSV:** `admissionDx.csv.gz`
- **Columns:** 6
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `admissiondxid` | `INT` | No | None declared | No key constraint declared |
| `patientunitstayid` | `INT` | No | None declared | No key constraint declared |
| `admitdxenteredoffset` | `INT` | No | None declared | No key constraint declared |
| `admitdxpath` | `VARCHAR(500)` | No | None declared | No key constraint declared |
| `admitdxname` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |
| `admitdxtext` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |

### `allergy`

- **Matching CSV:** `allergy.csv.gz`
- **Columns:** 13
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `allergyid` | `INT` | No | None declared | No key constraint declared |
| `patientunitstayid` | `INT` | No | None declared | No key constraint declared |
| `allergyoffset` | `INT` | No | None declared | No key constraint declared |
| `allergyenteredoffset` | `INT` | No | None declared | No key constraint declared |
| `allergynotetype` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |
| `specialtytype` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |
| `usertype` | `VARCHAR(255)` | No | None declared | No key constraint declared |
| `rxincluded` | `VARCHAR(5)` | Yes | None declared | No key constraint declared |
| `writtenineicu` | `VARCHAR(5)` | Yes | None declared | No key constraint declared |
| `drugname` | `VARCHAR(255)` | No | None declared | No key constraint declared |
| `allergytype` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |
| `allergyname` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |
| `drughiclseqno` | `INT` | Yes | None declared | No key constraint declared |

### `apacheapsvar`

- **Matching CSV:** `apacheApsVar.csv.gz`
- **Columns:** 26
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `apacheapsvarid` | `INT` | Yes | None declared | No key constraint declared |
| `patientunitstayid` | `INT` | Yes | None declared | No key constraint declared |
| `intubated` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `vent` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `dialysis` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `eyes` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `motor` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `verbal` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `meds` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `urine` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `wbc` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `temperature` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `respiratoryrate` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `sodium` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `heartrate` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `meanbp` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `ph` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `hematocrit` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `creatinine` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `albumin` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `pao2` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `pco2` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `bun` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `glucose` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `bilirubin` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `fio2` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |

### `apachepatientresult`

- **Matching CSV:** `apachePatientResult.csv.gz`
- **Columns:** 23
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `apachepatientresultsid` | `INT` | No | None declared | No key constraint declared |
| `patientunitstayid` | `INT` | No | None declared | No key constraint declared |
| `physicianspeciality` | `VARCHAR(50)` | Yes | None declared | No key constraint declared |
| `physicianinterventioncategory` | `VARCHAR(50)` | Yes | None declared | No key constraint declared |
| `acutephysiologyscore` | `INT` | Yes | None declared | No key constraint declared |
| `apachescore` | `INT` | Yes | None declared | No key constraint declared |
| `apacheversion` | `VARCHAR(5)` | No | None declared | No key constraint declared |
| `predictedicumortality` | `VARCHAR(50)` | Yes | None declared | No key constraint declared |
| `actualicumortality` | `VARCHAR(50)` | Yes | None declared | No key constraint declared |
| `predictediculos` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `actualiculos` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `predictedhospitalmortality` | `VARCHAR(50)` | Yes | None declared | No key constraint declared |
| `actualhospitalmortality` | `VARCHAR(50)` | Yes | None declared | No key constraint declared |
| `predictedhospitallos` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `actualhospitallos` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `preopmi` | `INT` | Yes | None declared | No key constraint declared |
| `preopcardiaccath` | `INT` | Yes | None declared | No key constraint declared |
| `ptcawithin24h` | `INT` | Yes | None declared | No key constraint declared |
| `unabridgedunitlos` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `unabridgedhosplos` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `actualventdays` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `predventdays` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `unabridgedactualventdays` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |

### `apachepredvar`

- **Matching CSV:** `apachePredVar.csv.gz`
- **Columns:** 51
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `apachepredvarid` | `INT` | Yes | None declared | No key constraint declared |
| `patientunitstayid` | `INT` | Yes | None declared | No key constraint declared |
| `sicuday` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `saps3day1` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `saps3today` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `saps3yesterday` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `gender` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `teachtype` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `region` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `bedcount` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `admitsource` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `graftcount` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `meds` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `verbal` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `motor` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `eyes` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `age` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `admitdiagnosis` | `VARCHAR(11)` | Yes | None declared | No key constraint declared |
| `thrombolytics` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `diedinhospital` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `aids` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `hepaticfailure` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `lymphoma` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `metastaticcancer` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `leukemia` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `immunosuppression` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `cirrhosis` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `electivesurgery` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `activetx` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `readmit` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `ima` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `midur` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `ventday1` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `oobventday1` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `oobintubday1` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `diabetes` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `managementsystem` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `var03hspxlos` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `pao2` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `fio2` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `ejectfx` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `creatinine` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `dischargelocation` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `visitnumber` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `amilocation` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `day1meds` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `day1verbal` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `day1motor` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `day1eyes` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `day1pao2` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `day1fio2` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |

### `careplancareprovider`

- **Matching CSV:** `carePlanCareProvider.csv.gz`
- **Columns:** 8
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `cplcareprovderid` | `INT` | No | None declared | No key constraint declared |
| `patientunitstayid` | `INT` | No | None declared | No key constraint declared |
| `careprovidersaveoffset` | `INT` | No | None declared | No key constraint declared |
| `providertype` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |
| `specialty` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |
| `interventioncategory` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |
| `managingphysician` | `VARCHAR(50)` | Yes | None declared | No key constraint declared |
| `activeupondischarge` | `VARCHAR(10)` | No | None declared | No key constraint declared |

### `careplaneol`

- **Matching CSV:** `carePlanEOL.csv.gz`
- **Columns:** 5
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `cpleolid` | `INT` | No | None declared | No key constraint declared |
| `patientunitstayid` | `INT` | No | None declared | No key constraint declared |
| `cpleolsaveoffset` | `INT` | No | None declared | No key constraint declared |
| `cpleoldiscussionoffset` | `INT` | Yes | None declared | No key constraint declared |
| `activeupondischarge` | `VARCHAR(10)` | Yes | None declared | No key constraint declared |

### `careplangeneral`

- **Matching CSV:** `carePlanGeneral.csv.gz`
- **Columns:** 6
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `cplgeneralid` | `INT` | No | None declared | No key constraint declared |
| `patientunitstayid` | `INT` | No | None declared | No key constraint declared |
| `activeupondischarge` | `VARCHAR(10)` | No | None declared | No key constraint declared |
| `cplitemoffset` | `INT` | No | None declared | No key constraint declared |
| `cplgroup` | `VARCHAR(255)` | No | None declared | No key constraint declared |
| `cplitemvalue` | `VARCHAR(1024)` | Yes | None declared | No key constraint declared |

### `careplangoal`

- **Matching CSV:** `carePlanGoal.csv.gz`
- **Columns:** 7
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `cplgoalid` | `INT` | No | None declared | No key constraint declared |
| `patientunitstayid` | `INT` | No | None declared | No key constraint declared |
| `CPLGOALoffset` | `INT` | No | None declared | No key constraint declared |
| `CPLGOALCATEGORY` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |
| `CPLGOALVALUE` | `VARCHAR(1000)` | Yes | None declared | No key constraint declared |
| `CPLGOALSTATUS` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |
| `ACTIVEUPONDISCHARGE` | `VARCHAR(10)` | No | None declared | No key constraint declared |

### `careplaninfectiousdisease`

- **Matching CSV:** `carePlanInfectiousDisease.csv.gz`
- **Columns:** 8
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `cplinfectid` | `INT` | No | None declared | No key constraint declared |
| `patientunitstayid` | `INT` | No | None declared | No key constraint declared |
| `activeupondischarge` | `VARCHAR(10)` | No | None declared | No key constraint declared |
| `cplinfectdiseaseoffset` | `INT` | No | None declared | No key constraint declared |
| `infectdiseasesite` | `VARCHAR(64)` | Yes | None declared | No key constraint declared |
| `infectdiseaseassessment` | `VARCHAR(64)` | Yes | None declared | No key constraint declared |
| `responsetotherapy` | `VARCHAR(32)` | Yes | None declared | No key constraint declared |
| `treatment` | `VARCHAR(32)` | Yes | None declared | No key constraint declared |

### `customlab`

- **Matching CSV:** `customLab.csv.gz`
- **Columns:** 7
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `customlabid` | `INT` | No | None declared | No key constraint declared |
| `patientunitstayid` | `INT` | No | None declared | No key constraint declared |
| `labotheroffset` | `INT` | No | None declared | No key constraint declared |
| `labothertypeid` | `INT` | No | None declared | No key constraint declared |
| `labothername` | `VARCHAR(64)` | Yes | None declared | No key constraint declared |
| `labotherresult` | `VARCHAR(64)` | Yes | None declared | No key constraint declared |
| `labothervaluetext` | `VARCHAR(128)` | Yes | None declared | No key constraint declared |

### `diagnosis`

- **Matching CSV:** `diagnosis.csv.gz`
- **Columns:** 7
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `diagnosisid` | `INT` | No | None declared | No key constraint declared |
| `patientunitstayid` | `INT` | No | None declared | No key constraint declared |
| `activeupondischarge` | `VARCHAR(64)` | Yes | None declared | No key constraint declared |
| `diagnosisoffset` | `INT` | No | None declared | No key constraint declared |
| `diagnosisstring` | `VARCHAR(200)` | No | None declared | No key constraint declared |
| `icd9code` | `VARCHAR(100)` | Yes | None declared | No key constraint declared |
| `diagnosispriority` | `VARCHAR(10)` | No | None declared | No key constraint declared |

### `hospital`

- **Matching CSV:** `hospital.csv.gz`
- **Columns:** 4
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `hospitalid` | `INT` | No | None declared | No key constraint declared |
| `numbedscategory` | `VARCHAR(32)` | Yes | None declared | No key constraint declared |
| `teachingstatus` | `BOOLEAN` | Yes | None declared | No key constraint declared |
| `region` | `VARCHAR(64)` | Yes | None declared | No key constraint declared |

### `infusiondrug`

- **Matching CSV:** `infusiondrug.csv.gz`
- **Columns:** 9
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `infusiondrugid` | `INT` | No | None declared | No key constraint declared |
| `patientunitstayid` | `INT` | No | None declared | No key constraint declared |
| `infusionoffset` | `INT` | No | None declared | No key constraint declared |
| `drugname` | `VARCHAR(255)` | No | None declared | No key constraint declared |
| `drugrate` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |
| `infusionrate` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |
| `drugamount` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |
| `volumeoffluid` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |
| `patientweight` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |

### `intakeoutput`

- **Matching CSV:** `intakeOutput.csv.gz`
- **Columns:** 12
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `intakeoutputid` | `INT` | No | None declared | No key constraint declared |
| `patientunitstayid` | `INT` | No | None declared | No key constraint declared |
| `intakeoutputoffset` | `INT` | No | None declared | No key constraint declared |
| `intaketotal` | `NUMERIC(12,4)` | Yes | None declared | No key constraint declared |
| `outputtotal` | `NUMERIC(12,4)` | Yes | None declared | No key constraint declared |
| `dialysistotal` | `NUMERIC(12,4)` | Yes | None declared | No key constraint declared |
| `nettotal` | `NUMERIC(12,4)` | Yes | None declared | No key constraint declared |
| `intakeoutputentryoffset` | `INT` | No | None declared | No key constraint declared |
| `cellpath` | `VARCHAR(500)` | Yes | None declared | No key constraint declared |
| `celllabel` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |
| `cellvaluenumeric` | `NUMERIC(12,4)` | No | None declared | No key constraint declared |
| `cellvaluetext` | `VARCHAR(255)` | No | None declared | No key constraint declared |

### `lab`

- **Matching CSV:** `lab.csv.gz`
- **Columns:** 10
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `labid` | `INT` | No | None declared | No key constraint declared |
| `patientunitstayid` | `INT` | No | None declared | No key constraint declared |
| `labresultoffset` | `INT` | No | None declared | No key constraint declared |
| `labtypeid` | `NUMERIC(3,0)` | No | None declared | No key constraint declared |
| `labname` | `VARCHAR(256)` | Yes | None declared | No key constraint declared |
| `labresult` | `NUMERIC(11,4)` | Yes | None declared | No key constraint declared |
| `labresulttext` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |
| `labmeasurenamesystem` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |
| `labmeasurenameinterface` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |
| `labresultrevisedoffset` | `INT` | Yes | None declared | No key constraint declared |

### `medication`

- **Matching CSV:** `medication.csv.gz`
- **Columns:** 15
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `medicationid` | `INT` | No | None declared | No key constraint declared |
| `patientunitstayid` | `INT` | No | None declared | No key constraint declared |
| `drugorderoffset` | `INT` | No | None declared | No key constraint declared |
| `drugstartoffset` | `INT` | No | None declared | No key constraint declared |
| `drugivadmixture` | `VARCHAR(6)` | No | None declared | No key constraint declared |
| `drugordercancelled` | `VARCHAR(6)` | No | None declared | No key constraint declared |
| `drugname` | `VARCHAR(220)` | Yes | None declared | No key constraint declared |
| `drughiclseqno` | `INT` | Yes | None declared | No key constraint declared |
| `dosage` | `VARCHAR(60)` | Yes | None declared | No key constraint declared |
| `routeadmin` | `VARCHAR(120)` | Yes | None declared | No key constraint declared |
| `frequency` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |
| `loadingdose` | `VARCHAR(120)` | No | None declared | No key constraint declared |
| `prn` | `VARCHAR(6)` | No | None declared | No key constraint declared |
| `drugstopoffset` | `INT` | No | None declared | No key constraint declared |
| `gtc` | `INT` | No | None declared | No key constraint declared |

### `microlab`

- **Matching CSV:** `microLab.csv.gz`
- **Columns:** 7
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `microlabid` | `INT` | No | None declared | No key constraint declared |
| `patientunitstayid` | `INT` | No | None declared | No key constraint declared |
| `culturetakenoffset` | `INT` | No | None declared | No key constraint declared |
| `culturesite` | `VARCHAR(255)` | No | None declared | No key constraint declared |
| `organism` | `VARCHAR(255)` | No | None declared | No key constraint declared |
| `antibiotic` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |
| `sensitivitylevel` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |

### `note`

- **Matching CSV:** `note.csv.gz`
- **Columns:** 8
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `NOTEID` | `INT` | No | None declared | No key constraint declared |
| `patientunitstayid` | `INT` | No | None declared | No key constraint declared |
| `NOTEOFFSET` | `INT` | No | None declared | No key constraint declared |
| `NOTEENTEREDOFFSET` | `INT` | No | None declared | No key constraint declared |
| `NOTETYPE` | `VARCHAR(50)` | No | None declared | No key constraint declared |
| `NOTEPATH` | `VARCHAR(255)` | No | None declared | No key constraint declared |
| `NOTEVALUE` | `VARCHAR(150)` | Yes | None declared | No key constraint declared |
| `NOTETEXT` | `VARCHAR(500)` | Yes | None declared | No key constraint declared |

### `nurseassessment`

- **Matching CSV:** `nurseAssessment.csv.gz`
- **Columns:** 8
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `nurseassessid` | `INT` | No | None declared | No key constraint declared |
| `patientunitstayid` | `INT` | No | None declared | No key constraint declared |
| `NURSEASSESSOFFSET` | `INT` | No | None declared | No key constraint declared |
| `NURSEASSESSENTRYOFFSET` | `INT` | No | None declared | No key constraint declared |
| `CELLATTRIBUTEPATH` | `VARCHAR(255)` | No | None declared | No key constraint declared |
| `CELLLABEL` | `VARCHAR(255)` | No | None declared | No key constraint declared |
| `CELLATTRIBUTE` | `VARCHAR(255)` | No | None declared | No key constraint declared |
| `CELLATTRIBUTEVALUE` | `VARCHAR(4000)` | Yes | None declared | No key constraint declared |

### `nursecare`

- **Matching CSV:** `nurseCare.csv.gz`
- **Columns:** 8
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `nursecareid` | `INT` | No | None declared | No key constraint declared |
| `patientunitstayid` | `INT` | No | None declared | No key constraint declared |
| `CELLLABEL` | `VARCHAR(255)` | No | None declared | No key constraint declared |
| `NURSECAREOFFSET` | `INT` | No | None declared | No key constraint declared |
| `NURSECAREENTRYOFFSET` | `INT` | No | None declared | No key constraint declared |
| `CELLATTRIBUTEPATH` | `VARCHAR(255)` | No | None declared | No key constraint declared |
| `CELLATTRIBUTE` | `VARCHAR(255)` | No | None declared | No key constraint declared |
| `CELLATTRIBUTEVALUE` | `VARCHAR(4000)` | Yes | None declared | No key constraint declared |

### `nursecharting`

- **Matching CSV:** `nurseCharting.csv.gz`
- **Columns:** 8
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `nursingchartid` | `BIGINT` | No | None declared | No key constraint declared |
| `patientunitstayid` | `INT` | No | None declared | No key constraint declared |
| `NURSINGCHARTOFFSET` | `INT` | No | None declared | No key constraint declared |
| `NURSINGCHARTENTRYOFFSET` | `INT` | No | None declared | No key constraint declared |
| `NURSINGCHARTCELLTYPECAT` | `VARCHAR(255)` | No | None declared | No key constraint declared |
| `NURSINGCHARTCELLTYPEVALLABEL` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |
| `NURSINGCHARTCELLTYPEVALNAME` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |
| `NURSINGCHARTVALUE` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |

### `pasthistory`

- **Matching CSV:** `pastHistory.csv.gz`
- **Columns:** 8
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `pasthistoryid` | `INT` | No | None declared | No key constraint declared |
| `patientunitstayid` | `INT` | No | None declared | No key constraint declared |
| `pasthistoryoffset` | `INT` | No | None declared | No key constraint declared |
| `pasthistoryenteredoffset` | `INT` | No | None declared | No key constraint declared |
| `pasthistorynotetype` | `VARCHAR(40)` | Yes | None declared | No key constraint declared |
| `pasthistorypath` | `VARCHAR(255)` | No | None declared | No key constraint declared |
| `pasthistoryvalue` | `VARCHAR(100)` | Yes | None declared | No key constraint declared |
| `pasthistoryvaluetext` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |

### `patient`

- **Matching CSV:** `patient.csv.gz`
- **Columns:** 29
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `patientunitstayid` | `INT` | Yes | None declared | No key constraint declared |
| `patienthealthsystemstayid` | `INT` | Yes | None declared | No key constraint declared |
| `gender` | `VARCHAR(25)` | Yes | None declared | No key constraint declared |
| `age` | `VARCHAR(10)` | Yes | None declared | No key constraint declared |
| `ethnicity` | `VARCHAR(50)` | Yes | None declared | No key constraint declared |
| `hospitalid` | `INT` | Yes | None declared | No key constraint declared |
| `wardid` | `INT` | Yes | None declared | No key constraint declared |
| `apacheadmissiondx` | `VARCHAR(1000)` | Yes | None declared | No key constraint declared |
| `admissionheight` | `NUMERIC(10,2)` | Yes | None declared | No key constraint declared |
| `hospitaladmittime24` | `VARCHAR(8)` | Yes | None declared | No key constraint declared |
| `hospitaladmitoffset` | `INT` | Yes | None declared | No key constraint declared |
| `hospitaladmitsource` | `VARCHAR(30)` | Yes | None declared | No key constraint declared |
| `hospitaldischargeyear` | `SMALLINT` | Yes | None declared | No key constraint declared |
| `hospitaldischargetime24` | `VARCHAR(8)` | Yes | None declared | No key constraint declared |
| `hospitaldischargeoffset` | `INT` | Yes | None declared | No key constraint declared |
| `hospitaldischargelocation` | `VARCHAR(100)` | Yes | None declared | No key constraint declared |
| `hospitaldischargestatus` | `VARCHAR(10)` | Yes | None declared | No key constraint declared |
| `unittype` | `VARCHAR(50)` | Yes | None declared | No key constraint declared |
| `unitadmittime24` | `VARCHAR(8)` | Yes | None declared | No key constraint declared |
| `unitadmitsource` | `VARCHAR(100)` | Yes | None declared | No key constraint declared |
| `unitvisitnumber` | `INT` | Yes | None declared | No key constraint declared |
| `unitstaytype` | `VARCHAR(15)` | Yes | None declared | No key constraint declared |
| `admissionweight` | `NUMERIC(10,2)` | Yes | None declared | No key constraint declared |
| `dischargeweight` | `NUMERIC(10,2)` | Yes | None declared | No key constraint declared |
| `unitdischargetime24` | `VARCHAR(8)` | Yes | None declared | No key constraint declared |
| `unitdischargeoffset` | `INT` | Yes | None declared | No key constraint declared |
| `unitdischargelocation` | `VARCHAR(100)` | Yes | None declared | No key constraint declared |
| `unitdischargestatus` | `VARCHAR(10)` | Yes | None declared | No key constraint declared |
| `uniquepid` | `VARCHAR(10)` | Yes | None declared | No key constraint declared |

### `physicalexam`

- **Matching CSV:** `physicalExam.csv.gz`
- **Columns:** 6
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `physicalexamid` | `INT` | Yes | None declared | No key constraint declared |
| `patientunitstayid` | `INT` | No | None declared | No key constraint declared |
| `physicalexamoffset` | `INT` | No | None declared | No key constraint declared |
| `physicalexampath` | `VARCHAR(255)` | No | None declared | No key constraint declared |
| `physicalexamvalue` | `VARCHAR(100)` | Yes | None declared | No key constraint declared |
| `physicalexamtext` | `VARCHAR(500)` | Yes | None declared | No key constraint declared |

### `respiratorycare`

- **Matching CSV:** `respiratoryCare.csv.gz`
- **Columns:** 34
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `RESPCAREID` | `INT` | Yes | None declared | No key constraint declared |
| `PATIENTUNITSTAYID` | `INT` | Yes | None declared | No key constraint declared |
| `RESPCARESTATUSOFFSET` | `INT` | Yes | None declared | No key constraint declared |
| `CURRENTHISTORYSEQNUM` | `INT` | Yes | None declared | No key constraint declared |
| `AIRWAYTYPE` | `VARCHAR(30)` | Yes | None declared | No key constraint declared |
| `AIRWAYSIZE` | `VARCHAR(10)` | Yes | None declared | No key constraint declared |
| `AIRWAYPOSITION` | `VARCHAR(32)` | Yes | None declared | No key constraint declared |
| `CUFFPRESSURE` | `NUMERIC(5,1)` | Yes | None declared | No key constraint declared |
| `VENTSTARTOFFSET` | `INT` | Yes | None declared | No key constraint declared |
| `VENTENDOFFSET` | `INT` | Yes | None declared | No key constraint declared |
| `PRIORVENTSTARTOFFSET` | `INT` | Yes | None declared | No key constraint declared |
| `PRIORVENTENDOFFSET` | `INT` | Yes | None declared | No key constraint declared |
| `APNEAPARAMS` | `VARCHAR(80)` | Yes | None declared | No key constraint declared |
| `LOWEXHMVLIMIT` | `NUMERIC(11,4)` | Yes | None declared | No key constraint declared |
| `HIEXHMVLIMIT` | `NUMERIC(11,4)` | Yes | None declared | No key constraint declared |
| `LOWEXHTVLIMIT` | `NUMERIC(11,4)` | Yes | None declared | No key constraint declared |
| `HIPEAKPRESLIMIT` | `NUMERIC(11,4)` | Yes | None declared | No key constraint declared |
| `LOWPEAKPRESLIMIT` | `NUMERIC(11,4)` | Yes | None declared | No key constraint declared |
| `HIRESPRATELIMIT` | `NUMERIC(11,4)` | Yes | None declared | No key constraint declared |
| `LOWRESPRATELIMIT` | `NUMERIC(11,4)` | Yes | None declared | No key constraint declared |
| `SIGHPRESLIMIT` | `NUMERIC(11,4)` | Yes | None declared | No key constraint declared |
| `LOWIRONOXLIMIT` | `NUMERIC(11,4)` | Yes | None declared | No key constraint declared |
| `HIGHIRONOXLIMIT` | `NUMERIC(11,4)` | Yes | None declared | No key constraint declared |
| `MEANAIRWAYPRESLIMIT` | `NUMERIC(11,4)` | Yes | None declared | No key constraint declared |
| `PEEPLIMIT` | `NUMERIC(11,4)` | Yes | None declared | No key constraint declared |
| `CPAPLIMIT` | `NUMERIC(11,4)` | Yes | None declared | No key constraint declared |
| `SETAPNEAINTERVAL` | `VARCHAR(80)` | Yes | None declared | No key constraint declared |
| `SETAPNEATV` | `VARCHAR(80)` | Yes | None declared | No key constraint declared |
| `SETAPNEAIPPEEPHIGH` | `VARCHAR(80)` | Yes | None declared | No key constraint declared |
| `SETAPNEARR` | `VARCHAR(80)` | Yes | None declared | No key constraint declared |
| `SETAPNEAPEAKFLOW` | `VARCHAR(80)` | Yes | None declared | No key constraint declared |
| `SETAPNEAINSPTIME` | `VARCHAR(80)` | Yes | None declared | No key constraint declared |
| `SETAPNEAIE` | `VARCHAR(80)` | Yes | None declared | No key constraint declared |
| `SETAPNEAFIO2` | `VARCHAR(80)` | Yes | None declared | No key constraint declared |

### `respiratorycharting`

- **Matching CSV:** `respiratoryCharting.csv.gz`
- **Columns:** 7
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `RESPCHARTID` | `INT` | Yes | None declared | No key constraint declared |
| `PATIENTUNITSTAYID` | `INT` | Yes | None declared | No key constraint declared |
| `RESPCHARTOFFSET` | `INT` | Yes | None declared | No key constraint declared |
| `RESPCHARTENTRYOFFSET` | `INT` | Yes | None declared | No key constraint declared |
| `RESPCHARTTYPECAT` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |
| `RESPCHARTVALUELABEL` | `VARCHAR(255)` | Yes | None declared | No key constraint declared |
| `RESPCHARTVALUE` | `VARCHAR(1000)` | Yes | None declared | No key constraint declared |

### `treatment`

- **Matching CSV:** `treatment.csv.gz`
- **Columns:** 5
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `treatmentid` | `INT` | Yes | None declared | No key constraint declared |
| `patientunitstayid` | `INT` | Yes | None declared | No key constraint declared |
| `treatmentoffset` | `INT` | Yes | None declared | No key constraint declared |
| `treatmentstring` | `VARCHAR(200)` | Yes | None declared | No key constraint declared |
| `activeupondischarge` | `VARCHAR(10)` | Yes | None declared | No key constraint declared |

### `vitalaperiodic`

- **Matching CSV:** `vitalAperiodic.csv.gz`
- **Columns:** 13
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `vitalaperiodicid` | `INT` | No | None declared | No key constraint declared |
| `patientunitstayid` | `INT` | No | None declared | No key constraint declared |
| `observationoffset` | `INT` | No | None declared | No key constraint declared |
| `noninvasivesystolic` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `noninvasivediastolic` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `noninvasivemean` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `paop` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `cardiacoutput` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `cardiacinput` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `svr` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `svri` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `pvr` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `pvri` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |

### `vitalperiodic`

- **Matching CSV:** `vitalPeriodic.csv.gz`
- **Columns:** 19
- **Primary key:** None declared
- **Foreign keys:** 0
- **Indexes:** 0

| Column Name | Declared SQLite Type | Nullable | Default | Key / Constraint Notes |
|---|---|---|---|---|
| `vitalperiodicid` | `BIGINT` | Yes | None declared | No key constraint declared |
| `patientunitstayid` | `INT` | Yes | None declared | No key constraint declared |
| `observationoffset` | `INT` | Yes | None declared | No key constraint declared |
| `temperature` | `NUMERIC(11,4)` | Yes | None declared | No key constraint declared |
| `sao2` | `INT` | Yes | None declared | No key constraint declared |
| `heartrate` | `INT` | Yes | None declared | No key constraint declared |
| `respiration` | `INT` | Yes | None declared | No key constraint declared |
| `cvp` | `INT` | Yes | None declared | No key constraint declared |
| `etco2` | `INT` | Yes | None declared | No key constraint declared |
| `systemicsystolic` | `INT` | Yes | None declared | No key constraint declared |
| `systemicdiastolic` | `INT` | Yes | None declared | No key constraint declared |
| `systemicmean` | `INT` | Yes | None declared | No key constraint declared |
| `pasystolic` | `INT` | Yes | None declared | No key constraint declared |
| `padiastolic` | `INT` | Yes | None declared | No key constraint declared |
| `pamean` | `INT` | Yes | None declared | No key constraint declared |
| `st1` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `st2` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `st3` | `DOUBLE PRECISION` | Yes | None declared | No key constraint declared |
| `icp` | `INT` | Yes | None declared | No key constraint declared |
