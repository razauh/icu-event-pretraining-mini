# eICU Demo Raw-File Header Summary

> This document records schema-level information only. It does not reproduce patient rows or sample values.

## Summary

- **Total CSV/CSV.GZ files scanned:** 31
- **Total columns documented:** 391
- **Generation date:** 2026-06-07
- **Raw data directory:** `data/raw/eicu_demo`
- **Read policy:** Header plus at most 5 rows per file; files were never loaded in full.
- **Scope note:** Ancillary license, checksum, HTML, and SQLite files were excluded; the SQLite archive duplicates the CSV-oriented dataset representation.
- **Interpretation note:** eICU `offset` fields are generally minute offsets relative to ICU admission, but downstream code should confirm units and sentinel-value handling before event construction.

## Potentially Important Identifier Columns

These columns can affect joins, grouping, leakage prevention, and event-stream construction. Occurrence counts are numbers of CSV tables containing the exact header.

| Column Name | Tables | Likely Role |
|---|---:|---|
| `patientunitstayid` | 30 | De-identified identifier for one ICU unit stay. |
| `drughiclseqno` | 3 | HICL drug terminology sequence number used to standardize a medication. |
| `hospitalid` | 2 | De-identified hospital identifier. |
| `admissiondrugid` | 1 | Row or entity identifier for the admissiondrug record. |
| `admissiondxid` | 1 | Row or entity identifier for the admissiondx record. |
| `allergyid` | 1 | Row or entity identifier for the allergy record. |
| `apacheapsvarid` | 1 | Row or entity identifier for the apacheapsvar record. |
| `apachepatientresultsid` | 1 | Row or entity identifier for the apachepatientresults record. |
| `apachepredvarid` | 1 | Row or entity identifier for the apachepredvar record. |
| `cplcareprovderid` | 1 | Row or entity identifier for the cplcareprovder record. |
| `cpleolid` | 1 | Row or entity identifier for the cpleol record. |
| `cplgeneralid` | 1 | Row or entity identifier for the cplgeneral record. |
| `cplgoalid` | 1 | Row or entity identifier for the cplgoal record. |
| `cplinfectid` | 1 | Row or entity identifier for the cplinfect record. |
| `customlabid` | 1 | Row or entity identifier for the customlab record. |
| `diagnosisid` | 1 | Row or entity identifier for the diagnosis record. |
| `infusiondrugid` | 1 | Row or entity identifier for the infusiondrug record. |
| `intakeoutputid` | 1 | Row or entity identifier for the intakeoutput record. |
| `labid` | 1 | Row or entity identifier for the lab record. |
| `labothertypeid` | 1 | Identifier for the custom-laboratory result type. |
| `labtypeid` | 1 | Identifier for the laboratory-test type. |
| `medicationid` | 1 | Row or entity identifier for the medication record. |
| `microlabid` | 1 | Row or entity identifier for the microlab record. |
| `noteid` | 1 | Row or entity identifier for the note record. |
| `nurseassessid` | 1 | Row or entity identifier for the nurseassess record. |
| `nursecareid` | 1 | Row or entity identifier for the nursecare record. |
| `nursingchartid` | 1 | Row or entity identifier for the nursingchart record. |
| `pasthistoryid` | 1 | Row or entity identifier for the pasthistory record. |
| `patienthealthsystemstayid` | 1 | De-identified identifier linking ICU stays within one hospital-system encounter. |
| `physicalexamid` | 1 | Row or entity identifier for the physicalexam record. |
| `respcareid` | 1 | Row or entity identifier for the respcare record. |
| `respchartid` | 1 | Row or entity identifier for the respchart record. |
| `treatmentid` | 1 | Row or entity identifier for the treatment record. |
| `uniquepid` | 1 | De-identified patient identifier used to link encounters for the same patient. |
| `vitalaperiodicid` | 1 | Row or entity identifier for the vitalaperiodic record. |
| `vitalperiodicid` | 1 | Row or entity identifier for the vitalperiodic record. |
| `wardid` | 1 | De-identified ICU unit or ward identifier. |

The central event-stream key is `patientunitstayid`. `patienthealthsystemstayid` and `uniquepid` provide broader encounter/patient linkage in `patient.csv`. Table-specific row IDs should not be treated as patient grouping keys. The demo includes `hospitalid` and `wardid` headers; any use for site-level claims must remain consistent with the project’s demo-dataset limitations.

## Potential Time / Offset Columns

Occurrence counts are numbers of CSV tables containing the exact header. Most offsets are suitable candidates for chronological ordering after table-specific semantics and missing/sentinel values are handled.

| Column Name | Tables | Likely Role |
|---|---:|---|
| `observationoffset` | 2 | Offset of the vital-sign observation relative to ICU admission. |
| `admitdxenteredoffset` | 1 | Offset when the admission diagnosis was entered, relative to ICU admission. |
| `allergyenteredoffset` | 1 | Offset when the allergy entry was recorded, relative to ICU admission. |
| `allergyoffset` | 1 | Offset of the allergy event relative to ICU admission. |
| `careprovidersaveoffset` | 1 | Offset when the care-provider record was saved, relative to ICU admission. |
| `cpleoldiscussionoffset` | 1 | Offset of the end-of-life discussion relative to ICU admission. |
| `cpleolsaveoffset` | 1 | Offset when the end-of-life care-plan record was saved. |
| `cplgoaloffset` | 1 | Offset of the care-plan goal relative to ICU admission. |
| `cplinfectdiseaseoffset` | 1 | Offset of the infectious-disease care-plan entry relative to ICU admission. |
| `cplitemoffset` | 1 | Offset of the general care-plan item relative to ICU admission. |
| `culturetakenoffset` | 1 | Offset when the microbiology culture was collected. |
| `diagnosisoffset` | 1 | Offset of the diagnosis entry relative to ICU admission. |
| `drugenteredoffset` | 1 | Offset when the drug entry was recorded, relative to ICU admission. |
| `drugoffset` | 1 | Offset of the drug event relative to ICU admission, generally in minutes. |
| `drugorderoffset` | 1 | Offset when the medication was ordered. |
| `drugstartoffset` | 1 | Offset when the medication was scheduled or recorded to start. |
| `drugstopoffset` | 1 | Offset when the medication was scheduled or recorded to stop. |
| `hospitaladmitoffset` | 1 | Hospital admission offset relative to ICU admission, generally in minutes. |
| `hospitaladmittime24` | 1 | De-identified hospital admission clock time in 24-hour format. |
| `hospitaldischargeoffset` | 1 | Hospital discharge offset relative to ICU admission, generally in minutes. |
| `hospitaldischargetime24` | 1 | De-identified hospital discharge clock time in 24-hour format. |
| `hospitaldischargeyear` | 1 | De-identified year of hospital discharge. |
| `infusionoffset` | 1 | Offset of the infusion event relative to ICU admission. |
| `intakeoutputentryoffset` | 1 | Offset when the intake/output entry was recorded. |
| `intakeoutputoffset` | 1 | Offset of the intake/output event relative to ICU admission. |
| `labotheroffset` | 1 | Offset of the custom laboratory result relative to ICU admission. |
| `labresultoffset` | 1 | Offset of the laboratory result relative to ICU admission. |
| `labresultrevisedoffset` | 1 | Offset when the laboratory result was revised. |
| `noteenteredoffset` | 1 | Offset when the clinical note was entered. |
| `noteoffset` | 1 | Offset of the clinical note event relative to ICU admission. |
| `nurseassessentryoffset` | 1 | Offset when the nursing assessment was entered. |
| `nurseassessoffset` | 1 | Offset of the nursing assessment relative to ICU admission. |
| `nursecareentryoffset` | 1 | Offset when the nursing-care entry was recorded. |
| `nursecareoffset` | 1 | Offset of the nursing-care event relative to ICU admission. |
| `nursingchartentryoffset` | 1 | Offset when the nursing chart observation was entered. |
| `nursingchartoffset` | 1 | Offset of the nursing chart observation relative to ICU admission. |
| `oobintubday1` | 1 | Out-of-bounds or exception flag for day-one intubation data. |
| `oobventday1` | 1 | Out-of-bounds or exception flag for day-one ventilation data. |
| `pasthistoryenteredoffset` | 1 | Offset when the past-history entry was recorded. |
| `pasthistoryoffset` | 1 | Offset of the past-history event relative to ICU admission. |
| `physicalexamoffset` | 1 | Offset of the physical-exam event relative to ICU admission. |
| `priorventendoffset` | 1 | Offset when the prior ventilation episode ended. |
| `priorventstartoffset` | 1 | Offset when the prior ventilation episode started. |
| `respcarestatusoffset` | 1 | Offset of the respiratory-care status record relative to ICU admission. |
| `respchartentryoffset` | 1 | Offset when the respiratory chart event was entered. |
| `respchartoffset` | 1 | Offset of the respiratory chart event relative to ICU admission. |
| `saps3day1` | 1 | SAPS 3 day-one model variable or score component. |
| `saps3today` | 1 | SAPS 3 current-day model variable or score component. |
| `saps3yesterday` | 1 | SAPS 3 prior-day model variable or score component. |
| `sicuday` | 1 | SICU-day variable used by the APACHE prediction model. |
| `treatmentoffset` | 1 | Offset of the treatment event relative to ICU admission. |
| `unitadmittime24` | 1 | De-identified ICU-unit admission clock time in 24-hour format. |
| `unitdischargeoffset` | 1 | ICU-unit discharge offset relative to ICU admission, generally in minutes. |
| `unitdischargetime24` | 1 | De-identified ICU-unit discharge clock time in 24-hour format. |
| `ventday1` | 1 | Indicator of ventilation on day one. |
| `ventendoffset` | 1 | Offset when the current ventilation episode ended. |
| `ventstartoffset` | 1 | Offset when the current ventilation episode started. |

## File Schemas

### `physionet.org/files/eicu-crd-demo/2.0.1/admissiondrug.csv.gz`

- **Columns:** 14
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `admissiondrugid` | Row or entity identifier for the admissiondrug record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `drugoffset` | Offset of the drug event relative to ICU admission, generally in minutes. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `drugenteredoffset` | Offset when the drug entry was recorded, relative to ICU admission. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `drugnotetype` | Type of note or record associated with the drug entry. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `specialtytype` | Clinical specialty category associated with the documentation. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `usertype` | Category of user who entered the documentation. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `rxincluded` | Indicator related to inclusion of the item in medication reconciliation. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `writtenineicu` | Indicator of whether the item was documented while the patient was in the ICU. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `drugname` | Recorded medication or drug name. | Directly obvious from the column name. A bounded sample contained text values; no values are reproduced here. |
| `drugdosage` | Recorded numeric drug dose. | Based on known eICU naming conventions. A bounded sample contained numeric values; no values are reproduced here. |
| `drugunit` | Unit associated with the recorded drug dose. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `drugadmitfrequency` | Recorded medication frequency at admission. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `drughiclseqno` | HICL drug terminology sequence number used to standardize a medication. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |

### `physionet.org/files/eicu-crd-demo/2.0.1/admissionDx.csv.gz`

- **Columns:** 6
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `admissiondxid` | Row or entity identifier for the admissiondx record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `admitdxenteredoffset` | Offset when the admission diagnosis was entered, relative to ICU admission. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `admitdxpath` | Hierarchical path of the admission diagnosis in the source interface. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `admitdxname` | Admission diagnosis name. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `admitdxtext` | Textual value or description associated with the admission diagnosis. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |

### `physionet.org/files/eicu-crd-demo/2.0.1/allergy.csv.gz`

- **Columns:** 13
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `allergyid` | Row or entity identifier for the allergy record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `allergyoffset` | Offset of the allergy event relative to ICU admission. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `allergyenteredoffset` | Offset when the allergy entry was recorded, relative to ICU admission. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `allergynotetype` | Type of note or record associated with the allergy entry. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `specialtytype` | Clinical specialty category associated with the documentation. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `usertype` | Category of user who entered the documentation. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `rxincluded` | Indicator related to inclusion of the item in medication reconciliation. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `writtenineicu` | Indicator of whether the item was documented while the patient was in the ICU. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `drugname` | Recorded medication or drug name. | Directly obvious from the column name. A bounded sample contained text values; no values are reproduced here. |
| `allergytype` | Category of recorded allergy. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `allergyname` | Recorded allergen or allergy name. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `drughiclseqno` | HICL drug terminology sequence number used to standardize a medication. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |

### `physionet.org/files/eicu-crd-demo/2.0.1/apacheApsVar.csv.gz`

- **Columns:** 26
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `apacheapsvarid` | Row or entity identifier for the apacheapsvar record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `intubated` | APACHE indicator of intubation status. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `vent` | APACHE indicator of mechanical ventilation status. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `dialysis` | APACHE indicator of dialysis status. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `eyes` | Eye-response component used by the APACHE/Glasgow Coma Scale assessment. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `motor` | Motor-response component used by the APACHE/Glasgow Coma Scale assessment. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `verbal` | Verbal-response component used by the APACHE/Glasgow Coma Scale assessment. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `meds` | APACHE indicator that neurological assessment was affected by medication. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `urine` | Urine-output value used by APACHE scoring. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `wbc` | White blood cell count used by APACHE scoring. | Based on known eICU naming conventions. A bounded sample contained integer-like, numeric values; no values are reproduced here. |
| `temperature` | Recorded temperature measurement. | Directly obvious from the column name. A bounded sample contained numeric values; no values are reproduced here. |
| `respiratoryrate` | Recorded respiratory rate. | Directly obvious from the column name. A bounded sample contained integer-like values; no values are reproduced here. |
| `sodium` | Recorded serum sodium value. | Directly obvious from the column name. A bounded sample contained integer-like values; no values are reproduced here. |
| `heartrate` | Recorded heart rate. | Directly obvious from the column name. A bounded sample contained integer-like values; no values are reproduced here. |
| `meanbp` | Recorded mean blood pressure. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `ph` | Recorded blood pH value. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `hematocrit` | Recorded hematocrit value. | Directly obvious from the column name. A bounded sample contained integer-like, numeric values; no values are reproduced here. |
| `creatinine` | Recorded creatinine value. | Directly obvious from the column name. A bounded sample contained integer-like, numeric values; no values are reproduced here. |
| `albumin` | Recorded albumin value. | Directly obvious from the column name. A bounded sample contained integer-like, numeric values; no values are reproduced here. |
| `pao2` | Recorded arterial oxygen partial pressure (PaO2). | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `pco2` | Recorded arterial carbon dioxide partial pressure (PaCO2). | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `bun` | Recorded blood urea nitrogen value. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `glucose` | Recorded glucose value. | Directly obvious from the column name. A bounded sample contained integer-like values; no values are reproduced here. |
| `bilirubin` | Recorded bilirubin value. | Directly obvious from the column name. A bounded sample contained integer-like, numeric values; no values are reproduced here. |
| `fio2` | Recorded fraction of inspired oxygen (FiO2). | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |

### `physionet.org/files/eicu-crd-demo/2.0.1/apachePatientResult.csv.gz`

- **Columns:** 23
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `apachepatientresultsid` | Row or entity identifier for the apachepatientresults record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `physicianspeciality` | Specialty of the physician associated with the APACHE result. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `physicianinterventioncategory` | Physician intervention category used in the APACHE result. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `acutephysiologyscore` | Acute Physiology Score component of APACHE. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `apachescore` | Calculated APACHE severity score. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `apacheversion` | APACHE model version used for the result. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `predictedicumortality` | APACHE-predicted probability of ICU mortality. | Based on known eICU naming conventions. A bounded sample contained numeric values; no values are reproduced here. |
| `actualicumortality` | Observed ICU mortality outcome. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `predictediculos` | APACHE-predicted ICU length of stay. | Based on known eICU naming conventions. A bounded sample contained numeric values; no values are reproduced here. |
| `actualiculos` | Observed ICU length of stay. | Based on known eICU naming conventions. A bounded sample contained numeric values; no values are reproduced here. |
| `predictedhospitalmortality` | APACHE-predicted probability of hospital mortality. | Based on known eICU naming conventions. A bounded sample contained numeric values; no values are reproduced here. |
| `actualhospitalmortality` | Observed hospital mortality outcome. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `predictedhospitallos` | APACHE-predicted hospital length of stay. | Based on known eICU naming conventions. A bounded sample contained numeric values; no values are reproduced here. |
| `actualhospitallos` | Observed hospital length of stay. | Based on known eICU naming conventions. A bounded sample contained numeric values; no values are reproduced here. |
| `preopmi` | APACHE preoperative myocardial-infarction indicator. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `preopcardiaccath` | APACHE preoperative cardiac-catheterization indicator. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `ptcawithin24h` | Indicator for percutaneous transluminal coronary angioplasty within 24 hours. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `unabridgedunitlos` | Unabridged ICU unit length-of-stay value used in APACHE results. | Based on known eICU naming conventions. A bounded sample contained numeric values; no values are reproduced here. |
| `unabridgedhosplos` | Unabridged hospital length-of-stay value used in APACHE results. | Based on known eICU naming conventions. A bounded sample contained numeric values; no values are reproduced here. |
| `actualventdays` | Observed number of ventilation days. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `predventdays` | Predicted number of ventilation days. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `unabridgedactualventdays` | Unabridged observed ventilation-days value. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |

### `physionet.org/files/eicu-crd-demo/2.0.1/apachePredVar.csv.gz`

- **Columns:** 51
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `apachepredvarid` | Row or entity identifier for the apachepredvar record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `sicuday` | SICU-day variable used by the APACHE prediction model. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `saps3day1` | SAPS 3 day-one model variable or score component. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `saps3today` | SAPS 3 current-day model variable or score component. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `saps3yesterday` | SAPS 3 prior-day model variable or score component. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `gender` | Recorded patient sex or gender category. | Directly obvious from the column name. A bounded sample contained integer-like values; no values are reproduced here. |
| `teachtype` | Encoded hospital teaching-status category used by the prediction model. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `region` | Geographic region category associated with the hospital or prediction model. | Directly obvious from the column name. A bounded sample contained integer-like values; no values are reproduced here. |
| `bedcount` | Encoded hospital bed-count category used by the prediction model. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `admitsource` | Encoded admission-source category used by the prediction model. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `graftcount` | Graft-count variable used by the APACHE prediction model. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `meds` | APACHE indicator that neurological assessment was affected by medication. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `verbal` | Verbal-response component used by the APACHE/Glasgow Coma Scale assessment. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `motor` | Motor-response component used by the APACHE/Glasgow Coma Scale assessment. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `eyes` | Eye-response component used by the APACHE/Glasgow Coma Scale assessment. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `age` | Patient age as represented in the demo dataset. | Directly obvious from the column name. A bounded sample contained integer-like values; no values are reproduced here. |
| `admitdiagnosis` | Admission diagnosis used by the prediction model. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `thrombolytics` | Indicator of thrombolytic therapy. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `diedinhospital` | Indicator that the patient died during the hospital encounter. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `aids` | APACHE comorbidity indicator for AIDS. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `hepaticfailure` | APACHE comorbidity indicator for hepatic failure. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `lymphoma` | APACHE comorbidity indicator for lymphoma. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `metastaticcancer` | APACHE comorbidity indicator for metastatic cancer. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `leukemia` | APACHE comorbidity indicator for leukemia. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `immunosuppression` | APACHE comorbidity indicator for immunosuppression. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `cirrhosis` | APACHE comorbidity indicator for cirrhosis. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `electivesurgery` | Indicator that the admission involved elective surgery. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `activetx` | Encoded indicator of active treatment in the APACHE model. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `readmit` | Encoded readmission indicator used by the APACHE model. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `ima` | APACHE prediction variable with an abbreviated source-system meaning. | Uncertain; cautious interpretation of an abbreviated eICU/APACHE field name. A bounded sample contained integer-like values; no values are reproduced here. |
| `midur` | APACHE prediction variable related to myocardial-infarction duration. | Uncertain; cautious interpretation of an abbreviated eICU/APACHE field name. A bounded sample contained integer-like values; no values are reproduced here. |
| `ventday1` | Indicator of ventilation on day one. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `oobventday1` | Out-of-bounds or exception flag for day-one ventilation data. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `oobintubday1` | Out-of-bounds or exception flag for day-one intubation data. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `diabetes` | APACHE comorbidity indicator for diabetes. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `managementsystem` | Encoded management-system variable used by the APACHE model. | Uncertain; cautious interpretation of an abbreviated eICU/APACHE field name. A bounded sample contained integer-like values; no values are reproduced here. |
| `var03hspxlos` | Abbreviated APACHE prediction variable associated with hospital length of stay. | Uncertain; cautious interpretation of an abbreviated eICU/APACHE field name. A bounded sample contained integer-like values; no values are reproduced here. |
| `pao2` | Recorded arterial oxygen partial pressure (PaO2). | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `fio2` | Recorded fraction of inspired oxygen (FiO2). | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `ejectfx` | Ejection-fraction variable used by the APACHE model. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `creatinine` | Recorded creatinine value. | Directly obvious from the column name. A bounded sample contained integer-like, numeric values; no values are reproduced here. |
| `dischargelocation` | Encoded discharge-location category used by the prediction model. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `visitnumber` | Visit sequence number used by the prediction model. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `amilocation` | Encoded acute-myocardial-infarction location variable. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `day1meds` | Day-one indicator that neurological assessment was affected by medication. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `day1verbal` | Day-one verbal-response score. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `day1motor` | Day-one motor-response score. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `day1eyes` | Day-one eye-response score. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `day1pao2` | Day-one arterial oxygen partial pressure value. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `day1fio2` | Day-one fraction of inspired oxygen value. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |

### `physionet.org/files/eicu-crd-demo/2.0.1/carePlanCareProvider.csv.gz`

- **Columns:** 8
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `cplcareprovderid` | Row or entity identifier for the cplcareprovder record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `careprovidersaveoffset` | Offset when the care-provider record was saved, relative to ICU admission. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `providertype` | Category of care provider. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `specialty` | Clinical specialty of the care provider. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `interventioncategory` | Category of intervention associated with the care provider. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `managingphysician` | Indicator or category identifying the managing-physician role. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `activeupondischarge` | Indicator of whether the documented item remained active at ICU discharge. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |

### `physionet.org/files/eicu-crd-demo/2.0.1/carePlanEOL.csv.gz`

- **Columns:** 5
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `cpleolid` | Row or entity identifier for the cpleol record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `cpleolsaveoffset` | Offset when the end-of-life care-plan record was saved. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `cpleoldiscussionoffset` | Offset of the end-of-life discussion relative to ICU admission. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `activeupondischarge` | Indicator of whether the documented item remained active at ICU discharge. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |

### `physionet.org/files/eicu-crd-demo/2.0.1/carePlanGeneral.csv.gz`

- **Columns:** 6
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `cplgeneralid` | Row or entity identifier for the cplgeneral record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `activeupondischarge` | Indicator of whether the documented item remained active at ICU discharge. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `cplitemoffset` | Offset of the general care-plan item relative to ICU admission. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `cplgroup` | Care-plan group or category. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `cplitemvalue` | Recorded value of the care-plan item. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |

### `physionet.org/files/eicu-crd-demo/2.0.1/carePlanGoal.csv.gz`

- **Columns:** 7
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `cplgoalid` | Row or entity identifier for the cplgoal record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `cplgoaloffset` | Offset of the care-plan goal relative to ICU admission. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `cplgoalcategory` | Category of the care-plan goal. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `cplgoalvalue` | Recorded care-plan goal value. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `cplgoalstatus` | Status of the care-plan goal. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `activeupondischarge` | Indicator of whether the documented item remained active at ICU discharge. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |

### `physionet.org/files/eicu-crd-demo/2.0.1/carePlanInfectiousDisease.csv.gz`

- **Columns:** 8
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `cplinfectid` | Row or entity identifier for the cplinfect record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `activeupondischarge` | Indicator of whether the documented item remained active at ICU discharge. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `cplinfectdiseaseoffset` | Offset of the infectious-disease care-plan entry relative to ICU admission. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `infectdiseasesite` | Body site associated with an infectious-disease care-plan entry. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `infectdiseaseassessment` | Recorded infectious-disease assessment. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `responsetotherapy` | Recorded response to therapy for the infectious-disease entry. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `treatment` | Recorded treatment associated with the infectious-disease care plan. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |

### `physionet.org/files/eicu-crd-demo/2.0.1/customLab.csv.gz`

- **Columns:** 7
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `customlabid` | Row or entity identifier for the customlab record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `labotheroffset` | Offset of the custom laboratory result relative to ICU admission. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `labothertypeid` | Identifier for the custom-laboratory result type. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `labothername` | Name of the custom laboratory test. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `labotherresult` | Numeric result of the custom laboratory test. | Based on known eICU naming conventions. A bounded sample contained numeric values; no values are reproduced here. |
| `labothervaluetext` | Text-form result of the custom laboratory test. | Based on known eICU naming conventions. A bounded sample contained integer-like, text values; no values are reproduced here. |

### `physionet.org/files/eicu-crd-demo/2.0.1/diagnosis.csv.gz`

- **Columns:** 7
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `diagnosisid` | Row or entity identifier for the diagnosis record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `activeupondischarge` | Indicator of whether the documented item remained active at ICU discharge. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `diagnosisoffset` | Offset of the diagnosis entry relative to ICU admission. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `diagnosisstring` | Hierarchical or descriptive diagnosis string. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `icd9code` | ICD-9 diagnosis code associated with the diagnosis. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `diagnosispriority` | Priority assigned to the diagnosis. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |

### `physionet.org/files/eicu-crd-demo/2.0.1/hospital.csv.gz`

- **Columns:** 4
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `hospitalid` | De-identified hospital identifier. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `numbedscategory` | Categorized number of hospital beds. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `teachingstatus` | Hospital teaching-status category. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `region` | Geographic region category associated with the hospital or prediction model. | Directly obvious from the column name. A bounded sample contained text values; no values are reproduced here. |

### `physionet.org/files/eicu-crd-demo/2.0.1/infusiondrug.csv.gz`

- **Columns:** 9
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `infusiondrugid` | Row or entity identifier for the infusiondrug record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `infusionoffset` | Offset of the infusion event relative to ICU admission. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `drugname` | Recorded medication or drug name. | Directly obvious from the column name. A bounded sample contained text values; no values are reproduced here. |
| `drugrate` | Recorded drug administration rate. | Based on known eICU naming conventions. A bounded sample contained integer-like, numeric values; no values are reproduced here. |
| `infusionrate` | Recorded infusion rate. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `drugamount` | Amount of drug in the infusion. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `volumeoffluid` | Volume of fluid used for the infusion. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `patientweight` | Patient weight associated with infusion-rate calculation. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |

### `physionet.org/files/eicu-crd-demo/2.0.1/intakeOutput.csv.gz`

- **Columns:** 12
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `intakeoutputid` | Row or entity identifier for the intakeoutput record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `intakeoutputoffset` | Offset of the intake/output event relative to ICU admission. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `intaketotal` | Recorded total fluid intake. | Based on known eICU naming conventions. A bounded sample contained numeric values; no values are reproduced here. |
| `outputtotal` | Recorded total fluid output. | Based on known eICU naming conventions. A bounded sample contained numeric values; no values are reproduced here. |
| `dialysistotal` | Recorded fluid total attributed to dialysis. | Based on known eICU naming conventions. A bounded sample contained numeric values; no values are reproduced here. |
| `nettotal` | Calculated net fluid balance. | Based on known eICU naming conventions. A bounded sample contained numeric values; no values are reproduced here. |
| `intakeoutputentryoffset` | Offset when the intake/output entry was recorded. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `cellpath` | Hierarchical source-interface path for the documented item. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `celllabel` | Display label for the documented item. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `cellvaluenumeric` | Numeric value of the documented item. | Based on known eICU naming conventions. A bounded sample contained numeric values; no values are reproduced here. |
| `cellvaluetext` | Text-form value of the documented item. | Based on known eICU naming conventions. A bounded sample contained integer-like, numeric values; no values are reproduced here. |

### `physionet.org/files/eicu-crd-demo/2.0.1/lab.csv.gz`

- **Columns:** 10
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `labid` | Row or entity identifier for the lab record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `labresultoffset` | Offset of the laboratory result relative to ICU admission. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `labtypeid` | Identifier for the laboratory-test type. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `labname` | Laboratory-test name. | Directly obvious from the column name. A bounded sample contained text values; no values are reproduced here. |
| `labresult` | Numeric laboratory result. | Based on known eICU naming conventions. A bounded sample contained numeric values; no values are reproduced here. |
| `labresulttext` | Text-form laboratory result. | Based on known eICU naming conventions. A bounded sample contained integer-like, numeric values; no values are reproduced here. |
| `labmeasurenamesystem` | Standardized system measurement name for the laboratory test. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `labmeasurenameinterface` | Source-interface measurement name for the laboratory test. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `labresultrevisedoffset` | Offset when the laboratory result was revised. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |

### `physionet.org/files/eicu-crd-demo/2.0.1/medication.csv.gz`

- **Columns:** 15
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `medicationid` | Row or entity identifier for the medication record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `drugorderoffset` | Offset when the medication was ordered. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `drugstartoffset` | Offset when the medication was scheduled or recorded to start. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `drugivadmixture` | Indicator that the medication is an IV admixture. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `drugordercancelled` | Indicator that the medication order was cancelled. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `drugname` | Recorded medication or drug name. | Directly obvious from the column name. A bounded sample contained text values; no values are reproduced here. |
| `drughiclseqno` | HICL drug terminology sequence number used to standardize a medication. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `dosage` | Recorded medication dosage, including source formatting where present. | Directly obvious from the column name. A bounded sample contained text values; no values are reproduced here. |
| `routeadmin` | Medication administration route. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `frequency` | Medication administration frequency. | Directly obvious from the column name. A bounded sample contained text values; no values are reproduced here. |
| `loadingdose` | Recorded medication loading dose. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `prn` | Indicator that the medication was ordered as needed (PRN). | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `drugstopoffset` | Offset when the medication was scheduled or recorded to stop. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `gtc` | Medication classification field with an abbreviated source-system meaning. | Uncertain; cautious interpretation of an abbreviated eICU/APACHE field name. A bounded sample contained integer-like values; no values are reproduced here. |

### `physionet.org/files/eicu-crd-demo/2.0.1/microLab.csv.gz`

- **Columns:** 7
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `microlabid` | Row or entity identifier for the microlab record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `culturetakenoffset` | Offset when the microbiology culture was collected. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `culturesite` | Body site from which the microbiology culture was taken. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `organism` | Organism identified by the culture. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `antibiotic` | Antibiotic tested against the cultured organism. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `sensitivitylevel` | Reported antimicrobial sensitivity category or level. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |

### `physionet.org/files/eicu-crd-demo/2.0.1/note.csv.gz`

- **Columns:** 8
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `noteid` | Row or entity identifier for the note record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `noteoffset` | Offset of the clinical note event relative to ICU admission. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `noteenteredoffset` | Offset when the clinical note was entered. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `notetype` | Category of clinical note. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `notepath` | Hierarchical source-interface path for the note item. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `notevalue` | Structured value associated with the note item. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `notetext` | Text-form content associated with the note item. | Based on known eICU naming conventions. A bounded sample contained integer-like, text values; no values are reproduced here. |

### `physionet.org/files/eicu-crd-demo/2.0.1/nurseAssessment.csv.gz`

- **Columns:** 8
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `nurseassessid` | Row or entity identifier for the nurseassess record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `nurseassessoffset` | Offset of the nursing assessment relative to ICU admission. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `nurseassessentryoffset` | Offset when the nursing assessment was entered. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `cellattributepath` | Hierarchical source-interface path for the nursing attribute. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `celllabel` | Display label for the documented item. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `cellattribute` | Name of the nursing documentation attribute. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `cellattributevalue` | Recorded value of the nursing documentation attribute. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |

### `physionet.org/files/eicu-crd-demo/2.0.1/nurseCare.csv.gz`

- **Columns:** 8
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `nursecareid` | Row or entity identifier for the nursecare record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `celllabel` | Display label for the documented item. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `nursecareoffset` | Offset of the nursing-care event relative to ICU admission. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `nursecareentryoffset` | Offset when the nursing-care entry was recorded. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `cellattributepath` | Hierarchical source-interface path for the nursing attribute. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `cellattribute` | Name of the nursing documentation attribute. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `cellattributevalue` | Recorded value of the nursing documentation attribute. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |

### `physionet.org/files/eicu-crd-demo/2.0.1/nurseCharting.csv.gz`

- **Columns:** 8
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `nursingchartid` | Row or entity identifier for the nursingchart record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `nursingchartoffset` | Offset of the nursing chart observation relative to ICU admission. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `nursingchartentryoffset` | Offset when the nursing chart observation was entered. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `nursingchartcelltypecat` | Category of the nursing chart item. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `nursingchartcelltypevallabel` | Display label for the nursing chart value type. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `nursingchartcelltypevalname` | Source name for the nursing chart value type. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `nursingchartvalue` | Recorded nursing chart value. | Based on known eICU naming conventions. A bounded sample contained integer-like, text values; no values are reproduced here. |

### `physionet.org/files/eicu-crd-demo/2.0.1/pastHistory.csv.gz`

- **Columns:** 8
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `pasthistoryid` | Row or entity identifier for the pasthistory record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `pasthistoryoffset` | Offset of the past-history event relative to ICU admission. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `pasthistoryenteredoffset` | Offset when the past-history entry was recorded. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `pasthistorynotetype` | Type of note or record associated with the past-history entry. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `pasthistorypath` | Hierarchical source-interface path for the past-history item. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `pasthistoryvalue` | Structured value of the past-history item. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `pasthistoryvaluetext` | Text-form value of the past-history item. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |

### `physionet.org/files/eicu-crd-demo/2.0.1/patient.csv.gz`

- **Columns:** 29
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patienthealthsystemstayid` | De-identified identifier linking ICU stays within one hospital-system encounter. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `gender` | Recorded patient sex or gender category. | Directly obvious from the column name. A bounded sample contained text values; no values are reproduced here. |
| `age` | Patient age as represented in the demo dataset. | Directly obvious from the column name. A bounded sample contained integer-like values; no values are reproduced here. |
| `ethnicity` | Recorded patient ethnicity category. | Directly obvious from the column name. A bounded sample contained text values; no values are reproduced here. |
| `hospitalid` | De-identified hospital identifier. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `wardid` | De-identified ICU unit or ward identifier. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `apacheadmissiondx` | APACHE admission diagnosis assigned to the ICU stay. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `admissionheight` | Patient height recorded at admission. | Based on known eICU naming conventions. A bounded sample contained numeric values; no values are reproduced here. |
| `hospitaladmittime24` | De-identified hospital admission clock time in 24-hour format. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `hospitaladmitoffset` | Hospital admission offset relative to ICU admission, generally in minutes. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `hospitaladmitsource` | Source or location from which the patient was admitted to the hospital. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `hospitaldischargeyear` | De-identified year of hospital discharge. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `hospitaldischargetime24` | De-identified hospital discharge clock time in 24-hour format. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `hospitaldischargeoffset` | Hospital discharge offset relative to ICU admission, generally in minutes. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `hospitaldischargelocation` | Destination or location after hospital discharge. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `hospitaldischargestatus` | Patient status at hospital discharge. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `unittype` | Type or specialty of the ICU unit. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `unitadmittime24` | De-identified ICU-unit admission clock time in 24-hour format. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `unitadmitsource` | Source or location from which the patient entered the ICU unit. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `unitvisitnumber` | Sequence number of the ICU visit within the encounter. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `unitstaytype` | Category describing the ICU stay or admission type. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `admissionweight` | Patient weight recorded at ICU admission. | Based on known eICU naming conventions. A bounded sample contained numeric values; no values are reproduced here. |
| `dischargeweight` | Patient weight recorded at ICU discharge. | Based on known eICU naming conventions. A bounded sample contained numeric values; no values are reproduced here. |
| `unitdischargetime24` | De-identified ICU-unit discharge clock time in 24-hour format. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `unitdischargeoffset` | ICU-unit discharge offset relative to ICU admission, generally in minutes. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `unitdischargelocation` | Destination or location after ICU-unit discharge. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `unitdischargestatus` | Patient status at ICU-unit discharge. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `uniquepid` | De-identified patient identifier used to link encounters for the same patient. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |

### `physionet.org/files/eicu-crd-demo/2.0.1/physicalExam.csv.gz`

- **Columns:** 6
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `physicalexamid` | Row or entity identifier for the physicalexam record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `physicalexamoffset` | Offset of the physical-exam event relative to ICU admission. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `physicalexampath` | Hierarchical source-interface path for the physical-exam item. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `physicalexamvalue` | Structured value of the physical-exam item. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `physicalexamtext` | Text-form value of the physical-exam item. | Based on known eICU naming conventions. A bounded sample contained integer-like, numeric, text values; no values are reproduced here. |

### `physionet.org/files/eicu-crd-demo/2.0.1/respiratoryCare.csv.gz`

- **Columns:** 34
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `respcareid` | Row or entity identifier for the respcare record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `respcarestatusoffset` | Offset of the respiratory-care status record relative to ICU admission. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `currenthistoryseqnum` | Sequence number for the current respiratory-care history record. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `airwaytype` | Recorded airway-device type. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `airwaysize` | Recorded airway-device size. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `airwayposition` | Recorded airway-device position. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `cuffpressure` | Recorded airway cuff pressure. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `ventstartoffset` | Offset when the current ventilation episode started. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `ventendoffset` | Offset when the current ventilation episode ended. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `priorventstartoffset` | Offset when the prior ventilation episode started. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `priorventendoffset` | Offset when the prior ventilation episode ended. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `apneaparams` | Recorded apnea-ventilation parameters. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `lowexhmvlimit` | Configured respiratory or ventilator limit for lowexhmv. | Inferred from the column name and respiratory-care table context; exact units are uncertain. The first five rows were empty for this field, so no sample-based clarification was available. |
| `hiexhmvlimit` | Configured respiratory or ventilator limit for hiexhmv. | Inferred from the column name and respiratory-care table context; exact units are uncertain. The first five rows were empty for this field, so no sample-based clarification was available. |
| `lowexhtvlimit` | Configured respiratory or ventilator limit for lowexhtv. | Inferred from the column name and respiratory-care table context; exact units are uncertain. The first five rows were empty for this field, so no sample-based clarification was available. |
| `hipeakpreslimit` | Configured respiratory or ventilator limit for hipeakpres. | Inferred from the column name and respiratory-care table context; exact units are uncertain. The first five rows were empty for this field, so no sample-based clarification was available. |
| `lowpeakpreslimit` | Configured respiratory or ventilator limit for lowpeakpres. | Inferred from the column name and respiratory-care table context; exact units are uncertain. The first five rows were empty for this field, so no sample-based clarification was available. |
| `hirespratelimit` | Configured respiratory or ventilator limit for hiresprate. | Inferred from the column name and respiratory-care table context; exact units are uncertain. The first five rows were empty for this field, so no sample-based clarification was available. |
| `lowrespratelimit` | Configured respiratory or ventilator limit for lowresprate. | Inferred from the column name and respiratory-care table context; exact units are uncertain. The first five rows were empty for this field, so no sample-based clarification was available. |
| `sighpreslimit` | Configured respiratory or ventilator limit for sighpres. | Inferred from the column name and respiratory-care table context; exact units are uncertain. The first five rows were empty for this field, so no sample-based clarification was available. |
| `lowironoxlimit` | Configured respiratory or ventilator limit for lowironox. | Inferred from the column name and respiratory-care table context; exact units are uncertain. The first five rows were empty for this field, so no sample-based clarification was available. |
| `highironoxlimit` | Configured respiratory or ventilator limit for highironox. | Inferred from the column name and respiratory-care table context; exact units are uncertain. The first five rows were empty for this field, so no sample-based clarification was available. |
| `meanairwaypreslimit` | Configured respiratory or ventilator limit for meanairwaypres. | Inferred from the column name and respiratory-care table context; exact units are uncertain. The first five rows were empty for this field, so no sample-based clarification was available. |
| `peeplimit` | Configured respiratory or ventilator limit for peep. | Inferred from the column name and respiratory-care table context; exact units are uncertain. The first five rows were empty for this field, so no sample-based clarification was available. |
| `cpaplimit` | Configured respiratory or ventilator limit for cpap. | Inferred from the column name and respiratory-care table context; exact units are uncertain. The first five rows were empty for this field, so no sample-based clarification was available. |
| `setapneainterval` | Configured apnea-backup setting for interval. | Inferred from the column name and respiratory-care table context; exact units are uncertain. The first five rows were empty for this field, so no sample-based clarification was available. |
| `setapneatv` | Configured apnea-backup setting for tv. | Inferred from the column name and respiratory-care table context; exact units are uncertain. The first five rows were empty for this field, so no sample-based clarification was available. |
| `setapneaippeephigh` | Configured apnea-backup setting for ippeephigh. | Inferred from the column name and respiratory-care table context; exact units are uncertain. The first five rows were empty for this field, so no sample-based clarification was available. |
| `setapnearr` | Configured apnea-backup setting for rr. | Inferred from the column name and respiratory-care table context; exact units are uncertain. The first five rows were empty for this field, so no sample-based clarification was available. |
| `setapneapeakflow` | Configured apnea-backup setting for peakflow. | Inferred from the column name and respiratory-care table context; exact units are uncertain. The first five rows were empty for this field, so no sample-based clarification was available. |
| `setapneainsptime` | Configured apnea-backup setting for insptime. | Inferred from the column name and respiratory-care table context; exact units are uncertain. The first five rows were empty for this field, so no sample-based clarification was available. |
| `setapneaie` | Configured apnea-backup setting for ie. | Inferred from the column name and respiratory-care table context; exact units are uncertain. The first five rows were empty for this field, so no sample-based clarification was available. |
| `setapneafio2` | Configured apnea-backup setting for fio2. | Inferred from the column name and respiratory-care table context; exact units are uncertain. The first five rows were empty for this field, so no sample-based clarification was available. |

### `physionet.org/files/eicu-crd-demo/2.0.1/respiratoryCharting.csv.gz`

- **Columns:** 7
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `respchartid` | Row or entity identifier for the respchart record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `respchartoffset` | Offset of the respiratory chart event relative to ICU admission. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `respchartentryoffset` | Offset when the respiratory chart event was entered. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `respcharttypecat` | Category of the respiratory chart item. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `respchartvaluelabel` | Display label for the respiratory chart value. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `respchartvalue` | Recorded respiratory chart value. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |

### `physionet.org/files/eicu-crd-demo/2.0.1/treatment.csv.gz`

- **Columns:** 5
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `treatmentid` | Row or entity identifier for the treatment record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `treatmentoffset` | Offset of the treatment event relative to ICU admission. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `treatmentstring` | Hierarchical or descriptive treatment entry. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |
| `activeupondischarge` | Indicator of whether the documented item remained active at ICU discharge. | Based on known eICU naming conventions. A bounded sample contained text values; no values are reproduced here. |

### `physionet.org/files/eicu-crd-demo/2.0.1/vitalAperiodic.csv.gz`

- **Columns:** 13
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `vitalaperiodicid` | Row or entity identifier for the vitalaperiodic record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `observationoffset` | Offset of the vital-sign observation relative to ICU admission. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `noninvasivesystolic` | Non-invasive systolic blood pressure. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `noninvasivediastolic` | Non-invasive diastolic blood pressure. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `noninvasivemean` | Non-invasive mean blood pressure. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `paop` | Pulmonary artery occlusion pressure. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `cardiacoutput` | Recorded cardiac output. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `cardiacinput` | Recorded cardiac index or related cardiac-input value; exact source meaning is unclear. | Uncertain; cautious interpretation of an abbreviated eICU/APACHE field name. The first five rows were empty for this field, so no sample-based clarification was available. |
| `svr` | Systemic vascular resistance. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `svri` | Systemic vascular resistance index. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `pvr` | Pulmonary vascular resistance. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `pvri` | Pulmonary vascular resistance index. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |

### `physionet.org/files/eicu-crd-demo/2.0.1/vitalPeriodic.csv.gz`

- **Columns:** 19
- **Inspection status:** Header and up to five rows read successfully.

| Column Name | One-line Explainer | Notes / Inference Basis |
|---|---|---|
| `vitalperiodicid` | Row or entity identifier for the vitalperiodic record. | Based on known eICU identifier naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `patientunitstayid` | De-identified identifier for one ICU unit stay. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `observationoffset` | Offset of the vital-sign observation relative to ICU admission. | Based on known eICU offset naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `temperature` | Recorded temperature measurement. | Directly obvious from the column name. The first five rows were empty for this field, so no sample-based clarification was available. |
| `sao2` | Arterial oxygen saturation. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `heartrate` | Recorded heart rate. | Directly obvious from the column name. A bounded sample contained integer-like values; no values are reproduced here. |
| `respiration` | Recorded respiratory rate or respiration measurement. | Based on known eICU naming conventions. A bounded sample contained integer-like values; no values are reproduced here. |
| `cvp` | Central venous pressure. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `etco2` | End-tidal carbon dioxide measurement. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `systemicsystolic` | Invasive systemic systolic blood pressure. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `systemicdiastolic` | Invasive systemic diastolic blood pressure. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `systemicmean` | Invasive systemic mean blood pressure. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `pasystolic` | Pulmonary artery systolic pressure. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `padiastolic` | Pulmonary artery diastolic pressure. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `pamean` | Pulmonary artery mean pressure. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `st1` | ST-segment measurement from monitoring channel 1. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `st2` | ST-segment measurement from monitoring channel 2. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `st3` | ST-segment measurement from monitoring channel 3. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
| `icp` | Intracranial pressure. | Based on known eICU naming conventions. The first five rows were empty for this field, so no sample-based clarification was available. |
