# Dataset-vs-Plan Audit Report

## 1. Executive Summary

The repository contains a complete eICU-CRD demo v2.0.1 distribution: 31 compressed CSV tables, a matching compressed SQLite database, and 2,520 ICU unit stays. The central proposal in `docs/plan.md` is technically feasible in reduced form: the data supports ICU-stay event construction from diagnoses, labs, medications, infusions, treatments, and periodic/aperiodic vitals; masked event pretraining; and hospital or ICU mortality prediction.

The plan is not fully aligned with the available data. Its most important factual error is the repeated claim that the demo removes hospital and unit identifiers. In this local dataset, `patient.csv.gz` has complete `hospitalid` and `wardid` fields: 186 distinct hospitals and 292 distinct wards. The `hospital.csv.gz` table also has one row for each of the 186 hospitals. The claim that the demo contains stays from 20 hospitals is inconsistent with the available files.

The largest methodological risk is outcome leakage. The plan proposes mortality prediction from a chronological stream built across each entire ICU stay, but it does not define a prediction time, observation window, or exclusion of post-outcome/proxy fields. Full-stay diagnoses, treatments, medications, and discharge-active flags can contain information recorded after the intended prediction point. `apachePatientResult` additionally includes actual and predicted mortality fields and must not be used as input features for mortality prediction.

Other major adjustments are required:

- Split by `uniquepid`, not independently by `patientunitstayid`, because 416 patients have multiple ICU stays.
- Define an observation window, preferably the first 24 hours after ICU admission for the primary mortality task.
- Handle negative offsets explicitly; they are common in labs and medications.
- Aggregate or subsample dense vitals before applying `max_seq_len: 128`; the median planned event count is 614 and 2,274 of 2,520 stays exceed 128 events.
- Fit variable-specific quantile bins on the training partition only. Dataset-wide or pooled-variable quantiles would leak information and produce clinically incoherent bins.
- Use `patient.hospitaldischargestatus` as the primary hospital-mortality label. It covers 2,492 stays, whereas `apachePatientResult` covers only 1,838 stays and has two rows per covered stay (APACHE IV and IVa).
- Replace pseudo-client-only evaluation with an exploratory hospital-grouped evaluation if site grouping is desired, while retaining strong caveats because each hospital contributes only 10-40 stays and 69 hospitals have no deaths.

Overall verdict: **feasible after major plan corrections**. The core event-pretraining prototype is supported, but the current outcome protocol, split protocol, site claims, sequence construction, and experiment scale are not sufficiently dataset-grounded.

## 2. Dataset Inventory

Raw root inspected:

`data/raw/eicu_demo/physionet.org/files/eicu-crd-demo/2.0.1/`

Row counts below exclude the CSV header. Descriptions are based on the inspected headers and aggregate availability, not on filenames alone.

| Raw file | Rows | Main contents and key fields |
|---|---:|---|
| `admissionDx.csv.gz` | 7,578 | Admission diagnoses; `patientunitstayid`, `admitdxenteredoffset`, diagnosis path/name/text. |
| `admissiondrug.csv.gz` | 7,417 | Drugs present around admission; drug/entry offsets, name, dose, unit, frequency, HICL sequence. |
| `allergy.csv.gz` | 2,475 | Allergy records; event/entry offsets, allergy/drug names and type. |
| `apacheApsVar.csv.gz` | 2,205 | APACHE acute physiology variables and severity inputs for 2,205 rows/stays. |
| `apachePatientResult.csv.gz` | 3,676 | APACHE IV/IVa scores, predicted and actual ICU/hospital mortality and length of stay; 1,838 unique stays. |
| `apachePredVar.csv.gz` | 2,205 | APACHE prediction covariates, comorbidities, day-one variables, and `diedinhospital`. |
| `carePlanCareProvider.csv.gz` | 5,627 | Care-provider type, specialty, intervention category, save offset. |
| `carePlanEOL.csv.gz` | 15 | End-of-life discussion records; extremely sparse. |
| `carePlanGeneral.csv.gz` | 33,148 | General care-plan items and offsets. |
| `carePlanGoal.csv.gz` | 3,633 | Care goals, categories, status, and offsets. |
| `carePlanInfectiousDisease.csv.gz` | 112 | Infectious-disease assessment and treatment; extremely sparse. |
| `customLab.csv.gz` | 30 | Custom laboratory results; extremely sparse. |
| `diagnosis.csv.gz` | 24,978 | Diagnoses for 2,155 stays; offset, diagnosis string, optional ICD-9 code, priority. |
| `hospital.csv.gz` | 186 | Hospital metadata; `hospitalid`, bed-count category, teaching status, and region. |
| `infusiondrug.csv.gz` | 38,256 | Infusion events for 826 stays; offset, drug name, rates, amount, fluid volume, weight. |
| `intakeOutput.csv.gz` | 100,466 | Intake/output observations, totals, entry offsets, labels, numeric/text values. |
| `lab.csv.gz` | 434,660 | Laboratory results for 2,444 stays; result/revision offsets, name, numeric and text result fields. |
| `medication.csv.gz` | 75,604 | Medication orders for 1,857 stays; order/start/stop offsets, drug, dose, route, frequency. |
| `microLab.csv.gz` | 342 | Microbiology culture site, organism, antibiotic, sensitivity, and collection offset. |
| `note.csv.gz` | 24,758 | Note metadata and text fields with note/entry offsets. |
| `nurseAssessment.csv.gz` | 91,589 | Nursing assessment attributes and values with event/entry offsets. |
| `nurseCare.csv.gz` | 42,080 | Nursing-care attributes and values with event/entry offsets. |
| `nurseCharting.csv.gz` | 1,477,163 | Dense nursing chart observations with category, label, name, value, and offsets. |
| `pastHistory.csv.gz` | 12,109 | Past-history entries with paths, values, text, and offsets. |
| `patient.csv.gz` | 2,520 | One row per ICU unit stay; patient/encounter/stay IDs, demographics, hospital/ward, admission/discharge context, outcomes, and offsets. |
| `physicalExam.csv.gz` | 84,058 | Physical-exam observations with offset, path, value, and text. |
| `respiratoryCare.csv.gz` | 5,436 | Airway/ventilation settings and ventilation episode offsets. |
| `respiratoryCharting.csv.gz` | 176,089 | Respiratory chart events with type, label, value, and offsets. |
| `treatment.csv.gz` | 38,290 | Treatment/procedure strings for 1,910 stays; treatment offset and active-at-discharge flag. |
| `vitalAperiodic.csv.gz` | 274,088 | Aperiodic blood pressure and haemodynamic measurements for 2,331 stays. |
| `vitalPeriodic.csv.gz` | 1,634,960 | Periodic temperature, oxygen saturation, heart rate, respiration, pressures, ECG ST values, and ICP for 2,375 stays. |
| `sqlite/eicu_v2_0_1.sqlite3.gz` | n/a | SQLite packaging of the same 31 tables and 391 columns; no declared primary keys, foreign keys, or indexes. |
| `LICENSE.txt` | n/a | Dataset licence text. |
| `SHA256SUMS.txt` | n/a | Checksums for distributed files. |
| `index.html`, `sqlite/index.html` | n/a | Distribution index pages. |

Key identifiers:

- `patientunitstayid`: ICU unit-stay key; present in 30 clinical tables.
- `patienthealthsystemstayid`: links ICU stays within one hospital-system encounter; only in `patient`.
- `uniquepid`: de-identified patient key across encounters; only in `patient`.
- `hospitalid`: complete in `patient`; 186 distinct values.
- `wardid`: complete in `patient`; 292 distinct values.

Key outcomes:

- `patient.hospitaldischargestatus`: 2,280 Alive, 212 Expired, 28 missing.
- `patient.unitdischargestatus`: 2,392 Alive, 126 Expired, 2 missing.
- `apachePatientResult.actualhospitalmortality`: 3,366 ALIVE and 310 EXPIRED rows, but these are duplicated across APACHE IV and IVa for 1,838 stays.
- `apachePatientResult.actualicumortality`: 3,494 ALIVE and 182 EXPIRED rows, with the same version duplication.

## 3. Metadata Inventory

| Metadata file | Contents | Audit use |
|---|---|---|
| `.data/eicu_demo_headers.md` | Header-level summary of all 31 CSV files, 391 columns, identifier fields, offset fields, and per-table schemas. It reports inspection of headers plus at most five rows per file without reproducing patient rows. | Confirmed CSV schemas, casing, identifier availability, and candidate time fields. |
| `.data/eicu_demo_sqlite.md` | SQLite catalog analysis, type families, nullability, table overview, and all table schemas. It confirms case-insensitive structural equivalence with the CSV distribution. | Confirmed that SQLite adds no keys, foreign keys, views, triggers, or indexes and should not be treated as an enforced relational schema. |

No separate data dictionary defining clinical units, valid ranges, sentinel values, or table-specific offset semantics was found in `.data/`. Those rules must be encoded from authoritative eICU documentation or conservatively inferred and documented; they cannot be obtained from these two metadata summaries alone.

## 4. Key Dataset Capabilities

The available data supports:

- ICU unit-stay event streams keyed by `patientunitstayid`.
- Patient-level leakage control using `uniquepid`.
- Hospital-system encounter grouping using `patienthealthsystemstayid`.
- Site-aware grouping using complete `hospitalid` and `wardid` values.
- Relative-time ordering using table-specific minute offsets.
- Static context from age, gender, ethnicity, admission source, unit type, hospital, and ward.
- Diagnosis events from diagnosis strings and partially available ICD-9 codes (21,206 of 24,978 diagnosis rows).
- Numeric and text laboratory events; 431,939 of 434,660 rows have `labresult` and 434,635 have `labresulttext`.
- Dense periodic and aperiodic vital-sign events.
- Medication, infusion, and treatment events, with table-specific missingness and terminology issues.
- Hospital and ICU mortality labels.
- APACHE severity variables and APACHE IV/IVa result records for subsets of stays.
- Exploratory site-grouped analysis, subject to small per-hospital samples.

Important scale characteristics:

- 2,520 ICU stays represent 2,174 hospital-system encounters and 1,841 patients.
- 416 patients have more than one ICU stay; one patient has 24 stays.
- Across the seven planned event tables, the median stay has 614 source rows, the 95th percentile has 3,160, and the maximum is 19,032.
- 2,274 stays exceed 128 source events and 2,067 exceed 256.
- 31 stays have no rows in the seven planned event tables; 39 have fewer than five.
- Periodic vitals contribute 1,634,960 of approximately 2.52 million rows across the seven planned event tables and will dominate an unbalanced token stream.
- Negative offsets occur in all seven planned event tables, including 97,488 lab rows and 22,853 medication rows.

The available data does not directly support:

- Calendar-time or prospective temporal validation. Only limited de-identified time fields such as discharge year and clock times are present; there are no reliable longitudinal calendar timestamps for event rows.
- Clinical reference-range labels such as LOW/NORMAL/HIGH for all variables. The lab table has no per-result lower/upper reference-range columns.
- Claims of deployment readiness, causal effects, prospective performance, or privacy guarantees.
- Stable hospital-level mortality estimates for every site. Hospital sample sizes range from 10 to 40 stays and 69 of 186 hospitals have no deaths in the demo.

## 5. Plan Assumptions Checked Against Data

| Plan section | Assessment | Dataset-grounded finding |
|---|---|---|
| Opening dataset description and Sections 1, 4, 22 | Contradicted | Hospital and ward identifiers are present and complete. There are 186 hospitals, not 20, in the local data. |
| Section 2: patient-level ICU event streams | Partially supported | Streams can be built, but the proposed key creates ICU **stay-level** streams. Patient-level splitting requires `uniquepid`. |
| Section 3: MVP tables | Supported with caveats | All nine named logical tables exist. The raw filename is `infusiondrug.csv.gz`, not `infusionDrug.csv.gz`; Linux code must map exact casing. |
| Section 3: `apachePatientResult` as severity/outcome information | Partially supported | It covers only 1,838 stays and contains two version rows per stay. Predicted and actual mortality columns are leakage if used as features. |
| Section 5: chronological event sequence | Supported with major specification gap | All planned event tables have offsets, but negative offsets and table-specific time meanings require an explicit policy. |
| Section 5: static context | Supported | Age, gender, admission source, and unit type exist, with missingness. `unitadmitsource` is much more complete than `hospitaladmitsource` (22 versus 594 missing). |
| Section 5: LOW/NORMAL/HIGH bins | Not generally supported | Reference ranges are absent. Training-only, variable-specific quantile bins are feasible. |
| Section 6: 128-token Transformer | Technically supported but lossy | Most stays exceed 128 source events; aggregation/windowing is mandatory before truncation. |
| Section 7: masked event modelling | Supported | Categorical/discretised event tokens can be masked and predicted. Event-family imbalance must be controlled. |
| Section 8: hospital or ICU mortality | Supported | Both labels are available. Hospital mortality has broader coverage and 212 positive outcomes in `patient`. |
| Sections 9-11: experiment matrix | Partially supported | The experiments are implementable, but the sample size, class imbalance, and CPU target make the full matrix plus tuning statistically and computationally ambitious. |
| Section 10: random split | Unsafe as written | A stay-random split can place stays from the same patient in multiple partitions. |
| Sections 10-11: pseudo-clients | Implementable but artificial | Density/diagnosis/context groups are available, but they are cohort strata rather than institutional clients. Real de-identified hospital groups are present. |
| Section 11: FedAvg simulation | Technically supported | It remains a simulation. Site-based clients are possible, but most individual hospitals are too small for stable local mortality training. |
| Section 11: rare-pattern memorisation probe | Technically supported with limits | Token n-gram diagnostics are possible, but cannot establish privacy or membership-inference risk. |
| Section 14: `min_events_per_stay: 5` | Supported | It would exclude 39 of 2,520 stays based on the seven planned event tables. The plan should state this effect. |
| Section 14: mortality config | Underspecified | The label source, observation window, exclusion criteria, grouping key, and handling of missing labels are absent. |
| Sections 15 and 20: raw path | Potentially inconsistent | The actual CSVs are nested below `data/raw/eicu_demo/physionet.org/files/eicu-crd-demo/2.0.1/`; scripts must discover this path or the config must name it. |
| Section 20: eight-day schedule | Risky | Preprocessing 2.5 million planned event rows is feasible, but 13 experiment blocks plus 15 Optuna trials and reporting on CPU is unlikely to be robustly completed in the stated schedule without aggressive reduction. |

## 6. Discrepancies Found

| Severity | Plan Claim / Section | Dataset Evidence | Problem | Recommended Correction |
|---|---|---|---|---|
| Critical | Opening text, Sections 1, 4, and 22: hospital and unit identifiers are removed; no true hospital-level evaluation is possible. | `patient.csv.gz` contains complete `hospitalid` and `wardid`: 186 hospitals and 292 wards. `hospital.csv.gz` has 186 rows. | The core site-data premise is factually wrong and causes the plan to ignore available grouping variables. | State that de-identified site IDs are present. Permit exploratory hospital-grouped evaluation, but do not claim generalisation because the demo is small and selectively sampled. |
| Critical | Opening text: over 2,500 stays selected from 20 larger hospitals. | The local `patient` table has 2,520 stays across 186 distinct `hospitalid` values; hospital metadata has 186 rows. | The hospital count is inconsistent with the source-of-truth files. | Replace the count with the locally verified figures and cite the exact demo version. Do not retain “20 hospitals” unless authoritative documentation explains a different sampling definition. |
| Critical | Sections 5 and 8: build each complete stay stream, then predict mortality. | Event offsets extend far into a stay (up to 525,599 minutes for medication), and some fields describe discharge state or late care. | Using full-stay events to predict discharge mortality creates severe temporal and target leakage and makes the task clinically ill-defined. | Define an index time and observation window, e.g. events with `0 <= offset <= 1,440` minutes to predict hospital mortality after the first 24 hours. Exclude stays discharged before the horizon or define their handling prospectively. |
| Critical | Section 3: use `apachePatientResult` as severity/outcome-related information without a feature exclusion rule. | The table includes actual/predicted ICU and hospital mortality and LOS. It has APACHE IV and IVa rows for each of 1,838 stays. | Mortality and predicted-mortality columns directly leak the target; duplicate version rows can duplicate stays. | Do not include `apachePatientResult` in event features for mortality prediction. Use it only for label cross-checks, optional non-target severity analyses, or carefully selected baseline covariates with leakage columns removed and one version chosen. |
| Major | Sections 10 and 14: `random split` without a grouping key. | 2,520 stays correspond to 1,841 patients; 416 patients have multiple stays. | Independent stay splitting can leak patient-specific patterns across train, validation, and test sets. | Stratify outcomes while grouping all records by `uniquepid`; at minimum group by `patienthealthsystemstayid`. Fit vocabulary and bins on training patients only. |
| Major | Sections 5, 6, and 14: chronological streams with `max_seq_len: 128`, without aggregation/truncation policy. | Median source-row count is 614; 2,274 stays exceed 128 and the maximum is 19,032. | Naive truncation discards most information and can systematically favor early or late events. Dense vitals dominate the sequence. | Define a fixed observation window, bucket vitals (for example 15- or 60-minute summaries), deduplicate simultaneous events, cap per-family tokens, and document deterministic truncation. |
| Major | Section 5: quantile bins for labs and vitals, unspecified scope. | Many different variables and units share the same tables. Reference ranges are absent; numeric completeness varies by variable. | Global quantiles across variables are meaningless; dataset-wide fitting leaks validation/test distributions. | Fit bins separately for each normalized measurement name using training data only. Add minimum support, missing-value, nonnumeric, unit, clipping, and unseen-variable rules. |
| Major | Section 5: LOW/NORMAL/HIGH/VERY_HIGH as an alternative. | No lab reference-range columns are present. | Clinical normality cannot be inferred reliably from the provided tables alone. | Remove this as the default unless externally documented reference ranges and unit normalization are added. Retain quantile bins as the dataset-supported option. |
| Major | Sections 3 and 5: all named medication rows become medication tokens. | `medication` has 75,604 rows, but `drugname` is populated in only 44,943; `dosage` in 66,299. `routeadmin` is nearly complete. | A drug-name-only parser would silently lose about 41% of medication rows or create meaningless missing-name tokens. | Define a hierarchy using normalized `drugname`, HICL/`gtc` where available, and explicit row exclusion. Report retained-row coverage. Do not tokenize route alone as a drug identity. |
| Major | Section 5: offsets can be directly sorted into one sequence. | Negative offsets include 97,488 lab, 22,853 medication, 1,650 periodic-vital, and smaller numbers of other planned rows. Offset fields also represent different semantics (result, observation, order/start, or documentation time). | Mixing pre-ICU, order, result, and documentation times without rules produces inconsistent chronology and can include prior-encounter context unintentionally. | Specify one timestamp per table, whether pre-ICU events are retained, and an allowed offset range. Document tie-breaking and same-minute aggregation. |
| Major | Sections 10-11: default pseudo-client grouping by event density is the main replacement for site evaluation. | Actual de-identified hospital IDs exist. Density groups are derived from the representation and have different acuity/documentation distributions by construction. | Density groups are not clients and can turn a distribution-shift ablation into a misleading federated-learning analogue. | Use density grouping only as a stress test. Prefer grouped holdout by hospital or larger site clusters for exploratory evaluation, with explicit sample-size caveats. |
| Major | Sections 10-11: hospital/client-style evaluation implied by three clients. | Hospitals have 10-40 stays each; 69 of 186 have zero deaths. | Per-hospital mortality training/evaluation is unstable and many clients cannot support binary metrics. | Combine hospitals into a small number of deterministic site clusters using metadata or use leave-cluster-out evaluation. Never report per-site AUROC where only one class is present. |
| Major | Sections 8-11: single random evaluation and tuning on a small imbalanced label set. | `patient.hospitaldischargestatus` has 212 deaths among 2,492 labelled stays (8.5%). | AUROC/AP/F1 can vary materially by split; 15-trial tuning risks selecting split noise. | Use stratified patient-group splits, an untouched test set, repeated seeds or bootstrap confidence intervals, and validation-only tuning. Report class counts for every partition. |
| Major | Section 20: 13 experiment blocks plus 10-20 tuning trials in eight days on CPU/8 GB RAM. | The seven planned event tables contain about 2.52 million rows before aggregation; most stays exceed planned sequence length. | The schedule leaves insufficient time for leakage controls, robust preprocessing, repeated evaluation, and debugging. | Prioritize one baseline, scratch versus pretrained comparison, one grouped evaluation, and a reduced tuning study. Make FedAvg and memorisation probes optional after the core protocol is validated. |
| Minor | Section 3 names `infusionDrug`. | Actual filename is `infusiondrug.csv.gz`; metadata also uses lowercase table name. | Case-sensitive path construction will fail on Linux. | Use the exact filename or an explicit logical-table-to-file mapping. |
| Minor | Section 5 example uses generic admission-source token and Section 10 suggests admission context. | `hospitaladmitsource` is missing in 594 stays; `unitadmitsource` is missing in 22. | An unspecified source field creates avoidable missingness and inconsistent experiments. | Prefer `unitadmitsource`, preserve an explicit unknown category, and document which source field is used. |
| Minor | Section 8 leaves the primary mortality label as hospital/discharge-style with a fallback. | `patient.hospitaldischargestatus` and `unitdischargestatus` are directly available with different positive counts; APACHE labels have narrower coverage. | Multiple interchangeable labels undermine reproducibility and make runs incomparable. | Fix `patient.hospitaldischargestatus == Expired` as primary. Define ICU mortality as a separate secondary task only. |
| Minor | Section 14 `min_events_per_stay: 5` has no cohort impact statement. | 39 stays have fewer than five planned event rows; 31 have none. | The cohort silently changes and may differentially exclude sparse sites or outcomes. | Report exclusions by outcome and hospital, and define whether static-only stays are excluded before splitting. |
| Minor | Sections 15 and 20 assume `data/raw/eicu_demo` directly contains tables. | Tables are nested under `physionet.org/files/eicu-crd-demo/2.0.1/`. | A non-recursive loader will report missing files. | Resolve the versioned nested directory explicitly or implement deterministic discovery with an ambiguity error. |
| Informational | Section 2 calls the streams patient-level. | `patientunitstayid` identifies one ICU unit stay; `uniquepid` identifies a patient. | Terminology may imply longitudinal patient histories when the representation is stay-specific. | Use “ICU stay-level event stream” and reserve “patient-level” for split grouping or longitudinal aggregation. |
| Informational | Section 11 rare-pattern probe links token overlap to privacy concerns. | A token n-gram overlap test can be computed, but the dataset and method do not establish membership inference or privacy leakage. | Readers may overinterpret a diagnostic as a privacy evaluation. | Keep it optional and label it a representation memorisation diagnostic with no privacy guarantee. |
| Informational | No temporal-generalisation experiment is proposed. | Event rows use relative offsets; no reliable event calendar dates are available. | This is appropriate, but future documents could accidentally imply temporal validation. | Explicitly state that temporal generalisation is not evaluated with this demo. |

## 7. Supported Parts of the Plan

- Use of the eICU demo only.
- ICU stay-level joins through `patientunitstayid`.
- Event construction from diagnosis, lab, medication, infusion, treatment, periodic vital, and aperiodic vital tables.
- Static context from age, gender, unit admission source, and unit type.
- Chronological ordering using relative offsets after table-specific rules are defined.
- Training-only quantile discretisation of numeric measurements.
- A compact PyTorch Transformer and masked event prediction objective.
- Bag-of-events logistic regression as a baseline.
- Hospital mortality and ICU mortality prediction as distinct binary tasks.
- AUROC, average precision, F1, and balanced accuracy, provided class counts and uncertainty are also reported.
- A FedAvg-style simulation, clearly framed as an engineering demonstration rather than deployed federated learning.
- A lightweight rare-token or rare-n-gram memorisation diagnostic, with the existing disclaimer.
- CPU-friendly scope in principle after event aggregation and experiment reduction.
- Data-safety rules, local raw/processed folders, and non-committable derived artifacts.

## 8. Partially Supported Parts of the Plan

- **APACHE use:** available for subsets, but outcome/prediction fields must be excluded from model inputs and duplicate APACHE versions resolved.
- **Static context:** supported, but field choices and missing categories must be fixed in advance.
- **Medication events:** supported with substantial missing drug names and terminology normalization requirements.
- **Infusion events:** supported for only 826 stays, so absence often means no recorded infusion table event rather than a reliable negative clinical state.
- **Time-gap tokens:** feasible only after defining same-time ordering, aggregation, negative-offset handling, and gap clipping.
- **128-token sequences:** feasible after deterministic temporal bucketing or family-aware sampling; not as raw-row truncation.
- **Diagnosis-group pseudo-clients:** diagnosis data covers 2,155 stays, but broad categories require a documented mapping and missing group.
- **Hospital-grouped evaluation:** identifiers support it, but the demo's per-hospital sample sizes are too small for strong site-generalisation claims.
- **FedAvg simulation:** feasible with clustered pseudo-sites or site clusters, but three event-density strata should not be presented as institutional clients.
- **Optuna tuning:** feasible at reduced scale, but 15 trials should not precede a stable leakage-controlled protocol and fixed test split.

## 9. Unsupported or Risky Parts of the Plan

- The claim that hospital and unit identifiers are removed.
- The claim that the available local demo represents 20 hospitals.
- Mortality prediction from unrestricted full-stay event streams.
- Using APACHE actual/predicted mortality or LOS fields as mortality-model features.
- Independent random splitting of ICU stays.
- LOW/NORMAL/HIGH lab bins without external reference ranges and unit normalization.
- Directly truncating raw heterogeneous sequences to 128 or 256 events without aggregation.
- Treating density, diagnosis, or admission-context strata as equivalent to real institutional clients.
- Reporting stable per-hospital mortality performance for all 186 hospitals.
- Any calendar-time/temporal-generalisation claim.
- Any claim of true multicentre generalisation, clinical readiness, causal inference, formal privacy auditing, secure federated learning, or privacy guarantees.

Additional information needed to execute the original plan without the recommended reductions:

- Authoritative eICU table documentation for offset semantics, valid values, units, and sentinel values.
- A documented medication normalization strategy or terminology mapping for drug names/HICL/GTC.
- External laboratory reference ranges if clinically meaningful LOW/NORMAL/HIGH tokens are retained.
- A predeclared mortality prediction protocol specifying horizon, eligible cohort, censoring, and label source.
- A site-clustering rule if site-aware FedAvg is attempted.
- More data per site, ideally the full credentialed eICU dataset, for credible hospital-level generalisation analysis. This would change the current demo-only scope and is not required for the reduced prototype.

## 10. Recommended Dataset-Grounded Plan Changes

1. Correct the dataset description to 2,520 ICU stays, 1,841 patients, 186 de-identified hospitals, and 292 de-identified wards in the locally available v2.0.1 files.
2. Rename the basic sample unit from “patient-level stream” to “ICU stay-level stream,” while grouping all splits by `uniquepid`.
3. Fix the primary task as hospital mortality from `patient.hospitaldischargestatus`, excluding 28 missing labels.
4. Define a primary 24-hour observation window using events with offsets from 0 through 1,440 minutes. Treat pre-ICU events as a separate ablation rather than silently mixing them into the primary stream.
5. Exclude all direct outcome, discharge-state, post-horizon, and outcome-prediction fields from features. Specifically exclude APACHE actual/predicted mortality and LOS fields, `diedinhospital`, and `activeupondischarge` fields.
6. Build train/validation/test partitions before fitting vocabularies, numeric bins, imputation rules, or category-frequency thresholds. Use stratified group splitting by `uniquepid`.
7. Reduce vital density through fixed time buckets and deterministic summaries. Then interleave event families chronologically and apply a documented truncation policy.
8. Use per-variable training-set quantiles for labs and vitals. Add explicit numeric parsing, clipping, missingness, minimum-count, and unseen-variable rules.
9. Define medication retention and normalization coverage before modelling. Keep only rows with a defensible drug identity and report discarded rows.
10. Replace the full experiment matrix with a core set: logistic baseline, scratch Transformer, pretrained Transformer, one representation ablation, and one grouped evaluation. Run tuning, FedAvg, and memorisation only after the core protocol is stable.
11. Replace pseudo-client-only site framing with two separate analyses: hospital-grouped exploratory evaluation using de-identified `hospitalid`, and event-density distribution-shift evaluation. Do not conflate them.
12. For FedAvg, use a few deterministic clusters of hospitals rather than individual small hospitals, or retain density pseudo-clients but call them synthetic cohorts, not sites.
13. Report positive/negative counts, excluded stays, patient counts, hospital counts, and confidence intervals for every final split.
14. Make the raw data path version-aware and use the exact `infusiondrug.csv.gz` casing.

## 11. Final Feasibility Verdict

**Verdict: feasible with major revisions.**

The available eICU demo is sufficient for a credible CPU-friendly prototype of heterogeneous ICU event construction, masked event pretraining, and downstream mortality prediction. No additional clinical table is strictly required for that reduced objective.

The current plan should not be executed unchanged. Its site-identifier premise is false for the local files, its mortality protocol permits severe temporal leakage, its random split can leak patients, and its raw sequence lengths are incompatible with the proposed 128-token model without explicit aggregation. The full experiment/tuning schedule is also too broad relative to 2,520 stays, 212 labelled hospital deaths, and the stated CPU-only timeline.

After correcting the site description, fixing a leakage-controlled 24-hour outcome protocol, using patient-grouped splits, aggregating dense events, and reducing the experiment matrix, the project remains aligned with the repository's intended scope and can make defensible exploratory claims about the eICU demo only.
