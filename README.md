# Smoking Status Extraction from Clinical Notes (MedCAT + Structured ICD)

Extracting per-patient smoking status from a MIMIC cohort by combining **MedCAT NLP over
clinical free text** with **structured ICD-9 diagnosis codes**, then reconciling the two
sources into a single status per patient.

CogStack Forward Deployed Engineer technical task (the accompanying 3-slide presentation is included as Presentation_Cogstack.pdf). 

---

## Headline finding

**The smoking signal is in the text, not the codes.** Clinical notes yielded a smoking
status for **518 admissions**; ICD-9 codes captured only **129** (~4× fewer) — and ICD-9
has no code for *"never smoked"* at all. Free text is by far the richer source, but needs
NLP and careful cleaning to use safely.

Final patient-level distribution (995 patients):

| Status  | Patients |
|---------|----------|
| Current | 297 |
| Former  | 116 |
| Never   | 130 |
| Unknown | 452 |

---

## Approach

1. **Targeted the high-yield note type.** The cohort has 34,619 notes across 1,000
   admissions (~35 each). Smoking/social history lives almost entirely in **discharge
   summaries** (1,069 of them), so these were processed rather than all note types —
   a deliberate precision/compute trade-off for the timebox.
2. **Ran MedCAT** with the provided `smoking_findings_only` model pack (filtered to
   tobacco/smoking SNOMED concepts) over every discharge summary, capturing each concept
   mention plus a surrounding text window.
3. **Cleaned & classified** mentions into `current` / `former` / `never` using context
   rules — dropping false positives (e.g. `ex` matched inside "ex-lap"), negations
   ("denies smoking" → never), and discharge advice ("smoking cessation" → not a status).
   This rule layer stands in for the **MetaCAT status model the pack does not include**.
4. **Joined structured ICD-9 smoking codes** (e.g. `3051` current, `V1582` former) as an
   independent second source.
5. **Resolved to one status per patient**, prioritising `current > former > never`,
   recording confidence, evidence source, and any conflicting-evidence flag.

---

## Output

`outputs/patient_smoking_status.csv`

| Column | Description |
|--------|-------------|
| `patient_id` | SUBJECT_ID |
| `smoking_status` | current / former / never / unknown |
| `confidence` | high / conflicting / none |
| `source_doc_count` | number of admissions contributing evidence |
| `notes` | evidence source (note / note+icd / icd_only) or conflict flag |

Charts used in the presentation are also in `outputs/`.

---

## How to run

The data files and model pack are **not** committed (see *Data & model setup* below).
Once they are in place:

```bash
python -m venv venv311
# Windows:  venv311\Scripts\activate
# macOS/Linux:  source venv311/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_md

cd notebooks
python Analysis.py
```

`Analysis.py` is written as a cell-delimited script (`# %%`) so it can also be run
interactively in VS Code / Jupyter. The full MedCAT pass over ~1,069 discharge summaries
takes roughly 10–12 minutes.

### Data & model setup (not committed)

Place the provided files as follows (these are gitignored — see *Repository hygiene*):

```
data/
    NOTEEVENTS.csv
    DIAGNOSES_ICD_1000HADM_IDs.csv
models/
    smoking_findings_only_<hash>.zip
```

---

## MedCAT vs where the model breaks

MedCAT reliably finds smoking concepts, but the provided pack has **no MetaCAT status
model** (every `meta_anns` is empty), so it cannot itself distinguish affirmed from
negated/hypothetical mentions. Observed failure modes:

- **Negation** — "did not smoke", "denies tobacco" flagged as smoking.
- **Discharge advice** — "importance of smoking cessation" is guidance, not patient status.
- **False positives** — `ex` inside "ex-lap" → Ex-smoker; "every day" → Daily.
- **Conflicts** — 61 admissions (~12%) carried internally contradictory mentions.
- **Coverage** — 452 patients (45%) have no smoking evidence in a discharge summary and
  are honestly recorded as `unknown` rather than guessed.

Where both sources exist, notes and ICD agreed **73%** of the time (74/101).

---

## MedCATtrainer

MedCATtrainer was installed and run locally via Docker; a project was created and
documents annotated (marking correct concepts and rejecting false positives) to exercise
the review workflow.

**Note on the provided pack:** `smoking_findings_only` is in **MedCAT v2** format (the CDB
is a folder containing a ~255 MB `raw_dict.dat`, plus `model_card.json`). The current
public MedCATtrainer image runs **MedCAT v1.16**, whose loader (`CAT.load_cdb`) expects a
flat `cdb.dat` file — so the pack's CDB extracts empty (~4 KB stub), no concepts import,
and nothing highlights.

Attempts to bridge this:

- **Convert v2 → v1**: not possible — MedCAT v2's serialiser only writes the folder format,
  and no v2→v1 converter exists (`cdb.py`'s version check only warns to "download the
  compatible model").
- **Rebuild the Trainer on MedCAT v2**: the image builds, but the Django app targets the
  v1 API and won't start.

CogStack archived the v1 MedCAT/MedCATtrainer repos in mid-2025 and consolidated into
[`cogstack-nlp`](https://github.com/CogStack/cogstack-nlp), where a v2-era trainer is in
active development — but the v2 model/trainer tooling is still mid-transition. This was
flagged to the team. For the analysis, the pack was run via the **MedCAT v2 library
directly** (see `Analysis.py`), and the Trainer workflow demonstrated with a compatible
model. See `outputs/medcat_mismatch_flowchart.png`.

---

## Next steps to make this production-ready

- **Add a trained MetaCAT status model** — the single biggest win; learns affirmed vs
  negated vs hypothetical, replacing the hand-written rules.
- **Tighten concept filtering** to eliminate substring false positives.
- **Widen beyond discharge summaries** (physician/nursing notes) to recover `unknown`s.
- **Capture temporality / pack-years** — "quit 20 years ago" vs "quit last week" both read
  as `former` today.
- **Validate against clinician annotation** for real precision/recall/F1 before any
  clinical use.

---

## Repository hygiene

The repo deliberately excludes, via `.gitignore`:

- MedCAT model packs (`models/`, `*.zip`) — including `smoking_findings_only`
- Raw MIMIC extracts and note text (`data/`, and the raw-text working files
  `outputs/mct_dataset.csv`, `outputs/_mentions_cache.csv`)
- Credentials (`.env`) and the virtual environment (`venv311/`)

## Repository contents

```
notebooks/Analysis.py - Full extraction + analysis pipeline
outputs/patient_smoking_status.csv - The deliverable
outputs/chart_*.png - Figures used in the presentation
outputs/medcat_mismatch_flowchart.png  - The v2/v1 Trainer issue, visualised
Presentation_Cogstack.pdf - 3-slide presentation (methodology, findings, next steps)
README.md
requirements.txt
```
