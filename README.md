---

## MedCAT: where the model breaks

MedCAT reliably finds smoking concepts, but the provided pack has **no MetaCAT status model** (every `meta_anns` is empty), so it cannot itself distinguish affirmed from negated/hypothetical mentions. Observed failure modes:

- **Negation** — "did not smoke", "denies tobacco" flagged as smoking.
- **Discharge advice** — "importance of smoking cessation" is guidance, not patient status.
- **False positives** — `ex` inside "ex-lap" → Ex-smoker; "every day" → Daily.
- **Conflicts** — 61 admissions (~12%) carried internally contradictory mentions.
- **Coverage** — 452 patients (45%) have no smoking evidence in a discharge summary and are honestly recorded as `unknown` rather than guessed.

Where both sources exist, notes and ICD agreed **73%** of the time (74/101).

---

## MedCATtrainer

MedCATtrainer was installed and run locally via Docker; a project was created and documents annotated (marking correct concepts and rejecting false positives) to exercise the review workflow.

**Note on the provided pack:** `smoking_findings_only` is in **MedCAT v2** format (the CDB is a folder containing a ~255 MB `raw_dict.dat`, plus `model_card.json`). The current public MedCATtrainer image runs **MedCAT v1.16**, whose loader (`CAT.load_cdb`) expects a flat `cdb.dat` file — so the pack's CDB extracts empty (~4 KB stub), no concepts import, and nothing highlights.

Attempts to bridge this:

- **Convert v2 → v1**: not possible — MedCAT v2's serialiser only writes the folder format, and no v2→v1 converter exists (`cdb.py`'s version check only warns to "download the compatible model").
- **Rebuild the Trainer on MedCAT v2**: the image builds, but the Django app targets the v1 API and won't start.

CogStack archived the v1 MedCAT/MedCATtrainer repos in mid-2025 and consolidated into [`cogstack-nlp`](https://github.com/CogStack/cogstack-nlp), where a v2-era trainer is in active development — but the v2 model/trainer tooling is still mid-transition. This was flagged to the team. For the analysis, the pack was run via the **MedCAT v2 library directly** (see `Analysis.py`), and the Trainer workflow demonstrated with a compatible model. See `outputs/medcat_mismatch_flowchart.png`.

---

## Next steps to make this production-ready

- **Add a trained MetaCAT status model** — the single biggest win; learns affirmed vs negated vs hypothetical, replacing the hand-written rules.
- **Tighten concept filtering** to eliminate substring false positives.
- **Widen beyond discharge summaries** (physician/nursing notes) to recover `unknown`s.
- **Capture temporality / pack-years** — "quit 20 years ago" vs "quit last week" both read as `former` today.
- **Validate against clinician annotation** for real precision/recall/F1 before any clinical use.

---

## Repository hygiene

The repo deliberately excludes, via `.gitignore`:

- MedCAT model packs (`models/`, `*.zip`) — including `smoking_findings_only`
- Raw MIMIC extracts and note text (`data/`, and the raw-text working files `outputs/mct_dataset.csv`, `outputs/_mentions_cache.csv`)
- Credentials (`.env`) and the virtual environment (`venv311/`)

## Repository contents
