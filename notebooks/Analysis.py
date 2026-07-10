"""
Smoking status extraction from MIMIC clinical notes using MedCAT.

Pipeline:
    1. Load clinical notes + ICD-9 diagnoses for the 1,000-admission cohort.
    2. Filter to discharge summaries (the high-yield note type for social/smoking history).
    3. Run MedCAT with the `smoking_findings_only` model pack to find tobacco/smoking concepts.
    4. Clean & classify mentions into current / former / never
       (filtering negation, discharge advice, and false positives via context rules,
        because the provided pack has no MetaCAT status model).
    5. Join structured ICD-9 smoking codes as an independent second source.
    6. Resolve to one status per patient and write the deliverable CSV.
    7. Produce summary figures / charts for the presentation.

Output:
    outputs/patient_smoking_status.csv
    outputs/chart_*.png

Requires (not committed to the repo — see README):
    data/NOTEEVENTS.csv
    data/DIAGNOSES_ICD_1000HADM_IDs.csv
    models/smoking_findings_only_<hash>.zip
"""

# %%
# --- Environment check ---
import os
import sys
print("Python executable:", sys.executable)   # confirm venv311
print("models:", os.listdir('../models'))
print("data:", os.listdir('../data'))

# %%
# --- Load the MedCAT model pack and smoke-test it ---
from medcat.cat import CAT

cat = CAT.load_model_pack('../models/smoking_findings_only_a7554ffe6df0f97f.zip')

text = "Patient is a former smoker, quit 10 years ago. Denies current tobacco use."
entities = cat.get_entities(text)
print(entities)

# %%
# --- Load the data ---
import pandas as pd

diag = pd.read_csv('../data/DIAGNOSES_ICD_1000HADM_IDs.csv')
notes = pd.read_csv('../data/NOTEEVENTS.csv')

print("DIAGNOSES shape:", diag.shape)
print(diag.columns.tolist())
print("\nNOTES shape:", notes.shape)
print(notes.columns.tolist())
print("\nNote categories:",
      notes['CATEGORY'].value_counts().to_dict() if 'CATEGORY' in notes.columns else "no CATEGORY col")

# %%
# Design note:
# 34,619 notes across 1,000 admissions (~35 notes each). Running MedCAT on all of them
# is slow and mostly wasteful: smoking status lives in specific note types.
# Smoking/social history almost always appears in Discharge summaries (and physician notes).
# Radiology / ECG / Echo / nursing obs essentially never carry smoking status.
# Both files share SUBJECT_ID and HADM_ID as clean join keys, and ICD9_CODE lets us
# scan for structured smoking codes as a second, independent source.

# %%
# --- Filter to high-yield note type: discharge summaries ---
disch = notes[notes['CATEGORY'] == 'Discharge summary'].copy()
print("Discharge summaries:", disch.shape[0])
print("Unique admissions with a discharge summary:", disch['HADM_ID'].nunique())
print("Unique patients:", disch['SUBJECT_ID'].nunique())

# %%
# --- Run MedCAT over every discharge summary ---
# Capture all entities plus a text window around each mention, so we can apply
# our own negation / advice / false-positive rules (the pack has no MetaCAT status model).
import time

disch_texts = disch[['SUBJECT_ID', 'HADM_ID', 'TEXT']].copy()
results = []
start = time.time()

for i, row in enumerate(disch_texts.itertuples(index=False)):
    ents = cat.get_entities(row.TEXT)['entities']
    for e in ents.values():
        s, en = e['start'], e['end']
        left = row.TEXT[max(0, s - 40):s]
        right = row.TEXT[en:en + 20]
        results.append({
            'SUBJECT_ID': row.SUBJECT_ID,
            'HADM_ID': row.HADM_ID,
            'cui': e['cui'],
            'pretty_name': e['pretty_name'],
            'source_value': e['source_value'],
            'window': (left + '[[' + row.TEXT[s:en] + ']]' + right).replace('\n', ' ')
        })
    if (i + 1) % 200 == 0:
        print(f"{i+1}/{len(disch_texts)} notes processed... ({time.time()-start:.0f}s)")

print(f"\nDone. {len(results)} smoking-related mentions found "
      f"across {len(disch_texts)} notes in {time.time()-start:.0f}s")

mentions = pd.DataFrame(results)
print("\nConcepts found (frequency):")
print(mentions['pretty_name'].value_counts())

# %%
# Cache mentions locally so re-runs don't need the ~12-min MedCAT pass again.
# NOTE: this file contains raw note snippets (the `window` column) and is gitignored.
mentions.to_csv('../outputs/_mentions_cache.csv', index=False)
print("cached:", mentions.shape)

# %%
# --- Map raw concepts to smoking-status categories + apply context rules ---
import re

STATUS_MAP = {
    'Smoker': 'current', 'Cigarette smoker': 'current', 'Daily': None,  # 'Daily' is junk
    'Tobacco user': 'current', 'Finding of tobacco smoking behavior': 'current',
    'Cigar smoker': 'current', 'Occasional tobacco smoker': 'current',
    'Smoking started': 'current', 'Smoke': 'current', 'Smoking': 'current',
    'Ex-smoker': 'former', 'Ex-tobacco user': 'former', 'Stopped smoking': 'former',
    'Non-smoker': 'never', 'Never smoked tobacco': 'never',
}

# Junk concepts to drop entirely (false positives)
JUNK = {'Daily', 'Excision of cervical lymph nodes group', 'every day'}

# Negation / non-patient cues in the left context
NEG_PATTERNS = re.compile(r'\b(no|non|not|denies|denied|negative for|without)\b', re.I)
ADVICE_PATTERNS = re.compile(r'\b(help you|to quit|should|nicotine gum|avoid|counsel)\b', re.I)


def classify(row):
    name = row['pretty_name']
    if name in JUNK:
        return 'DROP'
    left = row['window'].split('[[')[0]
    inside = row['window'].split('[[')[1].split(']]')[0] if '[[' in row['window'] else ''
    # 'ex' as substring of ex-lap etc — drop short false matches
    if inside.strip().lower() == 'ex':
        return 'DROP'
    # negation flips to 'never'
    if NEG_PATTERNS.search(left[-25:]):
        return 'never'
    # discharge advice about quitting is not patient status
    if ADVICE_PATTERNS.search(left[-40:]):
        return 'DROP'
    return STATUS_MAP.get(name, None)


mentions['status'] = mentions.apply(classify, axis=1)
clean = mentions[~mentions['status'].isin(['DROP', None])].copy()

print("Before cleanup:", len(mentions), "mentions")
print("After cleanup:", len(clean), "mentions")
print("\nStatus distribution (mention-level):")
print(clean['status'].value_counts())
print("\nDropped as junk/advice/ex:", (mentions['status'] == 'DROP').sum())

# %%
# --- Collapse mentions to one status per admission (note-derived) ---
priority = {'current': 3, 'former': 2, 'never': 1}


def resolve_admission(group):
    statuses = group['status'].tolist()
    best = max(statuses, key=lambda s: priority[s])
    distinct = set(statuses)
    return pd.Series({
        'note_status': best,
        'note_mention_count': len(statuses),
        'note_conflict': len(distinct) > 1,
        'note_statuses_seen': ','.join(sorted(distinct))
    })


adm = clean.groupby(['SUBJECT_ID', 'HADM_ID']).apply(resolve_admission).reset_index()

print("Admissions with a note-derived status:", adm.shape[0])
print("\nNote-derived status distribution (admission-level):")
print(adm['note_status'].value_counts())
print("\nAdmissions with conflicting mentions:", adm['note_conflict'].sum())
print("\nExamples of conflicts:")
print(adm[adm['note_conflict']][['HADM_ID', 'note_status', 'note_statuses_seen', 'note_mention_count']].head(10))

# %%
# --- Second source: structured ICD-9 smoking codes ---
# Codes in MIMIC are stored without the decimal point (e.g. '3051', 'V1582').
ICD9_SMOKING = {
    '3051': 'current',   # tobacco use disorder
    'V1582': 'former',   # personal history of tobacco use
    '98984': 'current',  # toxic effect of tobacco
    '64900': 'current', '64901': 'current', '64902': 'current',
    '64903': 'current', '64904': 'current',  # tobacco use in pregnancy
}

diag['ICD9_CODE'] = diag['ICD9_CODE'].astype(str).str.strip()
smoking_codes = diag[diag['ICD9_CODE'].isin(ICD9_SMOKING.keys())].copy()
smoking_codes['icd_status'] = smoking_codes['ICD9_CODE'].map(ICD9_SMOKING)

print("Rows with a smoking ICD-9 code:", smoking_codes.shape[0])
print("\nWhich codes appear:")
print(smoking_codes['ICD9_CODE'].value_counts())

# Collapse to admission level (current > former)
icd_priority = {'current': 2, 'former': 1}
icd_adm = (smoking_codes.groupby(['SUBJECT_ID', 'HADM_ID'])['icd_status']
           .apply(lambda s: max(s, key=lambda x: icd_priority[x]))
           .reset_index())

print("\nAdmissions with a smoking ICD code:", icd_adm.shape[0])
print(icd_adm['icd_status'].value_counts())

# %%
# --- Merge note-derived and ICD-derived status; agreement analysis ---
merged = adm.merge(icd_adm, on=['SUBJECT_ID', 'HADM_ID'], how='outer')

both = merged.dropna(subset=['note_status', 'icd_status'])
agree = (both['note_status'] == both['icd_status']).sum()
print(f"Admissions with BOTH note + ICD status: {len(both)}")
print(f"  Agree: {agree}  ({100*agree/len(both):.0f}%)")
print(f"  Disagree: {len(both)-agree}")
print("\nDisagreement breakdown:")
print(both[both['note_status'] != both['icd_status']].groupby(['note_status', 'icd_status']).size())


# Final admission status: prefer note (richer), fall back to ICD, record source
def final_status(r):
    if pd.notna(r['note_status']):
        return pd.Series({'smoking_status': r['note_status'],
                          'source': 'note' if pd.isna(r['icd_status']) else 'note+icd'})
    return pd.Series({'smoking_status': r['icd_status'], 'source': 'icd_only'})


merged[['smoking_status', 'source']] = merged.apply(final_status, axis=1)
print("\nAdmissions with any smoking status:", merged.shape[0])
print(merged['smoking_status'].value_counts())
print("\nBy source:")
print(merged['source'].value_counts())

# %%
# --- Resolve to ONE row per patient (SUBJECT_ID) ---
# A patient may have multiple admissions with different statuses over time.
# Priority: current > former > never (most safety-relevant wins).
pri = {'current': 3, 'former': 2, 'never': 1}


def resolve_patient(g):
    statuses = g['smoking_status'].dropna().tolist()
    best = max(statuses, key=lambda s: pri[s])
    distinct = set(statuses)
    srcs = ','.join(sorted(set(g['source'].dropna())))
    return pd.Series({
        'smoking_status': best,
        'source_doc_count': int(g.shape[0]),
        'confidence': 'high' if len(distinct) == 1 else 'conflicting',
        'notes': f"statuses_seen={'/'.join(sorted(distinct))}" if len(distinct) > 1 else srcs
    })


patient = merged.groupby('SUBJECT_ID').apply(resolve_patient, include_groups=False).reset_index()
patient = patient.rename(columns={'SUBJECT_ID': 'patient_id'})

# Bring in ALL cohort patients — those with no smoking evidence = 'unknown'
all_patients = pd.Index(notes['SUBJECT_ID'].unique(), name='patient_id')
patient = patient.set_index('patient_id').reindex(all_patients).reset_index()
patient['smoking_status'] = patient['smoking_status'].fillna('unknown')
patient['source_doc_count'] = patient['source_doc_count'].fillna(0).astype(int)
patient['confidence'] = patient['confidence'].fillna('none')
patient['notes'] = patient['notes'].fillna('no smoking evidence found')

print("Total patients in cohort:", patient.shape[0])
print("\nFinal patient-level smoking status:")
print(patient['smoking_status'].value_counts())
print("\nConfidence breakdown:")
print(patient['confidence'].value_counts())

# --- Write the deliverable ---
patient = patient[['patient_id', 'smoking_status', 'confidence', 'source_doc_count', 'notes']]
patient.to_csv('../outputs/patient_smoking_status.csv', index=False)
print("\nWritten to outputs/patient_smoking_status.csv")
print(patient.head(10))

# %%
# --- Summary figures (for the presentation) ---
print("=" * 50)
print("SUMMARY FIGURES")
print("=" * 50)
print("Cohort: 995 patients, 1,000 admissions, 34,619 notes")
print("Processed: 1,069 discharge summaries (targeted high-yield note type)")
print("Raw smoking mentions found: 811")
print("After cleanup (junk/negation/advice removed): 711  (100 dropped)")
print()
print("Patient-level status:")
print("  Current: 297 | Former: 116 | Never: 130 | Unknown: 452")
print()
print("SOURCE COMPARISON (key finding):")
print("  Notes captured status for 518 admissions")
print("  ICD-9 codes captured only 129 admissions (~4x less)")
print("  ICD-9 cannot represent 'never' at all")
print("  Where both exist: 73% agreement, 27 disagreements")
print()
print("Data quality flags:")
print("  61 admissions had internally conflicting mentions")
print("  MedCAT false positives: 'ex'->Ex-smoker (ex-lap), 'every day'->Daily")
print("  No MetaCAT status model: negation handled by our own rules")

# %%
# --- Annotation review (MedCATtrainer-equivalent inspection) ---
# Inspect MedCAT's annotations on a single rich document: the concepts it finds,
# the triggering text, and the (empty) meta_anns confirming no negation/status model.
top_hadm = clean['HADM_ID'].value_counts().index[0]
print(f"Using HADM_ID {top_hadm} — {clean['HADM_ID'].value_counts().iloc[0]} smoking mentions\n")

doc = disch[disch['HADM_ID'] == top_hadm]['TEXT'].iloc[0]
ents = cat.get_entities(doc)['entities']

print(f"MedCAT flagged {len(ents)} entities in this document\n" + "=" * 70)
for e in ents.values():
    s, en = e['start'], e['end']
    context = doc[max(0, s - 50):en + 30].replace('\n', ' ')
    print(f"CONCEPT : {e['pretty_name']}  (CUI {e['cui']})")
    print(f"MATCHED : '{e['source_value']}'")
    print(f"CONTEXT : ...{context}...")
    print(f"meta_anns: {e['meta_anns']}   <- empty = no negation/status model")
    print("-" * 70)

# %%
# --- Export a small dataset for MedCATtrainer (5 smoking-rich discharge summaries) ---
# NOTE: this file contains raw note text and is gitignored.
rich_hadms = clean['HADM_ID'].value_counts().head(5).index.tolist()
sample_notes = disch[disch['HADM_ID'].isin(rich_hadms)][['HADM_ID', 'TEXT']].copy()
sample_notes = sample_notes.rename(columns={'HADM_ID': 'name', 'TEXT': 'text'})
sample_notes.to_csv('../outputs/mct_dataset.csv', index=False)
print("Saved", len(sample_notes), "notes to outputs/mct_dataset.csv")

# %%
# --- Charts for the presentation ---
import matplotlib.pyplot as plt
os.makedirs('../outputs', exist_ok=True)

# Chart 1: patient status distribution
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(['Current', 'Former', 'Never', 'Unknown'], [297, 116, 130, 452],
              color=['#B85042', '#E09F3E', '#028090', '#C9C9C9'], width=0.68)
ax.set_ylabel('Patients', fontsize=11)
ax.set_title('Smoking status across 995 patients', fontsize=13, fontweight='bold')
for b, v in zip(bars, [297, 116, 130, 452]):
    ax.text(b.get_x() + b.get_width() / 2, v + 6, str(v), ha='center', fontweight='bold', fontsize=12)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('../outputs/chart_status_distribution.png', dpi=160, bbox_inches='tight')
plt.close()

# Chart 2: notes vs ICD coverage (the 4x finding)
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(['Clinical notes\n(free text)', 'ICD-9 codes\n(structured)'], [518, 129],
              color=['#028090', '#B85042'], width=0.6)
ax.set_ylabel('Admissions with smoking status', fontsize=11)
ax.set_title('Free text captures ~4x more smoking status than coded data', fontsize=12, fontweight='bold')
for b, c in zip(bars, [518, 129]):
    ax.text(b.get_x() + b.get_width() / 2, c + 8, str(c), ha='center', fontweight='bold', fontsize=13)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylim(0, 580)
plt.tight_layout()
plt.savefig('../outputs/chart_notes_vs_icd.png', dpi=160, bbox_inches='tight')
plt.close()

# Chart 3: evidence source breakdown
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(['Note only', 'Note + ICD', 'ICD only'], [417, 101, 28],
              color=['#028090', '#4C9F70', '#E09F3E'], width=0.6)
ax.set_ylabel('Admissions', fontsize=11)
ax.set_title('Evidence source for smoking status', fontsize=13, fontweight='bold')
for b, v in zip(bars, [417, 101, 28]):
    ax.text(b.get_x() + b.get_width() / 2, v + 4, str(v), ha='center', fontweight='bold', fontsize=12)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('../outputs/chart_source_breakdown.png', dpi=160, bbox_inches='tight')
plt.close()

print("Saved charts:")
for f in os.listdir('../outputs'):
    if f.endswith('.png'):
        print("  ", f)