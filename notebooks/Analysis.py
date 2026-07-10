# %%
import sys
print(sys.executable)


# %%
import os
print("Python check:")
import sys
print(sys.executable)   # confirm this now shows venv311

print("models:", os.listdir('../models'))
print("data:", os.listdir('../data'))

# %%
from medcat.cat import CAT

cat = CAT.load_model_pack('../models/smoking_findings_only_a7554ffe6df0f97f.zip')

text = "Patient is a former smoker, quit 10 years ago. Denies current tobacco use."
entities = cat.get_entities(text)
entities

# %%
import pandas as pd

diag = pd.read_csv('../data/DIAGNOSES_ICD_1000HADM_IDs.csv')
notes = pd.read_csv('../data/NOTEEVENTS.csv')

print("DIAGNOSES shape:", diag.shape)
print(diag.columns.tolist())
print(diag.head(3))
print("\nNOTES shape:", notes.shape)
print(notes.columns.tolist())
print("\nNote categories:", notes['CATEGORY'].value_counts().to_dict() if 'CATEGORY' in notes.columns else "no CATEGORY col")
print("\nSample note (first 400 chars):")
print(notes['TEXT'].iloc[0][:400] if 'TEXT' in notes.columns else "check column names above")

# %%
#34,619 notes across 1,000 admissions: averaging ~35 notes each. Running MedCAT on all of them will be slow and mostly wasteful, because smoking status lives in specific note types.
#The note types matter enormously. That sample was a radiology report — no smoking info, just a chest X-ray. Smoking status almost always lives in the social history section of Discharge summaries (1,069 of them) and Physician notes. Radiology, ECG, Echo, nursing obs — these essentially never carry smoking status.
#Both files share SUBJECT_ID and HADM_ID: clean join keys. Good.
#ICD9_CODE column confirmed: we can scan for structured smoking codes.

# %%
# Filter to high-yield note types
disch = notes[notes['CATEGORY'] == 'Discharge summary'].copy()
print("Discharge summaries:", disch.shape[0])
print("Unique admissions with a discharge summary:", disch['HADM_ID'].nunique())
print("Unique patients:", disch['SUBJECT_ID'].nunique())

# Sanity check: does smoking language actually appear?
import re
sample = disch['TEXT'].iloc[0]
# find a social history section in the first discharge summary
idx = sample.lower().find('social history')
print("\n--- Social History excerpt from first discharge summary ---")
print(sample[idx:idx+300] if idx > -1 else "no 'social history' header in first note - will still catch mentions elsewhere")

# %%
import time

# Concept CUIs the model uses for smoking (from our smoke test + common MedCAT smoking concepts)
# We'll capture ALL entities and inspect which concepts actually appear
disch_texts = disch[['SUBJECT_ID', 'HADM_ID', 'TEXT']].copy()

results = [] 
start = time.time()

for i, row in enumerate(disch_texts.itertuples(index=False)):
    ents = cat.get_entities(row.TEXT)['entities']
    for e in ents.values():
        # capture a text window around the mention for our own negation handling
        s, en = e['start'], e['end']
        left = row.TEXT[max(0, s-40):s]
        right = row.TEXT[en:en+20]
        results.append({
            'SUBJECT_ID': row.SUBJECT_ID,
            'HADM_ID': row.HADM_ID,
            'cui': e['cui'],
            'pretty_name': e['pretty_name'],
            'source_value': e['source_value'],
            'window': (left + '[[' + row.TEXT[s:en] + ']]' + right).replace('\n', ' ')
        })
    if (i+1) % 200 == 0:
        print(f"{i+1}/{len(disch_texts)} notes processed... ({time.time()-start:.0f}s)")

print(f"\nDone. {len(results)} smoking-related mentions found across {len(disch_texts)} notes in {time.time()-start:.0f}s")

import pandas as pd
mentions = pd.DataFrame(results)
print("\nConcepts found (frequency):")
print(mentions['pretty_name'].value_counts())

# %%
mentions.to_csv('../outputs/_mentions_cache.csv', index=False)
print("cached:", mentions.shape)

# %%
# 1. What concepts did it find?
print(mentions['pretty_name'].value_counts())
print("\n" + "="*60 + "\n")

# 2. The actual text that triggered each mention (the crucial bit)
for w in mentions['window'].head(30).tolist():
    print(w)

# %%
import re

# Map raw concepts to smoking status categories
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
    # left side of the window (before the [[ )
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
print("\nDropped as junk/advice/ex:", (mentions['status']=='DROP').sum())

# %%
# Collapse mentions to one status per admission (HADM_ID)
priority = {'current': 3, 'former': 2, 'never': 1}

def resolve_admission(group):
    statuses = group['status'].tolist()
    # pick highest-priority status present
    best = max(statuses, key=lambda s: priority[s])
    # flag if there was genuine disagreement
    distinct = set(statuses)
    conflict = len(distinct) > 1
    return pd.Series({
        'note_status': best,
        'note_mention_count': len(statuses),
        'note_conflict': conflict,
        'note_statuses_seen': ','.join(sorted(distinct))
    })

adm = clean.groupby(['SUBJECT_ID', 'HADM_ID']).apply(resolve_admission).reset_index()

print("Admissions with a note-derived status:", adm.shape[0])
print("\nNote-derived status distribution (admission-level):")
print(adm['note_status'].value_counts())
print("\nAdmissions with conflicting mentions:", adm['note_conflict'].sum())
print("\nExamples of conflicts:")
print(adm[adm['note_conflict']][['HADM_ID','note_status','note_statuses_seen','note_mention_count']].head(10))

# %%
# ICD-9 structured smoking codes
# Codes in MIMIC are stored without the decimal point (e.g. '3051', 'V1582')
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

# collapse to admission level (same priority: current > former)
icd_priority = {'current': 2, 'former': 1}
icd_adm = (smoking_codes.groupby(['SUBJECT_ID','HADM_ID'])['icd_status']
           .apply(lambda s: max(s, key=lambda x: icd_priority[x]))
           .reset_index()
           .rename(columns={'icd_status':'icd_status'}))

print("\nAdmissions with a smoking ICD code:", icd_adm.shape[0])
print(icd_adm['icd_status'].value_counts())

# %%
# Merge note-derived and ICD-derived status at admission level
merged = adm.merge(icd_adm, on=['SUBJECT_ID','HADM_ID'], how='outer')

# Agreement analysis where BOTH sources exist
both = merged.dropna(subset=['note_status','icd_status'])
agree = (both['note_status'] == both['icd_status']).sum()
print(f"Admissions with BOTH note + ICD status: {len(both)}")
print(f"  Agree: {agree}  ({100*agree/len(both):.0f}%)")
print(f"  Disagree: {len(both)-agree}")
print("\nDisagreement breakdown:")
print(both[both['note_status']!=both['icd_status']]
      .groupby(['note_status','icd_status']).size())

# Final admission status: prefer note (richer), fall back to ICD, record source
def final_status(r):
    if pd.notna(r['note_status']):
        return pd.Series({'smoking_status': r['note_status'],
                          'source': 'note' if pd.isna(r['icd_status']) else 'note+icd'})
    return pd.Series({'smoking_status': r['icd_status'], 'source': 'icd_only'})

merged[['smoking_status','source']] = merged.apply(final_status, axis=1)
print("\nAdmissions with any smoking status:", merged.shape[0])
print(merged['smoking_status'].value_counts())
print("\nBy source:")
print(merged['source'].value_counts())

# %%
# Collapse to ONE row per patient (SUBJECT_ID)
# A patient may have multiple admissions with different statuses over time.
# Priority: current > former > never (most safety-relevant wins)
pri = {'current': 3, 'former': 2, 'never': 1}

def resolve_patient(g):
    statuses = g['smoking_status'].dropna().tolist()
    best = max(statuses, key=lambda s: pri[s])
    distinct = set(statuses)
    # combined source across their admissions
    srcs = ','.join(sorted(set(g['source'].dropna())))
    return pd.Series({
        'smoking_status': best,
        'source_doc_count': int(g.shape[0]),
        'confidence': 'high' if len(distinct)==1 else 'conflicting',
        'notes': f"statuses_seen={'/'.join(sorted(distinct))}" if len(distinct)>1 else srcs
    })

patient = merged.groupby('SUBJECT_ID').apply(resolve_patient, include_groups=False).reset_index()
patient = patient.rename(columns={'SUBJECT_ID':'patient_id'})

# Bring in ALL patients from the cohort — those with no smoking evidence = 'unknown'
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

# Write the deliverable
patient = patient[['patient_id','smoking_status','confidence','source_doc_count','notes']]
patient.to_csv('../outputs/patient_smoking_status.csv', index=False)
print("\n Written to outputs/patient_smoking_status.csv")
print(patient.head(10))

# %%
print("="*50)
print("SUMMARY FIGURES FOR SLIDES")
print("="*50)
print(f"Cohort: 995 patients, 1,000 admissions, 34,619 notes")
print(f"Processed: 1,069 discharge summaries (targeted high-yield note type)")
print(f"Raw smoking mentions found: 811")
print(f"After cleanup (junk/negation/advice removed): 711  ({811-711} dropped)")
print()
print(f"Patient-level status:")
print(f"  Current: 297 | Former: 116 | Never: 130 | Unknown: 452")
print()
print(f"SOURCE COMPARISON (the key finding):")
print(f"  Notes captured status for 518 admissions")
print(f"  ICD-9 codes captured only 129 admissions (~4x less)")
print(f"  ICD-9 cannot represent 'never' at all")
print(f"  Where both exist: 73% agreement, 27 disagreements")
print()
print(f"Data quality flags:")
print(f"  61 admissions had internally conflicting mentions")
print(f"  MedCAT false positives: 'ex'→Ex-smoker (ex-lap), 'every day'→Daily")
print(f"  No MetaCAT status model: negation handled by our own rules")

# %%
# Build a manual annotation review as we cannot train the MedCat (multiple hour job and local PC does not have docker)

# Manual annotation review — replicating the MedCATtrainer inspection workflow
# Pick one discharge summary and show every smoking entity MedCAT found, with verdict

sample_hadm = clean['HADM_ID'].iloc[0]  # a note we know has smoking mentions
doc = disch[disch['HADM_ID']==sample_hadm]['TEXT'].iloc[0]

ents = cat.get_entities(doc)['entities']
print(f"Document HADM_ID {sample_hadm} — {len(ents)} entities flagged by MedCAT\n")
print("="*70)
for e in ents.values():
    s, en = e['start'], e['end']
    context = doc[max(0,s-50):en+30].replace('\n',' ')
    print(f"CONCEPT : {e['pretty_name']}  (CUI {e['cui']})")
    print(f"MATCHED : '{e['source_value']}'")
    print(f"CONTEXT : ...{context}...")
    print(f"meta_anns: {e['meta_anns']}   <- empty = no negation/status model")
    print("-"*70)

# %%
# Find the admission with the MOST smoking mentions — guaranteed rich example
top_hadm = clean['HADM_ID'].value_counts().index[0]
print(f"Using HADM_ID {top_hadm} — has {clean['HADM_ID'].value_counts().iloc[0]} smoking mentions\n")

doc = disch[disch['HADM_ID']==top_hadm]['TEXT'].iloc[0]
ents = cat.get_entities(doc)['entities']

print(f"MedCAT flagged {len(ents)} total entities in this document\n")
print("="*70)
for e in ents.values():
    s, en = e['start'], e['end']
    context = doc[max(0,s-50):en+30].replace('\n',' ')
    print(f"CONCEPT : {e['pretty_name']}  (CUI {e['cui']})")
    print(f"MATCHED : '{e['source_value']}'")
    print(f"CONTEXT : ...{context}...")
    print(f"meta_anns: {e['meta_anns']}")
    print("-"*70)

# %%
# Export ~5 smoking-rich discharge summaries for MedCATtrainer
rich_hadms = clean['HADM_ID'].value_counts().head(5).index.tolist()
sample_notes = disch[disch['HADM_ID'].isin(rich_hadms)][['HADM_ID','TEXT']].copy()
sample_notes = sample_notes.rename(columns={'HADM_ID':'name','TEXT':'text'})
sample_notes.to_csv('../outputs/mct_dataset.csv', index=False)
print("Saved", len(sample_notes), "notes to outputs/mct_dataset.csv")
print(sample_notes['name'].tolist())

# %%
# Check cat is still loaded; reload if needed
try:
    cat
    print("cat is loaded")
except NameError:
    from medcat.cat import CAT
    cat = CAT.load_model_pack('../models/smoking_findings_only_a7554ffe6df0f97f.zip')
    print("reloaded cat")


# %%
import os
import json
import numpy as np

export_dir = r"C:\Users\mtmic\cogstack-fde-task\models\v1_export"
os.makedirs(export_dir, exist_ok=True)

# Custom JSON encoder to catch sets and numpy arrays
class MedCATJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# 1. Export CDB State
try:
    # We grab the core mapping dictionary
    cdb_data = getattr(cat.cdb, 'cui2info', cat.cdb.__dict__)
    
    with open(os.path.join(export_dir, "cdb.json"), "w", encoding="utf-8") as f:
        json.dump(cdb_data, f, cls=MedCATJsonEncoder)
    print("--- Successfully dumped CDB to clean JSON! ---")
except Exception as e:
    print("CDB JSON export failed:", type(e).__name__, str(e))

# 2. Export Vocab State
try:
    vocab_data = getattr(cat.vocab, 'vocab', cat.vocab.__dict__)
    
    with open(os.path.join(export_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, cls=MedCATJsonEncoder)
    print("--- Successfully dumped Vocab to clean JSON! ---")
except Exception as e:
    print("Vocab JSON export failed:", type(e).__name__, str(e))

print("\nFiles in export directory:", os.listdir(export_dir))

# %%
import os
import pickle
import sys

export_dir = r"C:\Users\mtmic\cogstack-fde-task\models\v1_export"
os.makedirs(export_dir, exist_ok=True)

print("Applying version-alignment patch...")

# Trick the pickle serializer into using the old MedCAT v1 module path paths
cat.cdb.__class__.__module__ = 'medcat.cdb'
if hasattr(cat.vocab, '__class__'):
    cat.vocab.__class__.__module__ = 'medcat.vocab'

# Force save using standard pickle protocol 4 (highly compatible across Python 3.11)
try:
    with open(os.path.join(export_dir, "cdb.dat"), "wb") as f:
        pickle.dump(cat.cdb, f, protocol=4)
    print("--- Successfully compiled compatible cdb.dat! ---")
except Exception as e:
    print("CDB compilation failed:", e)

try:
    with open(os.path.join(export_dir, "vocab.dat"), "wb") as f:
        pickle.dump(cat.vocab, f, protocol=4)
    print("--- Successfully compiled compatible vocab.dat! ---")
except Exception as e:
    print("Vocab compilation failed:", e)

print("\nReady in directory:", os.listdir(export_dir))

# %%
patient = patient[['patient_id','smoking_status','confidence','source_doc_count','notes']]
patient.to_csv('../outputs/patient_smoking_status.csv', index=False)

# %%
import matplotlib.pyplot as plt
import os
os.makedirs('../outputs', exist_ok=True)

# ---- CHART 1: Patient status distribution (Slide 1 main graph) ----
fig, ax = plt.subplots(figsize=(7, 4))
status = ['Current', 'Former', 'Never', 'Unknown']
vals = [297, 116, 130, 452]
colors = ['#B85042', '#E09F3E', '#028090', '#C9C9C9']
bars = ax.bar(status, vals, color=colors, width=0.68)
ax.set_ylabel('Patients', fontsize=11)
ax.set_title('Smoking status across 995 patients', fontsize=13, fontweight='bold')
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+6, str(v), ha='center', fontweight='bold', fontsize=12)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('../outputs/chart_status_distribution.png', dpi=160, bbox_inches='tight')
plt.close()

# ---- CHART 2: Notes vs ICD coverage (the 4x headline finding) ----
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(['Clinical notes\n(free text)', 'ICD-9 codes\n(structured)'], [518, 129],
              color=['#028090', '#B85042'], width=0.6)
ax.set_ylabel('Admissions with smoking status', fontsize=11)
ax.set_title('Free text captures ~4x more smoking status than coded data',
             fontsize=12, fontweight='bold')
for b, c in zip(bars, [518, 129]):
    ax.text(b.get_x()+b.get_width()/2, c+8, str(c), ha='center', fontweight='bold', fontsize=13)
ax.spines[['top','right']].set_visible(False)
ax.set_ylim(0, 580)
plt.tight_layout()
plt.savefig('../outputs/chart_notes_vs_icd.png', dpi=160, bbox_inches='tight')
plt.close()

# ---- CHART 3: Source breakdown (how each status was derived) ----
fig, ax = plt.subplots(figsize=(7, 4))
srcs = ['Note only', 'Note + ICD', 'ICD only']
svals = [417, 101, 28]
bars = ax.bar(srcs, svals, color=['#028090', '#4C9F70', '#E09F3E'], width=0.6)
ax.set_ylabel('Admissions', fontsize=11)
ax.set_title('Evidence source for smoking status', fontsize=13, fontweight='bold')
for b, v in zip(bars, svals):
    ax.text(b.get_x()+b.get_width()/2, v+4, str(v), ha='center', fontweight='bold', fontsize=12)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('../outputs/chart_source_breakdown.png', dpi=160, bbox_inches='tight')
plt.close()

# confirm
import os
print("Saved charts:")
for f in os.listdir('../outputs'):
    if f.endswith('.png'):
        print("  ", f)

# %%
import os
import json
import numpy as np

export_dir = r"C:\Users\mtmic\cogstack-fde-task\models\v1_export"
os.makedirs(export_dir, exist_ok=True)

# Custom JSON encoder to catch sets and numpy arrays
class MedCATJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# 1. Export CDB State
try:
    # We grab the core mapping dictionary
    cdb_data = getattr(cat.cdb, 'cui2info', cat.cdb.__dict__)
    
    with open(os.path.join(export_dir, "cdb.json"), "w", encoding="utf-8") as f:
        json.dump(cdb_data, f, cls=MedCATJsonEncoder)
    print("--- Successfully dumped CDB to clean JSON! ---")
except Exception as e:
    print("CDB JSON export failed:", type(e).__name__, str(e))

# 2. Export Vocab State
try:
    vocab_data = getattr(cat.vocab, 'vocab', cat.vocab.__dict__)
    
    with open(os.path.join(export_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, cls=MedCATJsonEncoder)
    print("--- Successfully dumped Vocab to clean JSON! ---")
except Exception as e:
    print("Vocab JSON export failed:", type(e).__name__, str(e))

print("\nFiles in export directory:", os.listdir(export_dir))

# %%
import os
import json
import numpy as np

export_dir = r"C:\Users\mtmic\cogstack-fde-task\models\v1_export"
os.makedirs(export_dir, exist_ok=True)

# Custom JSON encoder to catch sets and numpy arrays
class MedCATJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# 1. Export CDB State
try:
    # We grab the core mapping dictionary
    cdb_data = getattr(cat.cdb, 'cui2info', cat.cdb.__dict__)
    
    with open(os.path.join(export_dir, "cdb.json"), "w", encoding="utf-8") as f:
        json.dump(cdb_data, f, cls=MedCATJsonEncoder)
    print("--- Successfully dumped CDB to clean JSON! ---")
except Exception as e:
    print("CDB JSON export failed:", type(e).__name__, str(e))

# 2. Export Vocab State
try:
    vocab_data = getattr(cat.vocab, 'vocab', cat.vocab.__dict__)
    
    with open(os.path.join(export_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, cls=MedCATJsonEncoder)
    print("--- Successfully dumped Vocab to clean JSON! ---")
except Exception as e:
    print("Vocab JSON export failed:", type(e).__name__, str(e))

print("\nFiles in export directory:", os.listdir(export_dir))


