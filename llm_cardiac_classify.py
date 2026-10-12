"""
LLM-based cardiac pathology classification from segmentation-derived clinical metrics.

Six conditions (2 metric sources x 3 prompting strategies):
  A: GT metrics   + Direct prompt
  B: GT metrics   + Naive CoT
  C: Pred metrics + Direct prompt
  D: Pred metrics + Naive CoT
  E: GT metrics   + Calibrated CoT  (proposed)
  F: Pred metrics + Calibrated CoT

Usage:
    export ANTHROPIC_API_KEY="sk-..."
    python llm_cardiac_classify.py
"""

import json
import os
import re
import time
from pathlib import Path

import anthropic

CLASSES = ["NOR", "DCM", "HCM", "MINF", "RV"]
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

DIRECT_PROMPT = """\
You are a cardiologist interpreting cardiac MRI findings.
A patient has the following MRI-derived cardiac metrics:
- Ejection Fraction (EF): {EF:.1f}%
- End-Diastolic Volume (EDV): {EDV:.1f} mL
- End-Systolic Volume (ESV): {ESV:.1f} mL

Based on these metrics, classify the cardiac condition into exactly one of:
NOR (Normal), DCM (Dilated Cardiomyopathy), HCM (Hypertrophic Cardiomyopathy),
MINF (Myocardial Infarction), RV (Right Ventricular Abnormality)

Reply with exactly one label: NOR, DCM, HCM, MINF, or RV\
"""

COT_PROMPT = """\
You are a cardiologist interpreting cardiac MRI findings.
A patient has the following MRI-derived cardiac metrics:
- Ejection Fraction (EF): {EF:.1f}%
- End-Diastolic Volume (EDV): {EDV:.1f} mL
- End-Systolic Volume (ESV): {ESV:.1f} mL

Reason step by step:
Step 1 - EF assessment: Normal range is 55-70%. Low EF (<40%) indicates systolic dysfunction. High EF (>70%) can indicate hypertrophic or hyperdynamic states.
Step 2 - Volume assessment: Normal EDV ~140-180 mL. Enlarged EDV (>200 mL) suggests dilated chambers. Small/normal EDV with high EF suggests hypertrophic physiology.
Step 3 - Pathology matching:
  DCM: low EF (<40%) + enlarged EDV (dilated, hypokinetic)
  HCM: normal or high EF (>=60%) + normal/small EDV (stiff, hypertrophied walls)
  MINF: reduced EF (variable, often 30-55%) + may reflect regional wall motion abnormality
  NOR: EF 55-70%, EDV 140-180 mL, normal physiology
  RV: primary right ventricular abnormality; LV metrics may appear near-normal or mildly abnormal

Reply format (use exactly this structure):
STEP1: <your EF reasoning>
STEP2: <your volume reasoning>
STEP3: <your diagnosis reasoning>
LABEL: <NOR|DCM|HCM|MINF|RV>\
"""

CALIBRATED_COT_PROMPT = """\
You are a cardiologist interpreting cardiac MRI findings.
A patient has the following MRI-derived cardiac metrics:
- Ejection Fraction (EF): {EF:.1f}%
- End-Diastolic Volume (EDV): {EDV:.1f} mL
- End-Systolic Volume (ESV): {ESV:.1f} mL

Here is an example of correct step-by-step reasoning:
[Example] EF=28.0%, EDV=310.0 mL, ESV=223.0 mL
STEP1: EF=28% is severely reduced (normal 55-70%), well below the <45% DCM threshold. Indicates significant systolic dysfunction.
STEP2: EDV=310 mL is markedly enlarged (normal 90-180 mL), far exceeding the >180 mL DCM threshold. The ventricle is dilated.
STEP3: Severely reduced EF (<45%) AND markedly dilated EDV (>180 mL) → classic Dilated Cardiomyopathy pattern.
LABEL: DCM

Now apply the same reasoning to the actual patient using these calibrated diagnostic thresholds:

  NOR  : EF 55–70%,  EDV 90–180 mL         — normal systolic function and chamber volume
  DCM  : EF <45%,    EDV >180 mL           — dilated, hypokinetic ventricle
  HCM  : EF ≥55%,   EDV <130 mL           — small/non-dilated cavity with preserved or hyperdynamic function (concentric hypertrophy)
  MINF : EF 25–55%, EDV 90–200 mL         — reduced function, near-normal or mildly enlarged chamber
  RV   : LV EF and EDV appear near-normal  — suspect RV when LV metrics do not match any above pattern,
         especially EDV <90 mL with EF 50–65% (right-heart dominant, small LV)

Important: If EF<0%, treat as severe systolic dysfunction (EDV/ESV tracking error → classify by EDV alone).

Reply format (use exactly this structure):
STEP1: <EF value, compare to normal 55-70%, state if reduced/normal/elevated and severity>
STEP2: <EDV value, compare to normal 90-180 mL, state if small/normal/enlarged>
STEP3: <match to ONE diagnostic criterion above, state which fits and why others are excluded>
LABEL: <NOR|DCM|HCM|MINF|RV>\
"""


def extract_label(text: str) -> str:
    """Extract the predicted class from LLM output."""
    # For CoT: look for LABEL: NOR/DCM/... pattern first
    m = re.search(r"LABEL:\s*([A-Z]+)", text.upper())
    if m and m.group(1) in CLASSES:
        return m.group(1)
    # Fallback: find any class word in text (last occurrence wins for CoT)
    found = [c for c in CLASSES if re.search(rf"\b{c}\b", text.upper())]
    return found[-1] if found else "NOR"


def ef_threshold_baseline(ef: float) -> str:
    """Simple EF-based rule — cannot distinguish MINF or RV."""
    if ef < 40:
        return "DCM"
    elif ef > 70:
        return "HCM"
    else:
        return "NOR"


def run_experiment(patients: list, prompt_template: str, condition_name: str,
                   client: anthropic.Anthropic) -> list:
    records = []
    for i, p in enumerate(patients):
        prompt = prompt_template.format(
            EF=p["EF"], EDV=p["EDV"], ESV=p["ESV"]
        )
        try:
            resp = client.messages.create(
                model="claude-opus-4-7",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            pred = extract_label(text)
        except Exception as e:
            print(f"  API error for {p['pid']}: {e}")
            text = ""
            pred = "NOR"

        correct = pred == p["gt"]
        records.append({
            "condition": condition_name,
            "pid": p["pid"],
            "gt": p["gt"],
            "pred": pred,
            "correct": correct,
            "llm_output": text,
            "EF": p["EF"],
            "EDV": p["EDV"],
            "ESV": p["ESV"],
        })
        status = "✓" if correct else "✗"
        print(f"  [{i+1:02d}/20] {p['pid']} GT={p['gt']:4s} PRED={pred:4s} {status}  "
              f"EF={p['EF']:6.1f}% EDV={p['EDV']:6.1f} ESV={p['ESV']:6.1f}")
        time.sleep(0.3)  # avoid rate limiting
    return records


def compute_metrics(records: list) -> dict:
    acc = sum(r["correct"] for r in records) / len(records)
    per_class_f1 = {}
    for cls in CLASSES:
        tp = sum(1 for r in records if r["gt"] == cls and r["pred"] == cls)
        fp = sum(1 for r in records if r["gt"] != cls and r["pred"] == cls)
        fn = sum(1 for r in records if r["gt"] == cls and r["pred"] != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_f1[cls] = round(f1, 3)
    return {"accuracy": round(acc, 3), "per_class_f1": per_class_f1,
            "macro_f1": round(sum(per_class_f1.values()) / len(CLASSES), 3)}


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY environment variable")
    client = anthropic.Anthropic(api_key=api_key)

    # Load data
    with open("data/metrics_acdc_val.json") as f:
        data = json.load(f)

    raw = data["MedSAM2_Dual"]  # list of 20 patient dicts

    # Build patient tables for GT and Pred metrics
    gt_patients = [{
        "pid": r["pid"], "gt": r["group"],
        "EF": r["gt_EF"], "EDV": r["gt_EDV"], "ESV": r["gt_ESV"],
    } for r in raw]

    pred_patients = [{
        "pid": r["pid"], "gt": r["group"],
        "EF": r["pred_EF"], "EDV": r["pred_EDV"], "ESV": r["pred_ESV"],
    } for r in raw]

    # --- Baselines ---
    print("\n=== Baselines ===")
    random_acc = 1 / len(CLASSES)
    print(f"  Random:        {random_acc:.1%}")

    ef_thresh_records = [{"gt": p["gt"], "pred": ef_threshold_baseline(p["EF"]),
                          "correct": ef_threshold_baseline(p["EF"]) == p["gt"]}
                         for p in gt_patients]
    ef_thresh_acc = sum(r["correct"] for r in ef_thresh_records) / len(ef_thresh_records)
    print(f"  EF-threshold:  {ef_thresh_acc:.1%}")

    # --- LLM Experiments ---
    all_records = []
    conditions = [
        ("A_GT_Direct",        gt_patients,   DIRECT_PROMPT,         "A: GT   + Direct    "),
        ("B_GT_CoT",           gt_patients,   COT_PROMPT,            "B: GT   + Naive CoT "),
        ("C_Pred_Direct",      pred_patients, DIRECT_PROMPT,         "C: Pred + Direct    "),
        ("D_Pred_CoT",         pred_patients, COT_PROMPT,            "D: Pred + Naive CoT "),
        ("E_GT_CalibCoT",      gt_patients,   CALIBRATED_COT_PROMPT, "E: GT   + Calib CoT "),
        ("F_Pred_CalibCoT",    pred_patients, CALIBRATED_COT_PROMPT, "F: Pred + Calib CoT "),
    ]

    summary = {}
    for cond_key, patients, prompt_tmpl, label in conditions:
        print(f"\n=== Condition {label} ===")
        records = run_experiment(patients, prompt_tmpl, cond_key, client)
        metrics = compute_metrics(records)
        summary[cond_key] = {"label": label, **metrics}
        all_records.extend(records)
        print(f"  Accuracy: {metrics['accuracy']:.1%}  Macro-F1: {metrics['macro_f1']:.3f}")
        print(f"  Per-class F1: {metrics['per_class_f1']}")

    # --- Save results ---
    output = {
        "baselines": {
            "random": {"accuracy": round(random_acc, 3)},
            "ef_threshold": {
                "accuracy": round(ef_thresh_acc, 3),
                **compute_metrics(ef_thresh_records)
            },
        },
        "conditions": summary,
        "records": all_records,
    }
    out_path = RESULTS_DIR / "llm_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n=== SUMMARY TABLE ===")
    print(f"{'Method':<25} {'Acc':>6} {'MacroF1':>8} {'NOR':>6} {'DCM':>6} {'HCM':>6} {'MINF':>6} {'RV':>6}")
    print("-" * 75)
    print(f"{'Random':25} {random_acc:>6.1%} {'—':>8}")
    print(f"{'EF-threshold':25} {ef_thresh_acc:>6.1%} {compute_metrics(ef_thresh_records)['macro_f1']:>8.3f}")
    for cond_key, info in summary.items():
        f1s = info["per_class_f1"]
        print(f"{info['label']:25} {info['accuracy']:>6.1%} {info['macro_f1']:>8.3f} "
              f"{f1s['NOR']:>6.3f} {f1s['DCM']:>6.3f} {f1s['HCM']:>6.3f} "
              f"{f1s['MINF']:>6.3f} {f1s['RV']:>6.3f}")


if __name__ == "__main__":
    main()
