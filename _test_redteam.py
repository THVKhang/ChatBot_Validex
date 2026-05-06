# -*- coding: utf-8 -*-
"""Red-team test: exercises 3 scenarios against the chunked generation pipeline.

Scenarios:
  1. RAG Blackout (DPKI) - should produce technical content without apologies
  2. HR Bait - title/intro must NOT contain HR terms
  3. Compliance Structure - must have conclusion heading, 700+ words, no repetition
"""
import os, sys, re

# Must set env vars BEFORE importing the pipeline
os.environ["USE_LIVE_LLM"] = "1"
os.environ["USE_STRUCTURED_OUTPUT"] = "1"
os.environ["ALLOW_HYBRID_FALLBACK"] = "1"
os.environ["ENFORCE_QUALITY_GATE"] = "0"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.langchain_pipeline import pipeline

SCENARIOS = [
    {
        "name": "1. RAG Blackout (DPKI)",
        "prompt": "Decentralized Public Key Infrastructure and Self-Sovereign Identity Systems",
    },
    {
        "name": "2. HR Bait",
        "prompt": "How does employee onboarding integrate with background screening and hiring compliance?",
    },
    {
        "name": "3. Compliance Structure (Spent Convictions)",
        "prompt": "Spent Convictions Scheme in the ACIC National Police Checking Service",
    },
]

HR_FORBIDDEN = {"onboarding", "hiring", "recruitment", "candidate", "talent acquisition", "employer"}

CONCLUSION_HEADING = "## Conclusion and Strategic Next Steps"


def word_count(text):
    return len(re.findall(r"\b\w+\b", text))


def check_hr_leaks(title, intro):
    leaks = []
    combined = (title + " " + intro).lower()
    for term in HR_FORBIDDEN:
        if term in combined:
            leaks.append(term)
    return leaks


def extract_intro(draft):
    """Extract text between # title and the first ## heading."""
    lines = draft.split("\n")
    intro_lines = []
    past_title = False
    for line in lines:
        if line.startswith("# ") and not past_title:
            past_title = True
            continue
        if past_title and line.startswith("## "):
            break
        if past_title:
            intro_lines.append(line)
    return " ".join(intro_lines).strip()


def run_scenario(scenario):
    print("\n" + "=" * 70)
    print("  SCENARIO: " + scenario["name"])
    print("  PROMPT:   " + scenario["prompt"][:60] + "...")
    print("=" * 70)

    result = pipeline.run(prompt=scenario["prompt"])
    generated = result.get("generated", {})
    runtime = result.get("runtime", {})

    draft = generated.get("draft", "")
    title = generated.get("title", "")
    outline = generated.get("outline", [])
    mode = runtime.get("generation_mode", "unknown")

    wc = word_count(draft)
    intro = extract_intro(draft)
    has_conclusion = CONCLUSION_HEADING in draft
    hr_leaks = check_hr_leaks(title, intro)

    print("\n  Generation Mode: " + mode)
    print("  Title: " + title)
    print("  Word Count: " + str(wc))
    print("  Has Conclusion Header: " + str(has_conclusion))
    print("  Outline: " + str(outline))
    if hr_leaks:
        print("  [WARN] HR LEAKS DETECTED: " + str(hr_leaks))
    else:
        print("  [PASS] No HR leaks in title/intro")

    # Print first 300 chars of intro
    print("\n  INTRO (first 300 chars):")
    print("  " + intro[:300])
    print("\n  DRAFT (first 500 chars):")
    print("  " + draft[:500])

    return {
        "scenario": scenario["name"],
        "mode": mode,
        "title": title,
        "wc": wc,
        "has_conclusion": has_conclusion,
        "hr_leaks": hr_leaks,
        "outline": outline,
    }


if __name__ == "__main__":
    results = []
    for scenario in SCENARIOS:
        try:
            r = run_scenario(scenario)
            results.append(r)
        except Exception as e:
            print("\n  [FAIL] SCENARIO FAILED: " + str(e))
            import traceback; traceback.print_exc()
            results.append({"scenario": scenario["name"], "error": str(e)})

    print("\n" + "=" * 70)
    print("  FINAL SCORECARD")
    print("=" * 70)
    for r in results:
        if "error" in r:
            print("  [FAIL] " + r["scenario"] + ": ERROR - " + r["error"])
            continue
        passed = r["has_conclusion"] and r["wc"] >= 500 and not r["hr_leaks"]
        status = "[PASS]" if passed else "[WARN]"
        print(
            "  " + status + " " + r["scenario"] + ": "
            "mode=" + r["mode"] + ", "
            "words=" + str(r["wc"]) + ", "
            "conclusion=" + ("YES" if r["has_conclusion"] else "NO") + ", "
            "hr_leaks=" + (str(r["hr_leaks"]) if r["hr_leaks"] else "none")
        )
