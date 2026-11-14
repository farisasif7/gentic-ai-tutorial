"""
data_agent.py -- specialist agent #1.

Role (from the orchestrator/specialist architecture): "fetches records, documents"
and analyzes them. Here: reads a raw medical report and extracts a diagnosis.
This agent never books anything and never writes files -- least privilege.

Educational demo only, not medical advice.
"""
import json
import ollama

MODEL = "qwen2.5:3b"

DATA_AGENT_SYSTEM = (
    "You are a careful medical data analysis agent. You will be given the raw text of a "
    "patient's medical report. Think about the full combination of symptoms before deciding -- "
    "do not jump to the most common illness if the symptoms point somewhere more specific. "
    "For example: fever + headache + pain behind the eyes + joint/muscle pain + rash, "
    "especially with local mosquito exposure, points to dengue, not flu. "
    "Excess thirst + frequent urination + weight loss points to diabetes. "
    "High blood pressure readings on their own point to hypertension. "
    "Sudden vomiting + diarrhea soon after a meal points to food poisoning / gastroenteritis. "
    "Extract: the single most likely primary diagnosis (a short disease/condition name, "
    "lowercase), a short list of the key symptoms mentioned, and a risk level of 'low', "
    "'medium', or 'high'. This is an educational demo, not real medical advice -- never say that. "
    "Respond with ONLY a JSON object, no extra text, exactly in this shape: "
    '{"diagnosis": "...", "symptoms": ["...", "..."], "risk_level": "low|medium|high"}'
)


def analyze_report(report_text):
    """Diagnosis agent: report text in, diagnosis JSON (as a dict) out."""
    reply = ollama.chat(
        model=MODEL,
        format="json",
        options={"temperature": 0.1},
        messages=[
            {"role": "system", "content": DATA_AGENT_SYSTEM},
            {"role": "user", "content": report_text},
        ],
    )
    try:
        data = json.loads(reply.message.content)
    except (json.JSONDecodeError, TypeError):
        data = {"diagnosis": "unknown", "symptoms": [], "risk_level": "medium"}

    data.setdefault("diagnosis", "unknown")
    data.setdefault("symptoms", [])
    data.setdefault("risk_level", "medium")
    return data


if __name__ == "__main__":
    sample = (
        "Patient reports high fever for 3 days, severe headache, pain behind the eyes, "
        "joint pain and a skin rash. No bleeding observed."
    )
    print(json.dumps(analyze_report(sample), indent=2))
