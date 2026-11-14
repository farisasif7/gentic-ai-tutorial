"""
orchestrator.py -- the orchestrator agent.

"Splits the goal, assigns work, merges results." It is itself an LLM (via
Ollama + tool-calling, same pattern as simple_medical_agent_free.py in the
parent folder) that decides WHEN to call each specialist tool:

  data_agent      -> reads the medical report, returns a diagnosis (no writes)
  analysis_agent  -> books the appointment for that diagnosis, writes appointments.json

The orchestrator never diagnoses or books anything itself; it only routes
work to the two specialists and merges their results into one final answer.

Educational demo only, not medical advice.

Run a single report:
    python orchestrator.py reports/report_1_dengue.txt
"""
import json
import sys
import ollama

from data_agent import analyze_report
from analysis_agent import book_appointment

MODEL = "qwen2.5:3b"
MAX_TURNS = 6  # safety cap: data_agent call + analysis_agent call + final answer, with room to spare

ORCHESTRATOR_SYSTEM = (
    "You are the orchestrator agent in a small multi-agent healthcare system. "
    "You do not diagnose patients and you do not book appointments yourself -- you only split "
    "the goal, assign work to the right specialist tool, and merge their results. "
    "You have exactly two tools: "
    "'data_agent' (reads a medical report and returns a diagnosis as JSON) and "
    "'analysis_agent' (books a doctor appointment for a diagnosis and saves it). "
    "For every medical report you are given, always do this in order: "
    "1) call data_agent exactly once with the full report text; "
    "2) call analysis_agent exactly once, passing it the diagnosis JSON that data_agent returned; "
    "3) once both tools have returned, write ONE short final summary for the patient: the "
    "diagnosis, the risk level, and the booked doctor, specialty and appointment date/time. "
    "Never skip a step and never invent a diagnosis or appointment yourself. "
    "This is an educational demo, not real medical advice."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "data_agent",
            "description": "Analyze a patient's medical report text and return a diagnosis, "
                            "symptoms and risk level as JSON.",
            "parameters": {
                "type": "object",
                "properties": {"report_text": {"type": "string"}},
                "required": ["report_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analysis_agent",
            "description": "Book a doctor appointment for a given diagnosis (the JSON that "
                            "data_agent returned) and store it in appointments.json.",
            "parameters": {
                "type": "object",
                "properties": {"diagnosis": {"type": "string"}},
                "required": ["diagnosis"],
            },
        },
    },
]


def _fallback_summary(last_diagnosis, last_appointment):
    """Built in code, not by the model -- used only if the model's own final
    answer comes back empty (small local models sometimes do this)."""
    if not last_diagnosis or not last_appointment:
        return "Sorry, the orchestrator could not finish this report. Please try again."
    return (
        f"Diagnosis: {last_diagnosis.get('diagnosis', 'unknown')} "
        f"(risk level: {last_diagnosis.get('risk_level', 'unknown')}).\n"
        f"Appointment booked: {last_appointment.get('doctor')} "
        f"({last_appointment.get('specialist')}) on {last_appointment.get('datetime')}, "
        f"ref {last_appointment.get('appointment_id')}.\n"
        "Please consult the doctor at your appointment; this is not a final medical diagnosis."
    )


def run_orchestrator(report_text, log=print):
    """Runs the full orchestrator -> data_agent -> analysis_agent pipeline for one report.

    The LLM decides WHEN and WHETHER to call each tool and in what order.
    The code below guarantees WHAT DATA actually flows between the two
    specialist agents, instead of trusting a small local model to retype the
    full report text or the full diagnosis JSON correctly through its own
    tool-call arguments -- the same "model asks, program decides" principle
    the tutorial's single-agent loop uses, just applied between two agents.

    Returns (final_answer_text, full_message_list) -- the message list is useful
    for saving a run transcript (see run_all.py).
    """
    messages = [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM},
        {"role": "user", "content": f"New patient medical report:\n\n{report_text}"},
    ]

    last_diagnosis = None
    last_appointment = None

    for _ in range(MAX_TURNS):
        reply = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
        messages.append(reply.message)

        if not reply.message.tool_calls:  # final answer
            content = (reply.message.content or "").strip()
            if content:
                return content, messages
            fallback = _fallback_summary(last_diagnosis, last_appointment)
            log(f"  [orchestrator gave an empty final answer -- using a code-built summary instead]")
            return fallback, messages

        for call in reply.message.tool_calls:
            name = call.function.name
            if name == "data_agent":
                # Always analyze the REAL report text, regardless of what the
                # model echoed back as the argument.
                last_diagnosis = analyze_report(report_text)
                result = last_diagnosis
            elif name == "analysis_agent":
                # Use the diagnosis data_agent actually produced, not the
                # model's possibly-truncated retelling of it.
                diagnosis_for_booking = last_diagnosis if last_diagnosis else call.function.arguments.get("diagnosis", "")
                last_appointment = book_appointment(diagnosis_for_booking)
                result = last_appointment
            else:
                result = {"error": f"unknown tool {name}"}

            log(f"  [orchestrator -> {name}({dict(call.function.arguments)})]")
            result_json = json.dumps(result)
            log(f"  [{name} result: {result_json}]")
            messages.append({"role": "tool", "tool_name": name, "content": result_json})

    return _fallback_summary(last_diagnosis, last_appointment), messages


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            report = f.read()
    else:
        print("Paste/type the medical report, then press Enter then Ctrl+Z (Windows) + Enter:")
        report = sys.stdin.read()

    answer, _ = run_orchestrator(report)
    print("\nOrchestrator final summary:\n" + answer)
