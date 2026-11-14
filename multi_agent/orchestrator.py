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
import blockchain_ledger as chain

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


class OrchestratorSuspended(Exception):
    """Raised the moment a vulnerability flag crosses the suspension threshold."""


def _flag(flag_type, details, report_ref, log):
    """Appends one block to the global vulnerability chain and, if that flag
    type has now reached the suspension threshold, stops the orchestrator
    immediately by raising OrchestratorSuspended."""
    block = chain.append_flag(flag_type, details, report_ref)
    log(f"  [chain block #{block['index']}: {flag_type} -- {details}]")
    suspended, tripped_flag, count = chain.is_orchestrator_suspended()
    if suspended:
        log(f"  [ORCHESTRATOR SUSPENDED -- '{tripped_flag}' has reached {count} occurrences "
            f"on the vulnerability chain]")
        raise OrchestratorSuspended(tripped_flag)


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


def run_orchestrator(report_text, log=print, report_ref=None):
    """Runs the full orchestrator -> data_agent -> analysis_agent pipeline for one report.

    The LLM decides WHEN and WHETHER to call each tool and in what order.
    The code below guarantees WHAT DATA actually flows between the two
    specialist agents, instead of trusting a small local model to retype the
    full report text or the full diagnosis JSON correctly through its own
    tool-call arguments -- the same "model asks, program decides" principle
    the tutorial's single-agent loop uses, just applied between two agents.

    Every run is also checked against the vulnerability blockchain
    (blockchain_ledger.py): if a flag type has already reached the
    suspension threshold, the orchestrator refuses to run at all. During the
    run, any of the 5 vulnerability conditions found gets appended to that
    chain, and crossing the threshold mid-run halts processing immediately.

    Returns (final_answer_text, full_message_list) -- the message list is useful
    for saving a run transcript (see run_all.py).
    """
    messages = [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM},
        {"role": "user", "content": f"New patient medical report:\n\n{report_text}"},
    ]

    suspended, tripped_flag, count = chain.is_orchestrator_suspended()
    if suspended:
        msg = (f"ORCHESTRATOR SUSPENDED: flag '{tripped_flag}' has reached {count} "
               f"occurrences on the vulnerability chain. Human review required -- run "
               f"'python blockchain_ledger.py reset' after reviewing to resume.")
        log(f"  [{msg}]")
        return msg, messages

    last_diagnosis = None
    last_appointment = None

    try:
        # Flag 1 -- structural: this pipeline has no dedicated agent that
        # authorizes/validates a diagnosis before it is acted on, so every
        # run is itself a vulnerability worth logging.
        _flag("MISSING_VALIDATION_AGENT",
              "No authorization/validation agent confirmed this diagnosis before action.",
              report_ref, log)

        # Flag 4 -- the report itself may be empty or too short to trust.
        if len(report_text.strip()) < 20:
            _flag("EMPTY_OR_INVALID_REPORT",
                  f"Report text is only {len(report_text.strip())} characters.",
                  report_ref, log)

        for _ in range(MAX_TURNS):
            reply = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
            messages.append(reply.message)

            if not reply.message.tool_calls:  # final answer
                content = (reply.message.content or "").strip()
                if content:
                    return content, messages
                fallback = _fallback_summary(last_diagnosis, last_appointment)
                log("  [orchestrator gave an empty final answer -- using a code-built summary instead]")
                return fallback, messages

            for call in reply.message.tool_calls:
                name = call.function.name
                if name == "data_agent":
                    # Always analyze the REAL report text, regardless of what the
                    # model echoed back as the argument.
                    last_diagnosis = analyze_report(report_text)
                    result = last_diagnosis

                    # Flag 3 -- data_agent couldn't produce a usable diagnosis.
                    diag_name = str(last_diagnosis.get("diagnosis", "")).strip().lower()
                    risk = last_diagnosis.get("risk_level")
                    if not diag_name or diag_name == "unknown" or risk not in ("low", "medium", "high"):
                        _flag("LOW_CONFIDENCE_DIAGNOSIS",
                              f"data_agent returned diagnosis='{diag_name}', risk_level='{risk}'.",
                              report_ref, log)

                elif name == "analysis_agent":
                    model_arg = call.function.arguments.get("diagnosis", "")

                    if last_diagnosis is None:
                        # Flag 5 -- booking requested with nothing backing it.
                        _flag("ORPHAN_APPOINTMENT",
                              "analysis_agent was called with no prior valid data_agent "
                              "diagnosis in this run.",
                              report_ref, log)
                    else:
                        # Flag 2 -- does what the model is about to hand to
                        # analysis_agent still reflect what data_agent actually said?
                        actual_name = str(last_diagnosis.get("diagnosis", "")).strip().lower()
                        model_arg_str = (json.dumps(model_arg) if isinstance(model_arg, (dict, list))
                                          else str(model_arg)).lower()
                        if actual_name and actual_name not in model_arg_str:
                            _flag("DATA_INTEGRITY_MISMATCH",
                                  f"Model passed diagnosis={model_arg_str!r} to analysis_agent, "
                                  f"which does not match data_agent's actual diagnosis "
                                  f"{actual_name!r}.",
                                  report_ref, log)

                    # Use the diagnosis data_agent actually produced, not the
                    # model's possibly-truncated retelling of it.
                    diagnosis_for_booking = last_diagnosis if last_diagnosis else model_arg
                    last_appointment = book_appointment(diagnosis_for_booking)
                    result = last_appointment
                else:
                    result = {"error": f"unknown tool {name}"}

                log(f"  [orchestrator -> {name}({dict(call.function.arguments)})]")
                result_json = json.dumps(result)
                log(f"  [{name} result: {result_json}]")
                messages.append({"role": "tool", "tool_name": name, "content": result_json})

        return _fallback_summary(last_diagnosis, last_appointment), messages

    except OrchestratorSuspended as e:
        msg = (f"ORCHESTRATOR SUSPENDED mid-run: flag '{e}' just reached the suspension "
               f"threshold on the vulnerability chain. Halting before finishing this report. "
               f"Human review required -- run 'python blockchain_ledger.py reset' to resume.")
        return msg, messages


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            report = f.read()
        ref = sys.argv[1]
    else:
        print("Paste/type the medical report, then press Enter then Ctrl+Z (Windows) + Enter:")
        report = sys.stdin.read()
        ref = "stdin"

    answer, _ = run_orchestrator(report, report_ref=ref)
    print("\nOrchestrator final summary:\n" + answer)
