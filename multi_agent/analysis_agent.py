"""
analysis_agent.py -- specialist agent #2.

Role (from the orchestrator/specialist architecture): the "action agent" --
"books, sends, updates systems". Here: takes a diagnosis and books a doctor
appointment, then is the ONLY agent allowed to write to disk (appointments.json).
Least privilege: data_agent cannot touch this file.

Doctor/specialist matching and appointment slots use a simple fixed mapping
table and a tiny mock schedule (no real booking system involved).

Educational demo only, not medical advice.
"""
import json
import os
from datetime import datetime, timedelta

APPOINTMENTS_FILE = os.path.join(os.path.dirname(__file__), "appointments.json")

SPECIALIST_MAP = {
    "dengue": "General Physician",
    "malaria": "General Physician",
    "diabetes": "Endocrinologist",
    "hypertension": "Cardiologist",
    "asthma": "Pulmonologist",
    "migraine": "Neurologist",
}
DEFAULT_SPECIALIST = "General Physician"

DOCTORS = {
    "General Physician": "Dr. A. Khan",
    "Endocrinologist": "Dr. S. Raza",
    "Cardiologist": "Dr. F. Malik",
    "Pulmonologist": "Dr. N. Qureshi",
    "Neurologist": "Dr. H. Siddiqui",
}

_slot_counter = 0  # rotates appointment times forward across bookings in this process


def _next_slot():
    global _slot_counter
    base = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0) + timedelta(days=1)
    slot = base + timedelta(hours=_slot_counter)
    _slot_counter += 1
    return slot.strftime("%Y-%m-%d %H:%M")


def _load_appointments():
    if os.path.exists(APPOINTMENTS_FILE):
        with open(APPOINTMENTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_appointments(appointments):
    with open(APPOINTMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(appointments, f, indent=2)


def book_appointment(diagnosis_input):
    """Analysis/action agent: diagnosis (dict or JSON string) in, appointment dict out.

    Also appends the appointment to appointments.json -- the one file this
    multi-agent system is allowed to write.
    """
    if isinstance(diagnosis_input, dict):
        diagnosis_data = diagnosis_input
    else:
        try:
            diagnosis_data = json.loads(diagnosis_input)
        except (json.JSONDecodeError, TypeError):
            diagnosis_data = {"diagnosis": str(diagnosis_input), "risk_level": "medium"}

    diagnosis_name = str(diagnosis_data.get("diagnosis", "unknown")).strip().lower()
    risk_level = diagnosis_data.get("risk_level", "medium")
    symptoms = diagnosis_data.get("symptoms", [])

    specialist = SPECIALIST_MAP.get(diagnosis_name, DEFAULT_SPECIALIST)
    doctor = DOCTORS.get(specialist, DOCTORS[DEFAULT_SPECIALIST])
    slot = _next_slot()

    appointments = _load_appointments()
    appointment = {
        "appointment_id": f"APT-{len(appointments) + 1:04d}",
        "diagnosis": diagnosis_name,
        "symptoms": symptoms,
        "risk_level": risk_level,
        "specialist": specialist,
        "doctor": doctor,
        "datetime": slot,
        "booked_at": datetime.now().isoformat(timespec="seconds"),
    }
    appointments.append(appointment)
    _save_appointments(appointments)
    return appointment


if __name__ == "__main__":
    sample_diagnosis = {"diagnosis": "dengue", "symptoms": ["fever", "rash"], "risk_level": "medium"}
    print(json.dumps(book_appointment(sample_diagnosis), indent=2))
