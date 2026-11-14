"""
blockchain_ledger.py -- a small hash-linked, append-only ledger (the
"blockchain") that records vulnerability flags raised while agents hand
data to each other in this system.

Each block links to the previous block's hash (like a real blockchain), so
any edit to an old block breaks every hash after it -- tamper-evident, not
just a plain log file. There is no mining/consensus here: this is a single
local ledger shared by every run of the pipeline (that's the "global chain"),
which is enough to demonstrate the integrity + audit idea for this exercise.

The 5 vulnerability flags this ledger understands:

  MISSING_VALIDATION_AGENT   No dedicated agent in this pipeline authorizes /
                              validates a diagnosis before it is acted on.
                              Structural gap -- raised on every run.
  DATA_INTEGRITY_MISMATCH    The diagnosis the orchestrator LLM tried to pass
                              to analysis_agent did not match what data_agent
                              actually returned (data can be altered/lost
                              between agents).
  LOW_CONFIDENCE_DIAGNOSIS   data_agent returned "unknown"/empty diagnosis or
                              no usable risk level.
  EMPTY_OR_INVALID_REPORT    The medical report text was empty/too short to
                              analyze.
  ORPHAN_APPOINTMENT         analysis_agent was invoked without a valid
                              diagnosis from data_agent earlier in the SAME
                              run (booking with nothing backing it).

SUSPENSION_THRESHOLD = 5: once any single flag type has been appended 5
times across the whole chain, the orchestrator must refuse to run until a
human resets/reviews the chain (see reset_chain()).
"""
import hashlib
import json
import os
from datetime import datetime

HERE = os.path.dirname(__file__)
CHAIN_FILE = os.path.join(HERE, "vulnerability_chain.json")

SUSPENSION_THRESHOLD = 5

FLAG_TYPES = {
    "MISSING_VALIDATION_AGENT",
    "DATA_INTEGRITY_MISMATCH",
    "LOW_CONFIDENCE_DIAGNOSIS",
    "EMPTY_OR_INVALID_REPORT",
    "ORPHAN_APPOINTMENT",
}

GENESIS_HASH = "0" * 64


def _hash_block(block):
    """sha256 over the block's content fields (everything except its own hash)."""
    payload = {k: v for k, v in block.items() if k != "hash"}
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_chain():
    if os.path.exists(CHAIN_FILE):
        with open(CHAIN_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    genesis = {
        "index": 0,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "flag_type": "GENESIS",
        "details": "Vulnerability chain initialized.",
        "report_ref": None,
        "previous_hash": GENESIS_HASH,
    }
    genesis["hash"] = _hash_block(genesis)
    return [genesis]


def save_chain(chain):
    with open(CHAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(chain, f, indent=2)


def append_flag(flag_type, details, report_ref=None):
    """Appends one vulnerability flag block to the global chain and returns it."""
    if flag_type not in FLAG_TYPES:
        raise ValueError(f"Unknown flag_type: {flag_type}")

    chain = load_chain()
    previous = chain[-1]
    block = {
        "index": previous["index"] + 1,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "flag_type": flag_type,
        "details": details,
        "report_ref": report_ref,
        "previous_hash": previous["hash"],
    }
    block["hash"] = _hash_block(block)
    chain.append(block)
    save_chain(chain)
    return block


def count_flag(flag_type):
    chain = load_chain()
    return sum(1 for b in chain if b["flag_type"] == flag_type)


def flag_counts():
    chain = load_chain()
    counts = {ft: 0 for ft in FLAG_TYPES}
    for b in chain:
        if b["flag_type"] in counts:
            counts[b["flag_type"]] += 1
    return counts


def is_orchestrator_suspended():
    """Returns (suspended: bool, tripped_flag: str|None, count: int)."""
    counts = flag_counts()
    for flag_type, count in counts.items():
        if count >= SUSPENSION_THRESHOLD:
            return True, flag_type, count
    return False, None, 0


def verify_chain():
    """Recomputes every block's hash and checks the previous_hash links.
    Returns (valid: bool, problem_index: int|None) -- tamper-evidence check."""
    chain = load_chain()
    for i, block in enumerate(chain):
        if _hash_block(block) != block["hash"]:
            return False, block["index"]
        if i > 0 and block["previous_hash"] != chain[i - 1]["hash"]:
            return False, block["index"]
    return True, None


def reset_chain():
    """Human action: wipes the chain back to a fresh genesis block.
    This is the only way to un-suspend the orchestrator."""
    if os.path.exists(CHAIN_FILE):
        os.remove(CHAIN_FILE)
    return load_chain()


def _print_status():
    chain = load_chain()
    valid, problem_index = verify_chain()
    suspended, tripped_flag, count = is_orchestrator_suspended()
    print(f"Chain length: {len(chain)} block(s) (including genesis)")
    print(f"Chain integrity: {'OK' if valid else f'TAMPERED at block {problem_index}'}")
    print("Flag counts:")
    for flag_type, c in flag_counts().items():
        marker = " <-- at/over threshold" if c >= SUSPENSION_THRESHOLD else ""
        print(f"  {flag_type}: {c}/{SUSPENSION_THRESHOLD}{marker}")
    if suspended:
        print(f"\nORCHESTRATOR SUSPENDED -- '{tripped_flag}' reached {count} occurrences.")
    else:
        print("\nOrchestrator is ACTIVE (no flag has reached the threshold).")


if __name__ == "__main__":
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    if command == "status":
        _print_status()
    elif command == "verify":
        valid, problem_index = verify_chain()
        print("Chain OK" if valid else f"Chain TAMPERED at block {problem_index}")
    elif command == "reset":
        reset_chain()
        print("Vulnerability chain reset. Orchestrator un-suspended.")
    else:
        print("Usage: python blockchain_ledger.py [status|verify|reset]")
