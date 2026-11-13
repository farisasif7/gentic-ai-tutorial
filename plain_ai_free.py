"""
plain_ai_free.py — plain AI: one question in, one answer out. No tools, no loop.
Compare with simple_medical_agent_free.py (agentic AI). Educational demo only.

python plain_ai_free.py
"""
import ollama

MODEL = "qwen2.5:3b"

SYSTEM = ("You are a medical information assistant. Give simple, general health information. "
          "Do not diagnose or give medicine doses. Always advise seeing a doctor.")

def ask_ai(question):
    reply = ollama.chat(model=MODEL, messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question}])
    return reply.message.content                                # one call, then done

if __name__ == "__main__":
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() == "quit":
            break
        if q:
            print("AI:", ask_ai(q), "\n")
