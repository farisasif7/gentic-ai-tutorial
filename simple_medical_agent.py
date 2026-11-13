"""
simple_medical_agent.py — one LLM agent with one tool that answers general medical questions.
Educational demo only, not medical advice.

pip install anthropic
export ANTHROPIC_API_KEY="your-key"
python simple_medical_agent.py
"""
import anthropic

client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"
MAX_TURNS = 5                                                   # safety cap on tool loops

SYSTEM = ("You are a medical information assistant. Give simple, general health information. "
          "Do not diagnose or give medicine doses. For serious symptoms, tell the user to call "
          "Rescue 1122 or go to hospital. Always advise seeing a doctor. "
          "Use the lookup tool when the user asks about a specific disease.")

# --- The agent's one tool: a small knowledge base ---
KNOWLEDGE = {
    "diabetes": "High blood sugar. Signs: thirst, frequent urination, tiredness.",
    "dengue": "Mosquito-borne fever with headache, body pain and rash. Bleeding needs urgent care.",
    "hypertension": "High blood pressure, often silent. Reduce salt, exercise, see a doctor.",
}

def lookup(disease):
    return KNOWLEDGE.get(disease.strip().lower(), "Not found in knowledge base.")

TOOLS = [{
    "name": "lookup",
    "description": "Look up basic information about a disease. Known diseases: "
                   + ", ".join(KNOWLEDGE) + ".",
    "input_schema": {"type": "object",
                     "properties": {"disease": {"type": "string"}},
                     "required": ["disease"],
                     "additionalProperties": False},
    "strict": True,
}]

# --- The agent loop: think -> use tool -> answer ---
def ask_agent(question):
    messages = [{"role": "user", "content": question}]
    for _ in range(MAX_TURNS):
        reply = client.messages.create(model=MODEL, max_tokens=4000, system=SYSTEM,
                                       tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": reply.content})

        if reply.stop_reason == "refusal":
            return "Sorry, I can't help with that. Please consult a doctor."
        if reply.stop_reason != "tool_use":                     # final answer
            return "".join(b.text for b in reply.content if b.type == "text")

        results = []                                            # agent wants the tool
        for block in reply.content:
            if block.type == "tool_use":
                disease = block.input.get("disease", "")
                print(f"  [agent used tool: lookup({disease})]")
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": lookup(disease)})
        messages.append({"role": "user", "content": results})   # all results in ONE message

    return "Sorry, I couldn't finish answering. Please ask again or consult a doctor."

# --- Chat ---
if __name__ == "__main__":
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() == "quit":
            break
        if q:
            print("Agent:", ask_agent(q), "\n")
