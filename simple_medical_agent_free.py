"""
simple_medical_agent_free.py — same agent as simple_medical_agent.py, but runs a free local
model through Ollama. No API key, no cost. Educational demo only, not medical advice.

Install Ollama from https://ollama.com, then:
ollama pull qwen2.5:3b
pip install ollama
python simple_medical_agent_free.py
"""
import ollama

MODEL = "qwen2.5:3b"                                            # small, free, supports tools
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
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look up basic information about a disease. Known diseases: "
                       + ", ".join(KNOWLEDGE) + ".",
        "parameters": {"type": "object",
                       "properties": {"disease": {"type": "string"}},
                       "required": ["disease"]},
    },
}]

# --- The agent loop: think -> use tool -> answer ---
def ask_agent(question):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    for _ in range(MAX_TURNS):
        reply = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
        messages.append(reply.message)

        if not reply.message.tool_calls:                        # final answer
            return reply.message.content

        for call in reply.message.tool_calls:                   # agent wants the tool
            disease = call.function.arguments.get("disease", "")
            print(f"  [agent used tool: lookup({disease})]")
            messages.append({"role": "tool", "tool_name": "lookup",
                             "content": lookup(disease)})

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
