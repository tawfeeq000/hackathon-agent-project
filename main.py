import os
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Multi-Agent Coordinator Online"}

@app.get("/chat")
def chat(user_input: str = "hello"):
    # Hardcoded Logic for the Demo
    if "help" in user_input.lower() or "find" in user_input.lower():
        return {
            "coordinator_decision": "Delegated to Skill-Database-Agent",
            "final_output": f"DATABASE AGENT: Match found for '{user_input}'"
        }
    return {
        "coordinator_decision": "Handled by General Agent",
        "final_output": "GENERAL AGENT: I can help with that!"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)