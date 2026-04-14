import os
import logging
import datetime
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from google.adk import Agent
from google.adk.agents import SequentialAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# --- 1. Setup Logging ---
try:
    import google.cloud.logging
    cloud_logging_client = google.cloud.logging.Client()
    cloud_logging_client.setup_logging()
except Exception:
    logging.basicConfig(level=logging.INFO)

load_dotenv()
model_name = os.getenv("MODEL", "gemini-2.0-flash")

# --- 2. Database Setup ---
# Leaving Client() arguments empty is the most stable way to deploy on Google
# Cloud — it auto-detects the project and uses the (default) database.
try:
    from google.cloud import datastore
    db = datastore.Client()
except ImportError:
    logging.warning("google-cloud-datastore not installed. DB operations will fail gracefully.")
    datastore = None  # type: ignore[assignment]
    db = None
except Exception as _db_err:
    logging.warning(f"Datastore client unavailable: {_db_err}. DB operations will fail gracefully.")
    # datastore module was imported successfully; only the client failed
    db = None

# ================= 3. TOOLS =================

def add_task(title: str) -> str:
    """Adds a new task to the workspace."""
    if db is None:
        return "Database not available."
    try:
        key = db.key('Task')
        task = datastore.Entity(key=key)
        task.update({
            'title': title,
            'completed': False,
            'created_at': datetime.datetime.now()
        })
        db.put(task)
        return f"Success: Task '{title}' saved (ID: {task.key.id})."
    except Exception as e:
        logging.error(f"DB Error in add_task: {e}")
        return f"Database Error: {str(e)}"

def list_tasks() -> str:
    """Lists all current tasks."""
    if db is None:
        return "Database not available."
    try:
        query = db.query(kind='Task')
        tasks = list(query.fetch())
        if not tasks:
            return "Your task list is empty."

        res = ["📋 Current Tasks:"]
        for t in tasks:
            status = "✅" if t.get('completed') else "⏳"
            res.append(f"{status} {t.get('title')} (ID: {t.key.id})")
        return "\n".join(res)
    except Exception as e:
        return f"Database Error: {str(e)}"

def complete_task(task_id: str) -> str:
    """Marks a task as complete. Input must be the numeric ID."""
    if db is None:
        return "Database not available."
    try:
        numeric_id = int(''.join(filter(str.isdigit, task_id)))
        key = db.key('Task', numeric_id)
        task = db.get(key)
        if task:
            task['completed'] = True
            db.put(task)
            return f"Task {numeric_id} marked as done."
        return f"Task {numeric_id} not found."
    except Exception as e:
        return f"Error processing task ID: {str(e)}"

def add_note(title: str, content: str) -> str:
    """Saves a detailed note to the workspace."""
    if db is None:
        return "Database not available."
    try:
        key = db.key('Note')
        note = datastore.Entity(key=key)
        note.update({'title': title, 'content': content, 'at': datetime.datetime.now()})
        db.put(note)
        return f"Note '{title}' saved successfully."
    except Exception as e:
        return f"Database Error: {str(e)}"

# ================= 4. AGENTS =================

def add_prompt_to_state(tool_context: ToolContext, prompt: str):
    """Internal tool to bridge user intent across the agent workflow."""
    tool_context.state["PROMPT"] = prompt
    return {"status": "ok"}

def workspace_instruction(ctx):
    # This pulls from the state we set in the root_agent
    user_prompt = ctx.state.get("PROMPT", "Welcome the user.")
    return f"""
You are the Workspace Executive Assistant.
Always start with a polite, professional greeting.
Then, use your tools to complete this request: {user_prompt}
"""

def root_instruction(ctx):
    # Pulls the prompt directly from the session state set at request time
    raw_input = ctx.state.get("user_input", "Hello")
    return f"""
1. Save this user input using 'add_prompt_to_state': {raw_input}
2. Hand off control to the 'workflow' agent.
"""

workspace_agent = Agent(
    name="workspace",
    model=model_name,
    instruction=workspace_instruction,
    tools=[add_task, list_tasks, complete_task, add_note]
)

workflow = SequentialAgent(
    name="workflow",
    sub_agents=[workspace_agent]
)

root_agent = Agent(
    name="root",
    model=model_name,
    instruction=root_instruction,
    tools=[add_prompt_to_state],
    sub_agents=[workflow]
)

# ================= 5. RUNNER SETUP =================

_APP_NAME = "workspace_app"
_session_service = InMemorySessionService()
_runner = Runner(agent=root_agent, app_name=_APP_NAME, session_service=_session_service)

# ================= 6. API =================

app = FastAPI(title="Productivity Agent Hub", version="1.0.0")

class UserRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {
        "message": "Productivity Agent Hub is running",
        "status": "ok",
        "version": "1.0.0",
        "endpoints": {
            "chat": "POST /api/v1/workspace/chat",
            "health": "GET /health"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy", "database": "connected" if db else "disconnected"}

@app.post("/api/v1/workspace/chat")
async def chat(request: UserRequest):
    try:
        user_id = "default_user"
        # Create a new session and inject the user prompt into state so that
        # root_instruction (and workspace_instruction) can read it.
        session = await _session_service.create_session(
            app_name=_APP_NAME,
            user_id=user_id,
            state={"user_input": request.prompt}
        )

        content = types.Content(
            role="user",
            parts=[types.Part(text=request.prompt)]
        )

        final_reply = ""
        async for event in _runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_reply = event.content.parts[0].text

        return {
            "status": "success",
            "reply": final_reply if final_reply else "Request processed."
        }

    except Exception as e:
        logging.error(f"Chat Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
