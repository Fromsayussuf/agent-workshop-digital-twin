from dotenv import load_dotenv
from openai import OpenAI
from agents import Agent, Runner, function_tool

import os
import csv
import chromadb
import streamlit as st
import smtplib
import requests
from email.mime.text import MIMEText
from datetime import datetime, timedelta, timezone

load_dotenv()

# --- OpenRouter Setup ---
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENAI_API_KEY")
)

# -------------------------
# RAG PART
# -------------------------

def load_documents():
    docs = []
    folder = "knowledge"
    for file in os.listdir(folder):
        with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
            docs.append(f.read())
    return docs

def chunk_text(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

def get_embedding(text):
    response = client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="can_knowledge")

docs = load_documents()

for doc_index, doc in enumerate(docs):
    chunks = chunk_text(doc)
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[get_embedding(chunk)],
            ids=[f"id_{doc_index}_{i}_{hash(chunk)}"]
        )

def retrieve_context(question):
    embedding = get_embedding(question)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=3
    )
    return "\n".join(results["documents"][0])

# -------------------------
# CALENDLY HELPERS
# -------------------------

def calendly_headers():
    token = os.getenv("CALENDLY_TOKEN")
    if not token:
        raise ValueError("CALENDLY_TOKEN is not set.")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

def get_calendly_user_uri():
    res = requests.get(
        "https://api.calendly.com/users/me",
        headers=calendly_headers(),
        timeout=15
    )
    return res.json()["resource"]["uri"]

def get_calendly_event_type_uri():
    user_uri = get_calendly_user_uri()
    res = requests.get(
        "https://api.calendly.com/event_types",
        headers=calendly_headers(),
        params={"user": user_uri},
        timeout=15
    )
    collection = res.json().get("collection", [])
    return collection[0]["uri"]

def format_slot_for_humans(iso_str: str):
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return dt.strftime("%A %d %B %Y %H:%M UTC")


@function_tool
def save_contact(name: str, company: str, intent: str):
    """Save extracted contact information to contacts.csv"""
    try:
        filename = "contacts.csv"
        file_exists = os.path.exists(filename)
        
        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "name", "company", "intent"])
            
            timestamp = datetime.now().isoformat()
            writer.writerow([timestamp, name.strip(), company.strip(), intent.strip()])
        
        return f"Contact saved: {name} from {company} — Intent: {intent}"
    except Exception as e:
        return f"Failed to save contact: {str(e)}"

# -------------------------
# OTHER TOOLS
# -------------------------

@function_tool
def search_knowledge(question: str):
    """Search Can's knowledge base for relevant context."""
    return retrieve_context(question)

@function_tool
def notify_owner(question: str):
    """Send email + push notification to Can for unanswered questions."""
    print("Notifying Can about unanswered question:", question)

    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    owner_email = os.getenv("OWNER_EMAIL")

    if smtp_host and smtp_user and smtp_pass and owner_email:
        msg = MIMEText(f"Unanswered question:\n\n{question}", "plain", "utf-8")
        msg["Subject"] = "Digital Twin - Unanswered Question"
        msg["From"] = smtp_user
        msg["To"] = owner_email

        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, owner_email, msg.as_string())

    pushover_token = os.getenv("PUSHOVER_TOKEN")
    pushover_user = os.getenv("PUSHOVER_USER")

    if pushover_token and pushover_user:
        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": pushover_token,
                "user": pushover_user,
                "title": "Digital Twin",
                "message": f"Unanswered question: {question}"
            },
            timeout=10
        )

    return "I notified Can about this unanswered question."

@function_tool
def get_available_slots(days_ahead: int = 7):
    """Get available Calendly slots for the next few days."""
    try:
        event_type_uri = get_calendly_event_type_uri()

        start = datetime.now(timezone.utc) + timedelta(hours=2)
        end = start + timedelta(days=days_ahead)

        res = requests.get(
            "https://api.calendly.com/event_type_available_times",
            headers=calendly_headers(),
            params={
                "event_type": event_type_uri,
                "start_time": start.isoformat().replace("+00:00", "Z"),
                "end_time": end.isoformat().replace("+00:00", "Z"),
            },
            timeout=20
        )

        slots = res.json().get("collection", [])
        if not slots:
            return "No available slots found in Calendly."

        lines = ["Available slots:"]
        for slot in slots[:5]:
            lines.append(f"- {format_slot_for_humans(slot['start_time'])}")

        return "\n".join(lines)

    except Exception as e:
        return f"Could not fetch Calendly slots: {str(e)}"

@function_tool
def book_meeting(name: str, email: str, slot: str, topic: str = ""):
    """
    Book a Calendly meeting.
    slot must be an ISO datetime string like 2026-04-24T14:00:00Z
    """
    try:
        event_type_uri = get_calendly_event_type_uri()

        payload = {
            "event_type": event_type_uri,
            "start_time": slot,
            "invitee": {
                "name": name,
                "email": email,
                "timezone": "UTC"
            },
            "location": {
                "kind": "google_conference"
            },
        }

        if topic:
            payload["questions_and_answers"] = [
                {
                    "question": "What would you like to discuss?",
                    "answer": topic,
                    "position": 0
                }
            ]

        res = requests.post(
            "https://api.calendly.com/invitees",
            headers=calendly_headers(),
            json=payload,
            timeout=20
        )

        data = res.json()["resource"]
        start_time = data.get("start_time", slot)

        return f"Meeting booked successfully for {name} at {format_slot_for_humans(start_time)}."

    except Exception as e:
        print(f"Error booking Calendly meeting: {str(e)}")
        return f"Could not book Calendly meeting: {str(e)}"

# -------------------------
# AGENT
# -------------------------

agent = Agent(
    name="Can's Digital Twin",
    model="openai/gpt-4o-mini",
    instructions="""
You are the AI digital twin of Can Özfuttu.

Core Rules:
1. When the user asks a question about Can, first use search_knowledge tool. Answer using ONLY the retrieved context.
2. If the context is not enough, say you don't know and use notify_owner tool. Do not ask the user — just call it.
3. For scheduling: Use get_available_slots first. Only use book_meeting when you have name, email, and exact ISO slot.
4. Be concise, natural, and professional at all times.

Contact Extraction Rules:
- If the user clearly introduces themselves (examples: "Hi, I’m John from Google, we’re hiring...", "I'm a recruiter from Acme...", "Hey, I’m a founder at a startup looking to collaborate..."), extract name, company, and intent, then call the save_contact tool.
- Only trigger save_contact when you are reasonably confident it's a genuine self-introduction with enough details.
- Do NOT trigger on casual or vague mentions like "I work at Google", "Google is great", "I’m from Acme", or general questions.
- NEVER mention words like "contact extraction", "tool", "saving contact", or any internal logic to the user.
- When you decide not to save any contact, reply in a completely natural and professional way without any meta-commentary.
- After successfully saving a contact, you can give a short, natural acknowledgment like "Thanks for the introduction!" or "Noted, nice to meet you!".
""",
    tools=[
        search_knowledge,
        notify_owner,
        get_available_slots,
        book_meeting,
        save_contact,
    ],
)


st.title("Can's Digital Twin")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Ask about Can:")

if question:
    st.session_state.history.append({"role": "user", "content": question})

    result = Runner.run_sync(agent, st.session_state.history)

    st.session_state.history = result.to_input_list()

    st.write(result.final_output)