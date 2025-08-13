from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_engine import query_index
from scraper import get_scraped_context
from openai import OpenAI
import os
from dotenv import load_dotenv

# ✅ Load environment variables (.env di local, Variables di Railway)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY tidak ditemukan. Set di Railway Variables.")

# ✅ Inisialisasi FastAPI App
app = FastAPI()

# ✅ CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Production: ganti ke ["https://planville.de"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Inisialisasi OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ Struktur permintaan dari frontend
class ChatRequest(BaseModel):
    message: str
    lang: str = "de"

# ✅ Keyword-based intent detection
VALID_KEYWORDS = [
    "photovoltaik", "photovoltaics", "dach", "roof",
    "wärmepumpe", "heat pump", "klimaanlage", "air conditioner",
    "beratung", "consultation", "angebot", "quote",
    "kontakt", "contact", "termin", "appointment", "montage", "installation"
]

def is_valid_intent(message: str) -> bool:
    msg = message.lower()
    return any(keyword in msg for keyword in VALID_KEYWORDS)

# ✅ Endpoint utama chatbot
@app.post("/chat")
async def chat(request: ChatRequest):
    print(f"[📨 Request] Language: {request.lang} | Message: {request.message}")

    if not is_valid_intent(request.message):
        fallback_msg = {
            "de": "Ich kann nur Fragen zu Planville Dienstleistungen beantworten. "
                  "Bitte kontaktieren Sie uns direkt unter: https://planville.de/kontakt",
            "en": "I can only answer questions related to Planville services. "
                  "Please contact us directly here: https://planville.de/kontakt"
        }
        return {"reply": fallback_msg.get(request.lang, fallback_msg["de"])}

    try:
        # 🧠 Ambil konteks dari RAG
        context_docs = query_index(request.message)

        # 🔄 Fallback ke scraping
        if not context_docs:
            print("[⚠️] RAG kosong → menggunakan fallback scraper.")
            context_docs = get_scraped_context(request.message)

        context_text = "\n".join(context_docs)

        prompt = f"""
Du bist ein professioneller Kundenservice-Assistent von Planville GmbH.
Antworte bitte höflich, direkt und hilfreich basierend auf dem folgenden Kontext.

🔎 Frage:
{request.message}

📄 Kontext:
{context_text}
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )

        reply_text = response.choices[0].message.content.strip()

        if not reply_text:
            return {"reply": (
                "Entschuldigung, ich habe leider keine passende Information zu Ihrer Anfrage.\n\n"
                "📞 Kontaktieren Sie unser Team direkt:\n"
                "👉 https://planville.de/kontakt"
            )}

        return {"reply": reply_text}

    except Exception as e:
        print(f"[❌ GPT ERROR]: {e}")
        return {"reply": (
            "Es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut "
            "oder kontaktieren Sie uns direkt.\n\n➡️ https://planville.de/kontakt"
        )}

# ✅ Healthcheck
@app.get("/healthz")
def health_check():
    return {"status": "ok"}
