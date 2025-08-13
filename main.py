from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_engine import query_index
from scraper import get_scraped_context
from openai import OpenAI
import os
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

# ✅ Inisialisasi FastAPI App
app = FastAPI()

# ✅ CORS Middleware – izinkan akses frontend lokal/frontend live
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Production: ubah ke ["https://planville.de"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Inisialisasi OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Struktur Permintaan dari Frontend
class ChatRequest(BaseModel):
    message: str
    lang: str = "de"  # default bahasa Jerman

# ✅ Keyword-based intent detection (untuk keamanan jawaban)
VALID_KEYWORDS = [
    "photovoltaik", "photovoltaics", "dach", "roof",
    "wärmepumpe", "heat pump", "klimaanlage", "air conditioner",
    "beratung", "consultation", "angebot", "quote",
    "kontakt", "contact", "termin", "appointment", "montage", "installation"
]

def is_valid_intent(message: str) -> bool:
    """Periksa apakah input user mengandung keyword valid"""
    msg = message.lower()
    return any(keyword in msg for keyword in VALID_KEYWORDS)

# ✅ Endpoint utama chatbot
@app.post("/chat")
async def chat(request: ChatRequest):
    print(f"[📨 Request] Language: {request.lang} | Message: {request.message}")

    # 🔒 Filter input: hanya pertanyaan yang sesuai keyword
    if not is_valid_intent(request.message):
        fallback_msg = {
            "de": "Ich kann nur Fragen zu Planville Dienstleistungen beantworten. "
                  "Bitte kontaktieren Sie uns direkt unter: https://planville.de/kontakt",
            "en": "I can only answer questions related to Planville services. "
                  "Please contact us directly here: https://planville.de/kontakt"
        }
        return {"reply": fallback_msg.get(request.lang, fallback_msg["de"])}

    try:
        # 🧠 Ambil konteks dari RAG index
        context_docs = query_index(request.message)

        # 🔄 Jika tidak ada hasil RAG, fallback ke hasil scraping
        if not context_docs:
            print("[⚠️] RAG kosong → menggunakan fallback scraper.")
            context_docs = get_scraped_context(request.message)

        # 🔗 Gabungkan semua dokumen hasil jadi konteks
        context_text = "\n".join(context_docs)

        # 📝 Bangun prompt untuk GPT
        prompt = f"""
Du bist ein professioneller Kundenservice-Assistent von Planville GmbH.
Antworte bitte höflich, direkt und hilfreich basierend auf dem folgenden Kontext.

🔎 Frage:
{request.message}

📄 Kontext:
{context_text}
"""

        # 🤖 Kirim ke OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )

        # ✅ Ambil jawaban
        reply_text = response.choices[0].message.content.strip()

        # 🔁 Jika kosong, fallback ke jawaban statis
        if not reply_text:
            fallback = (
                "Entschuldigung, ich habe leider keine passende Information zu Ihrer Anfrage.\n\n"
                "📞 Kontaktieren Sie unser Team direkt:\n"
                "👉 https://planville.de/kontakt"
            )
            return {"reply": fallback}

        return {"reply": reply_text}

    except Exception as e:
        print(f"[❌ GPT ERROR]: {e}")
        return {
            "reply": (
                "Es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut "
                "oder kontaktieren Sie uns direkt.\n\n➡️ https://planville.de/kontakt"
            )
        }

# ✅ Optional: Endpoint healthcheck
@app.get("/healthz")
def health_check():
    return {"status": "ok"}
