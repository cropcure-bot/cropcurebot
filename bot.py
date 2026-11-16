import os
import base64
import json
import logging
import time
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

# ----------------------------------------------------------------------
# 1. Credentials
# ----------------------------------------------------------------------
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")

# Optional whitelist: comma-separated Telegram user IDs.
# Leave empty ("") to allow everyone. For a deployed bot, set this.
# Example: ALLOWED_USER_IDS=123456789,987654321
ALLOWED_USER_IDS = [
    int(x) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip().isdigit()
]

LATEST_FILE = "latest.json"

# ----------------------------------------------------------------------
# 2. Logging
# ----------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# ----------------------------------------------------------------------
# 3. Free vision models (verified working on OpenRouter free tier)
# ----------------------------------------------------------------------
MODELS = [
    "nvidia/nemotron-3.5-super-vl:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "dots-studio/dots-3-note-preview:free",
    "openrouter/free",
]

PROMPT = """You are an expert plant pathologist AI. Analyze the image.

STEP 1 — VALIDATION:
If the image is NOT a plant leaf, reply exactly:
NOT_A_LEAF

STEP 2 — DIAGNOSIS (only if it IS a leaf):
Return ONLY a JSON object with these keys:
- "disease": common name (e.g. "Early Blight", "Healthy")
- "confidence": integer 0-100
- "infection_percentage": integer 0-100
- "cure": short practical treatment, max 15 words

Return ONLY the JSON, no other text."""


# ----------------------------------------------------------------------
# 4. OpenRouter call with 429 retry
# ----------------------------------------------------------------------
def call_openrouter(image_b64: str) -> str:
    last_error = None
    for model in MODELS:
        for attempt in range(3):
            try:
                r = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": PROMPT},
                                {"type": "image_url",
                                 "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                            ],
                        }],
                    },
                    timeout=90,
                )
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"].strip()
                logging.info(f"Model {model} replied: {content[:120]}")
                return content

            except requests.exceptions.HTTPError as e:
                last_error = e
                if e.response.status_code == 429:
                    wait = 5 * (attempt + 1)
                    logging.warning(f"Model {model} 429. Retry in {wait}s...")
                    time.sleep(wait)
                    continue
                logging.warning(f"Model {model} failed {e.response.status_code}. Next model.")
                break

            except Exception as e:
                last_error = e
                logging.warning(f"Model {model} failed: {e}")
                break

    try:
        logging.info("Trying openrouter/free auto-router...")
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
            json={
                "model": "openrouter/free",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ],
                }],
            },
            timeout=90,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        last_error = e

    raise RuntimeError(f"All models failed. Last error: {last_error}")


# ----------------------------------------------------------------------
# 5. Save last diagnosis for the Pico W to fetch later
# ----------------------------------------------------------------------
def save_latest(data: dict):
    try:
        with open(LATEST_FILE, "w") as f:
            json.dump(data, f)
        logging.info(f"Saved to {LATEST_FILE}: {data}")
    except Exception as e:
        logging.warning(f"Could not save {LATEST_FILE}: {e}")


# ----------------------------------------------------------------------
# 6. Telegram photo handler
# ----------------------------------------------------------------------
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if ALLOWED_USER_IDS and update.effective_user.id not in ALLOWED_USER_IDS:
        await update.message.reply_text("⛔ Not authorized.")
        return

    await update.message.reply_text("🔍 Analyzing your leaf…")

    photo_file = await update.message.photo[-1].get_file()
    local_path = f"leaf_{update.message.message_id}.jpg"
    await photo_file.download_to_drive(local_path)

    try:
        with open(local_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        raw = call_openrouter(image_b64)

        if "NOT_A_LEAF" in raw.upper():
            await update.message.reply_text(
                "❌ That doesn't look like a plant leaf.\n"
                "Please send a clear, close-up photo of a single leaf."
            )
            return

        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()

        data = json.loads(raw)
        data["timestamp"] = time.time()
        save_latest(data)

        await update.message.reply_text(
            f"🍃 *Diagnosis:* {data['disease']}\n"
            f"📊 *Confidence:* {data['confidence']}%\n"
            f"🟠 *Infection:* ~{data['infection_percentage']}%\n"
            f"💊 *Cure:* {data['cure']}",
            parse_mode="Markdown",
        )

    except json.JSONDecodeError:
        await update.message.reply_text(
            "⚠️ The AI replied in an unexpected format. Try again with a clearer photo."
        )
    except Exception as e:
        logging.error(f"Error: {e}")
        await update.message.reply_text("⚠️ Something went wrong. Try again in a moment.")
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)


# ----------------------------------------------------------------------
# 7. Entry point
# ----------------------------------------------------------------------
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    logging.info("Bot running. Send a leaf photo on Telegram.")
    app.run_polling()


if __name__ == "__main__":
    main()