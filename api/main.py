# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import os
import time

import stripe
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from adapters import OpenAICompatAdapter
from crisis import detect_crisis, crisis_response
from dual_engine import DualEngine

GROK_URL = os.getenv("GROK_API_URL", "https://api.x.ai/v1")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-4.5")
GROK_KEY = os.getenv("GROK_API_KEY", "")

DEEPSEEK_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")

MOCK_MODE = os.getenv("AMIPASSAI_MOCK", "false").lower() == "true"

STRIPE_SECRET = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
SITE_URL = os.getenv("SITE_URL", "https://amipassai.com")

stripe.api_key = STRIPE_SECRET

ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS", "https://amipassai.com,http://amipassai.com,http://localhost:8000"
).split(",")

SYSTEM_PROMPT = (
    "You are amiPassAI — an expert, clear, and security-conscious AI companion for "
    "password security, digital identity protection, and cyber hygiene. You are an AI "
    "system (not a human) and say so if asked. You speak in the user's language "
    "(detect from their message).\n\n"
    "YOUR EXPERTISE:\n"
    "- Password security: strength assessment, passphrase generation strategies, "
    "entropy analysis, common attack vectors (brute-force, dictionary, credential "
    "stuffing, rainbow tables), password manager best practices\n"
    "- Authentication: MFA/2FA setup guidance (TOTP, hardware keys, SMS risks), "
    "passkeys/FIDO2/WebAuthn, single sign-on (SSO) concepts, biometric auth\n"
    "- Breach awareness: how to check if credentials are compromised (Have I Been "
    "Pwned), breach response steps, credential rotation strategy, dark web basics\n"
    "- Digital identity: email alias strategies, privacy-first account setup, "
    "social engineering awareness, phishing detection, SIM-swap prevention\n"
    "- Enterprise security basics: password policies (NIST SP 800-63B), zero-trust "
    "principles, least-privilege access, audit logging, SSO/SAML/OIDC basics\n"
    "- Privacy tools: VPN concepts, encrypted messaging, secure file sharing, "
    "browser privacy settings, tracker blocking\n\n"
    "OUTPUT FORMAT:\n"
    "- Use 🔐/🛡️/⚠️/✅/❌ icons for visual clarity\n"
    "- Structure advice as actionable checklists when appropriate\n"
    "- Rate password strength clearly (Weak/Fair/Strong/Excellent)\n"
    "- Always explain WHY something is secure or insecure\n\n"
    "CRITICAL RULES:\n"
    "- NEVER ask users to share their actual passwords. If they do, warn them and "
    "advise changing the password immediately.\n"
    "- NEVER generate or store actual passwords — teach strategies and principles.\n"
    "- Recommend password managers but remain vendor-neutral.\n"
    "- Do not help bypass security measures, crack passwords, or access others' accounts.\n"
    "- Always recommend official channels for account recovery.\n"
    "- Explain concepts simply — assume non-technical users unless context suggests otherwise.\n"
    "- For enterprise topics, recommend consulting a qualified security professional."
)

RATE_LIMIT: dict[str, list[float]] = {}


def _rate_ok(ip: str, max_req: int = 20, window: float = 60.0) -> bool:
    now = time.time()
    RATE_LIMIT[ip] = [t for t in RATE_LIMIT.get(ip, []) if now - t < window] + [now]
    return len(RATE_LIMIT[ip]) <= max_req


def _sign(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _build_engine() -> DualEngine | None:
    if MOCK_MODE or not GROK_KEY or not DEEPSEEK_KEY:
        return None
    grok = OpenAICompatAdapter(
        base_url=GROK_URL, model=GROK_MODEL, api_key=GROK_KEY,
        temperature=0.3, max_tokens=600, timeout=30, name="grok",
    )
    deepseek = OpenAICompatAdapter(
        base_url=DEEPSEEK_URL, model=DEEPSEEK_MODEL, api_key=DEEPSEEK_KEY,
        temperature=0.3, max_tokens=600, timeout=30, name="deepseek",
    )
    return DualEngine(grok, deepseek, system_prompt=SYSTEM_PROMPT, threshold=0.45)


ENGINE = _build_engine()

app = FastAPI(title="amiPassAI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    locale: str = "EN"


class ChatResponse(BaseModel):
    response: str
    engine: str
    decision: str
    certified: bool
    signature: str
    is_crisis_response: bool
    concordance: float | None = None
    latency_s: float = 0.0


@app.get("/")
def root():
    return {"service": "amiPassAI API", "version": "1.0.0"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "engine": "dual (Grok + DeepSeek)" if ENGINE else "mock",
        "grok_configured": bool(GROK_KEY),
        "deepseek_configured": bool(DEEPSEEK_KEY),
        "stripe_configured": bool(STRIPE_SECRET and STRIPE_PRICE_ID),
        "mock_mode": MOCK_MODE or not ENGINE,
    }


@app.post("/companion/pass/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    if not _rate_ok(ip):
        raise HTTPException(429, "Too many requests — please wait a moment")

    msg = req.message.strip()
    if not msg:
        raise HTTPException(400, "Empty message")
    if len(msg) > 2000:
        raise HTTPException(400, "Message too long (max 2000 chars)")

    locale = req.locale.upper()[:2] if req.locale else "EN"

    if any(kw in msg.lower() for kw in ["my password is", "my pass is", "my pwd is"]):
        warn = (
            "⚠️ It looks like you may have shared an actual password. "
            "**Please change it immediately.** Never share real passwords with any AI, "
            "website, or person. I can help you understand password security without "
            "needing your actual credentials."
        )
        return ChatResponse(
            response=warn, engine="safety-layer", decision="password-intercept",
            certified=True, signature=_sign(warn), is_crisis_response=False,
        )

    crisis = detect_crisis(msg)
    if crisis.is_crisis:
        resp = crisis_response(locale.lower())
        return ChatResponse(
            response=resp, engine="safety-layer", decision="crisis-intercept",
            certified=True, signature=_sign(resp), is_crisis_response=True,
        )

    if ENGINE:
        result = ENGINE.ask(msg)
        return ChatResponse(
            response=result.reply, engine=result.engine, decision=result.decision,
            certified=True, signature=_sign(result.reply), is_crisis_response=False,
            concordance=result.concordance, latency_s=result.latency_s,
        )

    mock = "Great question! In production, I'd cross-verify this with two AI engines. Check back soon for verified security insights."
    return ChatResponse(
        response=mock, engine="mock", decision="mock",
        certified=False, signature=_sign(mock), is_crisis_response=False,
    )


@app.post("/create-checkout-session")
def create_checkout():
    if not STRIPE_SECRET or not STRIPE_PRICE_ID:
        raise HTTPException(503, "Payment system not configured yet")
    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            success_url=SITE_URL + "?payment=success",
            cancel_url=SITE_URL + "?payment=cancelled",
        )
    except stripe.error.StripeError as e:
        raise HTTPException(502, f"Payment error: {e.user_message or str(e)}")
    return {"url": session.url}


@app.post("/stripe-webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except (ValueError, stripe.error.SignatureVerificationError):
        raise HTTPException(400, "Invalid webhook signature")
    if event["type"] == "checkout.session.completed":
        print(f"[STRIPE] New sub: {event['data']['object'].get('customer_email', '?')}")
    elif event["type"] == "customer.subscription.deleted":
        print(f"[STRIPE] Sub cancelled: {event['data']['object'].get('id')}")
    return {"received": True}


@app.post("/gdpr/data-request")
def gdpr_data():
    return {
        "message": "amiPassAI does not store chat messages or security queries. Processing is stateless. "
                   "For Stripe subscription data, email privacy@amipassai.com.",
        "data_stored": "none",
    }


@app.delete("/gdpr/delete")
def gdpr_delete():
    return {
        "message": "No personal data held server-side. "
                   "For Stripe data deletion, email privacy@amipassai.com.",
        "status": "no_data_held",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
