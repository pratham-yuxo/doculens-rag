import logging
import httpx
import json
from typing import List, Dict
from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
SUPPORTED_MODELS = {"fast": "llama-3.1-8b-instant", "quality": "llama-3.3-70b-versatile"}
OFFICIAL_REFUSAL_PHRASE = "I cannot answer from the uploaded documents."

_http_client = None

async def init_client():
    global _http_client
    _http_client = httpx.AsyncClient(timeout=120.0, limits=httpx.Limits(max_keepalive_connections=20))

async def close_client():
    if _http_client: await _http_client.aclose()

async def classify_intent(query: str) -> Dict[str, str]:
    """
    Classifies intent AND rewrites the query to fix typos/grammar.
    Returns: {"intent": "factual", "query": "corrected query"}
    """
    # Default fallback if API fails
    default_resp = {"intent": "factual", "query": query}
    
    if not settings.groq_api_key: return default_resp
    
    # NEW: Dynamically add Chart option if enabled
    chart_option = ""
    if settings.enable_charts:
        chart_option = "3. 'chart': Asking about visual data, graphs, trends, candles, RSI, MACD, or specific figures/tables mentioned in the text.\n"

    prompt = (
        "You are a query optimizer. Analyze the user's query.\n"
        "1. 'factual': Asking for specific definitions, general questions, data, numbers, dates, names, or specific concepts. (e.g. 'What is X?', 'Who is Y?', 'Revenue?', 'Define Z')\n"
        "2. 'analysis': Asking for broad summaries, overviews, ratings, tone analysis, or entire document synthesis. (e.g. 'Summarize this', 'Rate this', 'Main themes', 'What is in the table of contents?')\n"
        f"{chart_option}"
        "2. Rewrite the query to be clear, professional, and fix any typos/spacing errors (e.g. 'captheorem' -> 'CAP theorem').\n"
        "   - Keep it concise. Do not answer the question.\n\n"
        "Output MUST be a valid JSON object with keys: 'intent' and 'query'.\n"
        f"User Query: {query}"
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        resp = await _http_client.post(
            GROQ_CHAT_URL,
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            json={
                "model": SUPPORTED_MODELS["fast"], 
                "messages": messages, 
                "temperature": 0, 
                "max_tokens": 100,
                "response_format": {"type": "json_object"} # Force JSON mode
            }
        )
        resp.raise_for_status()
        data = resp.json()["choices"][0]["message"]["content"]
        
        # Parse JSON response
        parsed = json.loads(data)
        
        # Normalize intent string
        intent = parsed.get("intent", "factual").strip().lower()
        
        # Safety check: Ensure intent is valid
        valid_intents = ["factual", "analysis"]
        if settings.enable_charts:
            valid_intents.append("chart")
            
        if intent not in valid_intents:
            intent = "factual"

        # Get rewritten query (or fallback to original)
        rewritten = parsed.get("query", query).strip()
        
        return {"intent": intent, "query": rewritten}
        
    except Exception as e:
        logger.error(f"Router/Rewriter failed: {e}")
        return default_resp

def build_prompt(context: List[Dict], intent: str = "factual") -> str:
    """
    Constructs the system prompt with strict citation rules and structure awareness.
    """
    # DEBUG LOGGING: Check which charts are being sent to the LLM
    if intent == "chart":
        chart_chunks = [c for c in context if "CHART" in c.get("text", "") or "VISUAL" in c.get("text", "")]
        if chart_chunks:
            logger.info(f"📊 Sending {len(chart_chunks)} chart descriptions to LLM:")
            for i, c in enumerate(chart_chunks):
                logger.info(f"   Chart {i+1} (Page {c['metadata'].get('page_no')}): {c['text'][:100]}...")
        else:
            logger.warning("⚠️ Intent is 'chart' but NO chart chunks were retrieved from Vector DB!")

    # Build context string with explicit page markers
    ctx_str = ""
    for c in context:
        page_num = c['metadata'].get('page_no', 'Unknown')
        text = c['text']
        ctx_str += f"\n--- START SOURCE (Page {page_num}) ---\n{text}\n--- END SOURCE ---\n"
    
    base_rules = (
        "You are a corporate AI assistant. Your goal is to be helpful and grounded in the provided documents.\n"
        "IMPORTANT CITATION RULE: You must cite the source page for every key point using the format [Source: Page X].\n"
    )
    
    # MERGED LOGIC: Charts and Analysis
    if intent == "analysis" or intent == "chart":
        specific_instructions = (
            "Task: Provide a high-level analysis, summary, or rating based on the available chunks. "
            "Synthesize general themes. You may generalize if the text supports it.\n"
            "If the text describes a Chart (e.g. 'CHART VISUAL ANALYSIS'), explain the visual trends, indicators, and key values described.\n"
            "If the user asks about the 'List of Figures' or 'Table of Contents', look for list-like structures in the text and use them as a map.\n"
            "If the user asks to 'Rate' the document, evaluate the quality, depth, and clarity of the text provided in the chunks.\n"
            "CRITICAL SAFETY CHECK: If the chunks are gibberish or completely unrelated (e.g. user asks about 'cooking' but chunks are 'finance'), "
            f"respond ONLY with: '{OFFICIAL_REFUSAL_PHRASE}'."
        )
    else:
        # Strict rules for facts
        specific_instructions = (
            "Task: Answer the specific question using ONLY the provided context. "
            "Do not use outside knowledge. "
            "If the exact answer is not in the text, state 'Not found in document'. "
            "Do not guess numbers or names."
        )

    return f"{base_rules}\n{specific_instructions}\nContext:\n{ctx_str}"

async def call_groq(messages: List[Dict]) -> str:
    if not settings.groq_api_key: return "API Key Missing."
    try:
        resp = await _http_client.post(
            GROQ_CHAT_URL,
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            json={"model": SUPPORTED_MODELS["quality"], "messages": messages, "temperature": 0}
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        return "Error generating response."