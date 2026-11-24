from fastapi import APIRouter
from ..models.schemas import ChatRequest, ChatAnswer, RetrievedChunk
from ..services.retrieval import retrieve
from ..services.llm import call_groq, build_prompt, classify_intent
from ..config import get_settings
import logging
import asyncio # Re-added for the robust retry wrapper if implemented elsewhere

# Safe import for modularity
try:
    # Note: Using app.core.retriever in the previous working version, 
    # but respecting the user's specific import path here:
    from app.charts.retriever import retrieve_charts
except ImportError:
    # Fallback mock if import fails
    async def retrieve_charts(q, k): return []

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)

# NOTE: Keeping the direct call_groq usage as per the provided file, 
# but ensuring an asyncio import is present for future features (like retries).
# If call_groq_with_retry exists, you should use that instead of call_groq here.

@router.post("/query", response_model=ChatAnswer)
async def query_chat(payload: ChatRequest):
    logger.info(f"🚀 [ChatAPI] Received Query: {payload.question}")

    # 1. INTELLIGENCE: Classify & Rewrite
    router_resp = await classify_intent(payload.question)
    intent = router_resp["intent"]
    query = router_resp["query"]
    
    # --- CRITICAL FIX: Structure / Meta-Questions ---
    structure_keywords = ["list of figures", "list of tables", "table of contents", "index", "chapters", "outline", "what page", "on which page"]
    if any(k in payload.question.lower() for k in structure_keywords):
        logger.info(f"📍 Structure Query Detected: '{payload.question}' -> Forcing Analysis Mode")
        intent = "analysis"

    logger.info(f"🧠 Final Intent: {intent} | Optimized Query: '{query}'")
    
    # 2. RETRIEVAL STRATEGY (Hybrid: Text + Visual)
    results_text = []
    results_charts = []

    # A. Always fetch Text Context (Foundation)
    logger.info("📚 [Step A] Fetching Text Context...")
    # Using top_k from payload for flexibility, defaulting to 4 if not provided
    text_k = getattr(payload, 'top_k', 4) 
    results_text = await retrieve(query, top_k=text_k)
    logger.info(f"   -> Found {len(results_text)} text chunks.")

    # B. Fetch Chart Context (Visuals)
    visual_intents = ["chart", "analysis", "explain", "query", "details"]
    
    if intent in visual_intents and settings.enable_charts:
        logger.info("🎨 [Step B] Intent warrants Chart Retrieval. Executing...")
        try:
            results_charts = await retrieve_charts(query, top_k=3)
            logger.info(f"   -> Found {len(results_charts)} chart candidates.")
            
            # CRITICAL TRANSFORMATION:
            for chart in results_charts:
                meta = chart.get('metadata', {})
                visual_desc = meta.get('visual_summary') or chart.get('text') or "No visual description available."
                src = meta.get('source', 'Unknown')
                page = meta.get('page_no', '?')
                
                # Overwrite 'text' with a tagged format the LLM can understand
                chart['text'] = (
                    f"--- [VISUAL DATA] ---\n"
                    f"Document: {src} (Page {page})\n" # Explicitly use 'Document' here
                    f"Description: {visual_desc}\n"
                    f"---------------------"
                )
                if 'filename' not in meta: meta['filename'] = src
                
        except Exception as e:
            logger.error(f"❌ Chart Retrieval Failed: {e}")
            results_charts = []
    else:
        logger.info(f"⏭️ [Step B] Skipping Chart Retrieval (Intent: {intent})")

    # 3. MERGE RESULTS
    results = results_text + results_charts
    logger.info(f"🔗 Merged Results: {len(results)} total chunks ({len(results_text)} Text + {len(results_charts)} Charts)")
    
    # --- NEW STEP: UNIVERSAL CONTEXT FORMATTING FOR UNAMBIGUOUS CITATION ---
    # This step ensures both text and chart results are wrapped in a standard tag 
    # for the LLM to correctly identify the source document.
    formatted_context = []
    for chunk in results:
        meta = chunk.get('metadata', {})
        # Use 'filename' or 'source' as the document name
        src = meta.get('filename') or meta.get('source', 'Unknown Document')
        page = meta.get('page_no', '?')
        
        # Check if the chunk is not already tagged as chart data
        if not chunk['text'].startswith("--- [VISUAL DATA] ---"):
            content = chunk['text']
            
            # Reformat the chunk text to be explicit about its source
            chunk['text'] = (
                f"--- [SOURCE TEXT] ---\n"
                f"Document: {src} (Page {page})\n"
                f"Content: {content}\n"
                f"---------------------"
            )
        
        # If it is chart data, it already has the 'Document' tag from Step 2.B

        formatted_context.append(chunk)

    results = formatted_context 

    # 4. GATEKEEPER LOGIC
    best_score = results[0]['rerank_score'] if results else 0.0
    should_refuse = False
    refusal_reason = ""

    if not results:
        should_refuse = True
        refusal_reason = "No documents found in Vector DB."
    
    # Rule: Factual (Strict)
    elif intent == "factual":
        if best_score < settings.threshold_factual:
            should_refuse = True
            refusal_reason = f"Score {best_score:.4f} < {settings.threshold_factual} (Strict)"
            
    # Rule: Analysis (Bypass)
    elif intent in ["analysis", "chart", "explain"]:
        should_refuse = False 

    if should_refuse:
        logger.info(f"⛔ Refused. {refusal_reason}")
        return ChatAnswer(
            answer="I cannot find relevant information in the uploaded documents to answer this specific query.",
            sources=[], 
            refused=True, 
            reasoning=refusal_reason,
            intent=intent
        )

    # 5. GENERATION
    sys_prompt = build_prompt(results, intent=intent)
    
    # --- NEW: EXPLICIT CITATION RULE FOR UNAMBIGUOUS OUTPUT ---
    sys_prompt += (
        "\n\nCITATION RULE: When providing an in-line citation, you MUST include "
        "the Document name and Page Number from the 'Document: X (Page Y)' tag found in the context. "
        "Format citation like this: [DocumentName, Page Y] (e.g., [AnnualReport_2024.pdf, Page 125])."
    )

    # Inject an explicit instruction about visual data if charts are present
    if results_charts:
        sys_prompt += "\n\nNOTE: The context includes [VISUAL DATA] blocks. These are AI-generated descriptions of charts/figures. Use them to answer questions about visual trends, lines, or specific data points."

    msgs = [{"role": "system", "content": sys_prompt}] + \
           [{"role": m.role, "content": m.content} for m in payload.history[-2:]] + \
           [{"role": "user", "content": payload.question}]
           
    logger.info("📝 Sending Prompt to Groq...")
    try:
        ans = await call_groq(msgs)
    except Exception as e:
        logger.error(f"❌ Groq Call Failed: {e}")
        ans = "I encountered an error while generating the response."

    # 6. SOURCE ATTRIBUTION
    sources = []
    for r in results:
        meta = r.get('metadata', {})
        # Handle 'page_no' being a string or int
        try:
            pg = int(meta.get('page_no', 0))
        except:
            pg = 0
            
        sources.append(RetrievedChunk(
            score=r.get('rerank_score', 0.0), 
            filename=meta.get('filename', 'unknown'), 
            page_no=pg, 
            chunk_id=0, 
            text_snippet=r['text'][:150] # Snippet for UI
        ))
    
    return ChatAnswer(
        answer=ans, 
        sources=sources, 
        reasoning=f"Mode: {intent} | Used: '{query}' | Combined Context: {len(results)} items",
        intent=intent
    )