import json
import os
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
from typing import Optional

from .cache import cache
from .agents.classifier import QueryClassifier
from .agents.guard import GuardAgent
from .agents.product import ProductAgent
from .agents.troubleshoot import TroubleshootAgent
from .agents.response import ResponseAgent
from .vector_db import ChromaVectorDB

# Load environment variables (optional, kept for parity with design doc)
load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PRODUCTS = None
TROUBLESHOOTING = None
vector_db: Optional[ChromaVectorDB] = None
classifier: Optional[QueryClassifier] = None
guard: Optional[GuardAgent] = None
product_agent: Optional[ProductAgent] = None
troubleshoot_agent: Optional[TroubleshootAgent] = None
response_agent: Optional[ResponseAgent] = None

app = FastAPI()
logger = logging.getLogger("app")
logging.basicConfig(level=logging.INFO)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    global PRODUCTS, TROUBLESHOOTING, vector_db, classifier, guard, product_agent, troubleshoot_agent, response_agent
    # Load data
    with open(os.path.join(DATA_DIR, "products.json"), "r") as f:
        PRODUCTS = json.load(f)
    with open(os.path.join(DATA_DIR, "troubleshooting.json"), "r") as f:
        TROUBLESHOOTING = json.load(f)

    vector_db = ChromaVectorDB()
    vector_db.add_products(PRODUCTS)
    vector_db.add_troubleshooting(TROUBLESHOOTING)

    classifier = QueryClassifier()
    guard = GuardAgent()
    product_agent = ProductAgent(PRODUCTS, vector_db=vector_db)
    troubleshoot_agent = TroubleshootAgent(TROUBLESHOOTING, vector_db=vector_db)
    response_agent = ResponseAgent()


class ChatRequest(BaseModel):
    message: str
    session_id: str


class ChatResponse(BaseModel):
    response: str
    products: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    steps: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    cached: bool = False


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if not all([classifier, guard, product_agent, troubleshoot_agent, response_agent]):
            raise HTTPException(status_code=503, detail="Service starting up, please retry shortly.")

        user_message = request.message.strip()
        history = cache.get_chat_history(request.session_id) or []

        cached_response = cache.get_cached_response(user_message)
        if cached_response:
            logger.info(f"⚡ Returning cached response for: {user_message[:50]}...")
            cache.add_message(request.session_id, "user", user_message)
            cache.add_message(request.session_id, "assistant", cached_response.get("response", ""))
            return ChatResponse(**cached_response, cached=True)

        # Fail-fast guard 
        in_scope = await guard.check_scope(user_message, context=None)
        if not in_scope:
            out_of_scope_response = {
                "response": "I can only assist with Refrigerator and Dishwasher parts. Please ask me about those appliances!",
                "products": [],
                "steps": []
            }

            # Add to history
            cache.add_message(request.session_id, "user", user_message)
            cache.add_message(request.session_id, "assistant", out_of_scope_response["response"])
            cache.set_cached_response(user_message, out_of_scope_response)
            
            return ChatResponse(**out_of_scope_response, cached=False)

        classification = await classifier.classify(user_message, history)
        entities = classification["entities"]
    
        # Route: troubleshooting vs product info
        if classification["intent"] == "troubleshooting":
            logger.info(f"Troubleshooting request classified with entities: {entities}")
            # Build response directly in troubleshoot agent (skip ResponseAgent)
            final_struct = await troubleshoot_agent.diagnose(entities, user_message)
        else:
            logger.info(f"Product info request classified with entities: {entities}")
            response_data = await product_agent.get_info(entities, user_message)
            # Final LLM response synthesis for product flow
            final_struct = await response_agent.generate(
                user_message=user_message,
                intent=classification["intent"],
                context=response_data,
                history=history,
            )

        # Save conversation
        cache.add_message(request.session_id, "user", user_message)
        cache.add_message(request.session_id, "assistant", final_struct.get("message", final_struct.get("response", "")))
        cache.set_cached_response(user_message, final_struct)

        return ChatResponse(
            response=final_struct.get("message") or final_struct.get("response", "I can help you with that."),
            products=final_struct.get("products", []),
            steps=final_struct.get("steps", []),
            metadata=final_struct.get("metadata", {}),
            cached=False
        )
        
    except Exception as e:
        logger.exception("Chat handler failed", exc_info=e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
