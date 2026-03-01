import os
import logging
from typing import Dict
from enum import Enum
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool,BaseTool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

@dataclass(frozen=True)
class AppConfig:
    model_type: str = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    server_host: str = os.getenv("OLLAMA_HOST", "127.0.0.1")
    server_port: int = int(os.getenv("OLLAMA_PORT", "11434"))
    max_iterations: int = 10

class DiscountTier(str, Enum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"

# Mock product catalog
PRODUCT_CATALOG: Dict[str, float] = {
    "laptop": 1200.00,
    "smartphone": 800.00,
    "headphones": 150.00,
    "monitor": 300.00,
    "keyboard": 50.00,
}

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog"""
    logger.info(f"Looking up price for product: {product}")
    
    price = PRODUCT_CATALOG.get(product.lower(), 0.0)
    if price == 0.0:
        logger.warning(f"Product '{product}' not found in catalog.")
    else:
        logger.info(f"Found price for '{product}': ${price}")
    return price

@tool
def apply_discount(price: float, discount_tier: DiscountTier) -> float:
    """Apply a discount to a price based on the discount tier.
    
    Tiers:
    - 'bronze': 5% discount
    - 'silver': 10% discount
    - 'gold': 20% discount
    """
    logger.info(f"Applying discount tier '{discount_tier.value}' to price ${price:.2f}")
    
    discounts = {
        DiscountTier.BRONZE: 0.05,
        DiscountTier.SILVER: 0.10,
        DiscountTier.GOLD: 0.20,
    }
    
    discount_percent = discounts.get(discount_tier, 0.0)
    
    discounted_price = price * (1 - discount_percent)
    logger.info(f"New price after {discount_tier.value} discount: ${discounted_price:.2f}")
    return discounted_price

def get_llm_tools()->list[BaseTool]:
    return [apply_discount, get_product_price]

def init_llm_model():
    config = AppConfig()
    llm = init_chat_model(
        model=config.model_type,
        model_provider="ollama",
        base_url=f"http://{config.server_host}:{config.server_port}"
    )
    return llm.bind_tools(get_llm_tools())

SYSTEM_PROMPT = """You are a helpful shopping assistant. Your goal is to provide accurate pricing information using the available tools.

### GUIDELINES & SAFETY:
1. **Tool Sequencing**: Always use `get_product_price` FIRST to find the base price before attempting to use `apply_discount`.
2. **Data Integrity**: Only use prices returned by the `get_product_price` tool. Do not hallucinate or guess prices.
3. **Discount Validation**: Only apply discounts if a valid `DiscountTier` (bronze, silver, gold) is specified or implied. If the user asks for a discount not in the list, inform them of the available tiers.
4. **Missing Information**: If a product is not found (price returns 0.0), do not attempt to apply a discount. Inform the user the product is unavailable.
5. **Reasoning**: Before calling a tool, briefly state your reasoning for why that tool is needed.

### EXECUTION FLOW:
- Step 1: Identify the product and discount tier from the user query.
- Step 2: Call `get_product_price` for the identified product.
- Step 3: If the price is > 0, call `apply_discount` with that price and the tier.
- Step 4: Provide the final answer to the user.
"""

@traceable(name="LangChain agent loop")
def run_agent_loop(query: str):
    llm_with_tools = init_llm_model()
    tools = get_llm_tools()
    tools_map = {t.name: t for t in tools}
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query)
    ]
    
    config = AppConfig()
    
    for i in range(config.max_iterations):
        logger.info(f"Iteration {i+1}/{config.max_iterations}")
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        if not response.tool_calls:
            return response.content
            
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if tool_name not in tools_map:
                result = f"Error: Tool {tool_name} not found."
            else:
                try:
                    result = tools_map[tool_name].invoke(tool_args)
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {e}")
                    result = f"Error: {str(e)}. Please use only allowed values for tool arguments."
                
            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            ))
    
    return "Reached maximum iterations without a final answer."

if __name__=="__main__":
    logger.info("Agent loop has been started")
    
    query = "What is the price of the monitor after a diamond discount?"
    logger.info(f"User Query: {query}")
    
    result = run_agent_loop(query)
    print(f"\nFinal Result: {result}")