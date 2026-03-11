import os
import logging
from typing import Callable, Dict
from enum import Enum
from dataclasses import dataclass
from dotenv import load_dotenv
import ollama
from ollama import Client
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
@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog.

    Args:
        product: The name of the product to look up.

    Returns:
        float: The price of the product. Returns 0.0 if not found.
    """
    logger.info(f"Looking up price for product: {product}")
    
    price = PRODUCT_CATALOG.get(product.lower(), 0.0)
    if price == 0.0:
        logger.warning(f"Product '{product}' not found in catalog.")
    else:
        logger.info(f"Found price for '{product}': ${price}")
    return price

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price based on the discount tier.

    Args:
        price: The original price of the product.
        discount_tier: The tier of discount to apply (bronze, silver, or gold).

    Returns:
        float: The discounted price.
    """
    # Convert string to Enum if needed
    if isinstance(discount_tier, str):
        try:
            tier = DiscountTier(discount_tier.lower())
        except ValueError:
            logger.error(f"Invalid discount tier: {discount_tier}")
            raise ValueError(f"Invalid discount tier: {discount_tier}. Must be one of: {[t.value for t in DiscountTier]}")
    else:
        tier = discount_tier

    logger.info(f"Applying discount tier '{tier.value}' to price ${price:.2f}")
    
    discounts = {
        DiscountTier.BRONZE: 0.05,
        DiscountTier.SILVER: 0.10,
        DiscountTier.GOLD: 0.20,
    }
    
    discount_percent = discounts.get(tier, 0.0)
    
    discounted_price = price * (1 - discount_percent)
    logger.info(f"New price after {tier.value} discount: ${discounted_price:.2f}")
    return discounted_price

llm_tools_schema = [
    {
      "type": "function",
      "function": {
        "name": "get_product_price",
        "description": "Look up the price of a product in the catalog",
        "parameters": {
          "type": "object",
          "required": ["product"],
          "properties": {
            "product": {
              "type": "string",
              "description": "The name of the product to look up"
            }
          }
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "apply_discount",
        "description": "Apply a discount to a price based on the discount tier (bronze, silver, gold)",
        "parameters": {
          "type": "object",
          "required": ["price", "discount_tier"],
          "properties": {
            "price": {
              "type": "number",
              "description": "The original price of the product"
            },
            "discount_tier": {
              "type": "string",
              "enum": ["bronze", "silver", "gold"],
              "description": "The discount tier to apply"
            }
          }
        }
      }
    }
  ]


def get_llm_tools()->dict[str,Callable]:
    return {
        "get_product_price":get_product_price,
        "apply_discount":apply_discount
    }

def init_llm_model(config:AppConfig):
    return ollama.Client(host=f"http://{config.server_host}:{config.server_port}")

@traceable(name="Ollama Chat",run_type="llm")
def ollama_chat_traced(config:AppConfig,ollama_client:Client,messages:list[dict]):
    return ollama_client.chat(model=config.model_type,tools=llm_tools_schema,messages=messages)

SYSTEM_PROMPT = {"role":"system","content":"""You are a helpful shopping assistant. Your goal is to provide accurate pricing information using the available tools.

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
"""}

@traceable(name="Ollama SDK agent loop")
def run_agent_loop(query_text: str):
    config = AppConfig()
    tools_map = get_llm_tools()
    ollama_client = init_llm_model(config)

    messages = [
        SYSTEM_PROMPT,
        {"role":"user","content":query_text}
    ]
    
    for i in range(config.max_iterations):
        logger.info(f"Iteration {i+1}/{config.max_iterations}")
        response = ollama_chat_traced(config,ollama_client,messages)
        message = response.get("message", {})
        messages.append(message)
        
        tool_calls = message.get("tool_calls", [])
        if not tool_calls:
            return message.get("content")
            
        for tool_call in tool_calls:
            function_info = tool_call.get("function", {})
            tool_name = function_info.get("name")
            tool_args = function_info.get("arguments", {})
            
            logger.info(f"Calling tool: {tool_name} with {tool_args}")
            
            if tool_name not in tools_map:
                result = f"Error: Tool {tool_name} not found."
            else:
                try:
                    result = tools_map[tool_name](**tool_args)
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {e}")
                    result = f"Error: {str(e)}. Please use only allowed values for tool arguments."
                
            messages.append({
                "role": "tool",
                "content": str(result),
                "name": tool_name
            })
    
    return "Reached maximum iterations without a final answer."

if __name__=="__main__":
    logger.info("Agent loop has been started")
    
    query = "What is the price of the monitor after a gold discount?"
    logger.info(f"User Query: {query}")
    
    result = run_agent_loop(query)
    print(f"\nFinal Result: {result}")