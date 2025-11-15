from dataclasses import dataclass
import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
load_dotenv()  # 加载.env文件中的环境变量

# Define system prompt
SYSTEM_PROMPT = """你是一个用双关语说话的专家天气预报员。

你有两个工具：

- get_weather_for_location：使用它来获取特定位置的天气
- get_user_location：使用它来获取用户的位置

如果用户问你天气，确保你知道位置。如果你能从问题中分辨出他们的意思是无论他们在哪里，使用get_user_location工具找到他们的位置。
"""

# Define context schema
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

# Define tools
@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "西安" if user_id == "1" else "SF"

# Configure model
model = ChatOpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    temperature=0.5,
    timeout=10,
    max_tokens=1000
)

# Define response format
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None

# Set up memory
checkpointer = InMemorySaver()

# Create agent
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

# Run agent
# `thread_id` is a unique identifier for a given conversation.
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#   punny_response="Well, it looks like the weather in Xi'an is absolutely radiant! You could say it's always sunny in Xi'an - talk about a bright outlook! Don't forget your sunglasses, because this forecast is looking positively glowing!",
#   weather_conditions="It's always sunny in 西安!"
# )



# Note that we can continue the conversation using the same `thread_id`.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#   punny_response="You're very welcome! I'm always happy to brighten your day with a sunny forecast. If you need any more weather wisdom, just let me know - I'm here to help you weather any situation!",
#   weather_conditions=None
# )