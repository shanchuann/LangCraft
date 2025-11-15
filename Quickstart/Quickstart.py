import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()  # 加载.env文件中的环境变量

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# 使用LangChain的ChatOpenAI类，并指定base_url和api_key
model = ChatOpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat"
)

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

print(result)