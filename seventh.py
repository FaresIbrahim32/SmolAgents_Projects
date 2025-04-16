from smolagents import CodeAgent, HfApiModel, Tool
from dotenv import load_dotenv

load_dotenv()

travel_duration_tool = Tool.from_space(
    "m-ric/get-travel-duration-tool",
    name="get_travel_duration_tool",
    description="Get travel duration between 2 locations"
)

model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct")

agent = CodeAgent(tools=[travel_duration_tool], model=model)

response = travel_duration_tool("Tampa","New Jersey","driving")
print(response)