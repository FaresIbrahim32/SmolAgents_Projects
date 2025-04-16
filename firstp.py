from smolagents import DuckDuckGoSearchTool, CodeAgent, HfApiModel
import os

# Get API token from environment variable
#hf_api_token = os.environ.get("HF_TOKEN")  # Your env variable name

# Initialize the HfApiModel correctly
model = HfApiModel(token='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')  # Use 'token' instead of 'api_key'

# Create the agent with the authenticated model
agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)

agent.run("How long does it take to drive from Tampa to New Jersey by car?")