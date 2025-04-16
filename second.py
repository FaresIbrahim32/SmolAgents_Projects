from smolagents import VisitWebpageTool, CodeAgent, HfApiModel
from dotenv import load_dotenv
from e2b import Sandbox
import os
import logging

# Set up proper logging with level parameter
logger = logging.getLogger("smolagents")
logger.setLevel(logging.INFO)

# Create a handler and formatter
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

load_dotenv()

# Create the agent with a standard executor
agent = CodeAgent(
    tools=[VisitWebpageTool()],
    model=HfApiModel(token='ENCRYPTED'),
    additional_authorized_imports=["requests", "markdownify"],
    executor_type="e2b"
)

agent.run("Where was Abraham Lincoln killed?")