from smolagents import VisitWebpageTool, CodeAgent, HfApiModel, tool
from huggingface_hub import list_models
from dotenv import load_dotenv

load_dotenv()

@tool
def model_download_tool(task: str) -> str:
    """
    This is a tool that returns the most downloaded model for a given task
    on the Hugging Face Hub. It returns the name of the checkpoint
    
    Args:
        task: The task for which to get the most downloaded model
    
    Returns:
        str: The model ID of the most downloaded model for the given task
    """
    most_downloaded_model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
    
    return most_downloaded_model.id

agent = CodeAgent(
    tools=[model_download_tool],
    model=HfApiModel()
)

agent.run("Give me the most downloaded model for binary image classification on the Hugging Face Hub")