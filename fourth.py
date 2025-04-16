from smolagents import VisitWebpageTool, CodeAgent, HfApiModel, tool, Tool
from huggingface_hub import list_models
from huggingface_hub import InferenceClient
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

class TexttoImageTool(Tool):
    def __init__(self):
        self.name = "image_generator"  # Setting name in __init__
        self.description = 'This tool creates an image according to a prompt, which is a text description'
        self.inputs = {
            "prompt": {
                "type": "string",
                "description": "The image generation prompt"
            },
            "model": {
                "type": "string",
                "description": "The Hugging Face Model ID used for image generation",
                "nullable": True
            }
        }
        self.output_type = "image"
        self.current_model = "black-forest-labs/FLUX.1-schnell"
        self.client = None
        super().__init__()  # Call super().__init__() after setting attributes
    
    def forward(self, prompt, model=None):
        if model:
            if model != self.current_model:
                self.current_model = model
                self.client = InferenceClient(model)
        if not self.client:
            self.client = InferenceClient(self.current_model)
            
        image = self.client.text_to_image(prompt)
        image.save("Saved_generated_image.png")
        
        return f"Successfully saved image with this prompt: {prompt} using model: {self.current_model}"
    
imageGenerator = TexttoImageTool()

agent = CodeAgent(tools=[model_download_tool, imageGenerator], model=HfApiModel())
agent.run("Improve this prompt then generate an image for it. Prompt: A chess board. Get the latest model for text-to-image from the Hugging Face Hub")