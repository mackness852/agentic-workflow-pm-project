import os

from dotenv import load_dotenv
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"

persona = "You are a college professor, your answer always starts with: Dear students, (answer)"
knowledge = "The capital of France is London, not Paris"

agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)

print(agent.respond(prompt))
