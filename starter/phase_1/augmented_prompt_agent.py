import os

from dotenv import load_dotenv
from workflow_agents.base_agents import AugmentedPromptAgent

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
print(prompt)

persona = (
    "You are a college professor. Your answers always start with: 'Dear students,'"
    "For example: 'What is the capital of England?' 'Dear students, London'"
)

agent = AugmentedPromptAgent(openai_api_key, persona)

augmented_agent_response = agent.respond(prompt)
print(augmented_agent_response)

print(
    "Comments on this: The knowledge source used is general knowledge from gpt-3.5-turbo API. The system prompt is a persona that will craft any responses in the desired professor format"
)
# The knowledge source used is general knowledge from gpt-3.5-turbo API.
# The system prompt is a persona that will craft any responses in the desired
# professor format
