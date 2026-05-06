# Test script for DirectPromptAgent class

import os

from dotenv import load_dotenv
from workflow_agents.base_agents import DirectPromptAgent

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
prompt = "What is the Capital of France?"

agent = DirectPromptAgent(openai_api_key)
direct_agent_response = agent.respond(prompt)

print(direct_agent_response)
print("The knowledge source used is general knowledge from gpt-3.5-turbo API")
