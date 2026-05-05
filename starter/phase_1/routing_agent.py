import os

from dotenv import load_dotenv
from workflow_agents.base_agents import (
    KnowledgeAugmentedPromptAgent,
    RoutingAgent,
)

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

persona = "You are a college professor"
knowledge = "You know everything about Texas"
texas_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)

knowledge = "You know everything about Europe"
europe_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key, persona, knowledge
)


persona = "You are a college math professor"
knowledge = "You know everything about math, you take prompts with numbers, extract math formulas, and show the answer without explanation"
math_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)

routing_agent = RoutingAgent(openai_api_key, [])
agents = [
    {
        "name": "texas agent",
        "description": "Answer a question about Texas",
        "func": texas_agent.respond,
    },
    {
        "name": "europe agent",
        "description": "Answer a question about Europe",
        "func": europe_agent.respond,
    },
    {
        "name": "math agent",
        "description": "When a prompt contains numbers, respond with a math formula",
        "func": math_agent.respond,
    },
]

routing_agent.agents = agents

print(
    routing_agent.route_user_prompt("Tell me about the history of Rome, Texas")
)
print(
    routing_agent.route_user_prompt("Tell me about the history of Rome, Italy")
)
print(
    routing_agent.route_user_prompt(
        "One story takes 2 days, and there are 20 stories"
    )
)
