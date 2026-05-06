# agentic_workflow.py

# from workflow_agents.base_agents import ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent
import os

from dotenv import load_dotenv
from workflow_agents.base_agents import (
    ActionPlanningAgent,
    EvaluationAgent,
    KnowledgeAugmentedPromptAgent,
    RoutingAgent,
)

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

with open("starter/phase_2/Product-Spec-Email-Router.txt") as infile:
    product_spec = infile.read()

# Instantiate all the agents

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)
action_agent = ActionPlanningAgent(openai_api_key, knowledge_action_planning)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the product spec and user stories for a product. Be brief in your responses."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    f"{product_spec}"
)
pm_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key, persona_product_manager, knowledge_product_manager
)

# Product Manager - Evaluation Agent
pm_eval_persona = "You are an evaluation agent that checks the answers of other worker agents"
pm_eval_criteria = "The answer should be stories that follow the following structure: As a [type of user], I want [an action or feature] so that [benefit/value]."
pm_eval_agent = EvaluationAgent(
    openai_api_key,
    pm_eval_persona,
    pm_eval_criteria,
    pm_agent,
    5,
)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = f"You are a Program Manager, you are responsible for defining the features for a product. Product spec is {product_spec}.\n Be brief in your responses."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."
program_manager_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key, persona_program_manager, knowledge_program_manager
)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
program_manager_eval_criteria = (
    "The answer should be product features that follow the following structure: "
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user"
)
program_mgr_eval_agent = EvaluationAgent(
    openai_api_key,
    persona_program_manager_eval,
    program_manager_eval_criteria,
    program_manager_agent,
    5,
)


# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = f"You are a Development Engineer, you are responsible for defining the development tasks for a product. Product spec is: {product_spec}.\n Be brief in your responses."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."
dev_engineer_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key, persona_dev_engineer, knowledge_dev_engineer
)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
dev_engineer_eval_criteria = (
    "The answer should be tasks following this exact structure: "
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first"
)
dev_engineer_eval_agent = EvaluationAgent(
    openai_api_key,
    persona_dev_engineer_eval,
    dev_engineer_eval_criteria,
    dev_engineer_agent,
    5,
)


# Routing Agent
agents = [
    {
        "name": "Product Manager",
        "description": "Identifyies product spec requirements and defines personas and user stories.",
        "func": lambda query: product_manager_support_function(query),
    },
    {
        "name": "Program Manager",
        "description": "Groups related user stories into features",
        "func": lambda query: program_manager_support_function(query),
    },
    {
        "name": "Development Engineer",
        "description": "Defines engineering tasks for each story to represent the engineering work required",
        "func": lambda query: developer_support_function(query),
    },
]

routing_agent = RoutingAgent(openai_api_key, agents)


def product_manager_support_function(query):
    eval = pm_eval_agent.evaluate(pm_agent.respond(query))
    return eval["final_response"]


def program_manager_support_function(query):
    eval = program_mgr_eval_agent.evaluate(
        program_manager_agent.respond(query)
    )
    return eval["final_response"]


def developer_support_function(query):
    eval = dev_engineer_eval_agent.evaluate(dev_engineer_agent.respond(query))
    return eval["final_response"]


# Run the workflow

print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
workflow_prompt = "What would the development tasks for this product be?"
# ****
print(
    f"Task to complete in this workflow, workflow prompt = {workflow_prompt}, product spec: {product_spec}"
)

print("\nDefining workflow steps from the workflow prompt")

steps = action_agent.extract_steps_from_prompt(workflow_prompt)
print(
    f"==============\n==============\nSteps to complete: \n{str(steps)}==============\n==============\n"
)
completed_steps = []
for step in steps:
    # print(f"Attempting step: {step}")
    result = routing_agent.route_user_prompt(f"{step}")
    # print(f"✅ Result from step {step}: \n{result}")
    completed_steps.append(result)
print("\n".join(completed_steps))
