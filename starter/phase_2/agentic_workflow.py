# agentic_workflow.py

import os

from dotenv import load_dotenv
from workflow_agents.base_agents import (
    ActionPlanningAgent,
    AugmentedPromptAgent,
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
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents and ensure they adhere to any constraints."
program_manager_eval_criteria = (
    "The answer should be product features that follow the below structure and only use the headings Feature Name, Description, Key Functionality and User Benefit: \n"
    "Example:\n"
    "Feature Name: [A clear, concise title that identifies the capability]\n"
    "- Description: [A brief explanation of what the feature does and its purpose]\n"
    "- Key Functionality: [The specific capabilities or actions the feature provides]\n"
    "- User Benefit: [How this feature creates value for the user]\n"
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


# Delivery Manager - Knowledge Augmented Prompt Agent
persona_delivery_manager = (
    "Role: You are a software delivery manager. \n"
    "Task: When given a list of User Stories, Product Features and Engineering"
    "tasks, you summarise them into a markdown format for discussion"
    "amongst the team.\n"
    "Example:\n"
    "# User Stories:\n"
    "- User story 1\n"
    "- User story 2...\n"
    "# Features:\n"
    "- Feature 1\n"
    "- Feature 2...\n"
    "# Engineering Tasks:\n"
    "- Engineering task 1\n"
    "- Engineering task 2...\n"
)
delivery_mgr_agent = AugmentedPromptAgent(
    openai_api_key, persona_delivery_manager
)

# Delivery Manager - Evaluation Agent
delivery_mgr_eval_persona = "You are an evaluation agent that checks the answers of other worker agents"
delivery_mgr_eval_criteria = (
    "The answer should be a markdown file with clear headings for User Stories, Features and Engineering Tasks."
    "Under these headings should be all of the original list's details about those topics."
)
delivery_mgr_eval_agent = EvaluationAgent(
    openai_api_key,
    delivery_mgr_eval_persona,
    delivery_mgr_eval_criteria,
    delivery_mgr_agent,
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
workflow_prompt = "What would the development tasks for this product be?"
steps = action_agent.extract_steps_from_prompt(workflow_prompt)
completed_steps = []
for step in steps:
    result = routing_agent.route_user_prompt(f"{step}")
    completed_steps.append(result)
print(
    delivery_mgr_eval_agent.evaluate(
        delivery_mgr_agent.respond(str(completed_steps))
    )["final_response"]
)
