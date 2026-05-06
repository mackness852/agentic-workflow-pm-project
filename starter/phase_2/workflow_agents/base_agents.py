import csv
import re
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from openai import OpenAI


# DirectPromptAgent class definition
class DirectPromptAgent:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        # # print(f"Prompt: {prompt}")
        client = OpenAI(
            api_key=self.openai_api_key,
            base_url="https://openai.vocareum.com/v1",
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


# AugmentedPromptAgent class definition
class AugmentedPromptAgent:
    def __init__(self, openai_api_key, persona):
        """Initialize the agent with given attributes."""
        self.persona = persona
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Generate a response using OpenAI API."""
        # print(f"Prompt: {input}")
        client = OpenAI(
            api_key=self.openai_api_key,
            base_url="https://openai.vocareum.com/v1",
        )
        system_prompt = (
            f"{self.persona}. Forget all previous conversational context."
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text},
            ],
            temperature=0,
        )

        return response.choices[0].message.content


# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent:
    def __init__(self, openai_api_key, persona, knowledge):
        """Initialize the agent with provided attributes."""
        self.persona = persona
        self.knowledge = knowledge
        self.openai_api_key = openai_api_key
        self.system_msg = (
            f"Persona: {persona}. You are a knowledge-based assistant. Forget all previous context.\n"
            f"Use only the following knowledge to answer, do not use your own knowledge: {knowledge}\n"
            "Answer the prompt based on this knowledge, not your own."
        )
        # print(
        # f"System prompt for KnowledgeAugmentedPromptAgent: {self.system_msg}"
        # )

    def respond(self, input_text):
        """Generate a response using the OpenAI API."""
        # print(f"Prompt: {input}")
        client = OpenAI(
            api_key=self.openai_api_key,
            base_url="https://openai.vocareum.com/v1",
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.system_msg},
                {"role": "user", "content": input_text},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(
        self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100
    ):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=self.openai_api_key,
        )
        response = client.embeddings.create(
            model="text-embedding-3-large", input=text, encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = " "
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], 0, 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text) and separator in text[start:end]:
                idx = text[start:end].rindex(separator)
                if idx > self.chunk_overlap:
                    end = start + idx + len(separator)

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": text[start:end],
                    "chunk_size": end - start,
                    "start_char": start,
                    "end_char": end,
                }
            )

            if end >= len(text):
                break
            start = max(end - self.chunk_overlap, start + 1)
            chunk_id += 1

        with open(
            f"chunks-{self.unique_filename}", "w", newline="", encoding="utf-8"
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding="utf-8")
        df["embeddings"] = df["text"].apply(self.get_embedding)
        df.to_csv(
            f"embeddings-{self.unique_filename}", encoding="utf-8", index=False
        )
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(
            f"embeddings-{self.unique_filename}", encoding="utf-8"
        )
        df["embeddings"] = df["embeddings"].apply(lambda x: np.array(eval(x)))
        df["similarity"] = df["embeddings"].apply(
            lambda emb: self.calculate_similarity(prompt_embedding, emb)
        )

        best_chunk = df.loc[df["similarity"].idxmax(), "text"]

        client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=self.openai_api_key,
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context.",
                },
                {
                    "role": "user",
                    "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}",
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content


class EvaluationAgent:
    def __init__(
        self,
        openai_api_key,
        persona,
        evaluation_criteria,
        worker_agent: KnowledgeAugmentedPromptAgent | AugmentedPromptAgent,
        max_interactions,
    ):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt):
        # This method manages interactions between agents to achieve a solution.
        client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=self.openai_api_key,
        )
        prompt_to_evaluate = initial_prompt
        iteration = 0
        response_from_worker = ""
        instruction_response_message = ""

        for i in range(self.max_interactions):
            # print(f"\n--- Interaction {i + 1} ---")
            iteration = i + 1

            # print(" Step 1: Worker agent generates a response to the prompt")
            # print(f"Prompt:\n{prompt_to_evaluate}")
            response_from_worker = self.worker_agent.respond(
                prompt_to_evaluate
            )
            # print(f"Worker Agent Response:\n{response_from_worker}")

            # print(" Step 2: Evaluator agent judges the response")
            eval_with_last_worker_output_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}\n"
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            evaluation_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.persona},
                    {
                        "role": "user",
                        "content": eval_with_last_worker_output_prompt,
                    },
                ],
                temperature=0,
            )
            evaluation_response = evaluation_response.choices[
                0
            ].message.content.strip()
            # print(f"Evaluator Agent Evaluation:\n{evaluation_response}")

            # print(" Step 3: Check if evaluation is positive")
            if evaluation_response.lower().startswith("yes"):
                # print("✅ Final solution accepted.")
                break
            else:
                # print(" Step 4: Generate instructions to correct the response")
                instruction_prompt = f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation_response}"
                instruction_to_correct_response = (
                    client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": str(self.persona)},
                            {
                                "role": "user",
                                "content": str(instruction_prompt),
                            },
                        ],
                        temperature=0,
                    )
                )
                instructions = instruction_to_correct_response.choices[
                    0
                ].message.content.strip()
                # print(f"Instructions to fix:\n{instructions}")

                # print(" Step 5: Send feedback to worker agent for refinement")
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}"
                )
        return {
            "final_response": response_from_worker,
            "evaluation": evaluation_response,
            "iterations": iteration,
        }


class RoutingAgent:
    def __init__(self, openai_api_key, agents):
        # Initialize the agent with given attributes
        self.openai_api_key = openai_api_key
        self.agents = agents

    def get_embedding(self, text):
        client = OpenAI(
            api_key=self.openai_api_key,
            base_url="https://openai.vocareum.com/v1",
        )

        response = client.embeddings.create(
            model="text-embedding-3-large", input=text
        )

        embedding = response.data[0].embedding
        return embedding

    def route_user_prompt(self, user_input):
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            agent_emb = self.get_embedding(agent["description"])

            if agent_emb is None:
                continue

            similarity = np.dot(input_emb, agent_emb) / (
                np.linalg.norm(input_emb) * np.linalg.norm(agent_emb)
            )
            # print(similarity)
            if similarity > best_score:
                best_agent = agent
                best_score = similarity

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        # print(
        # f"‼️‼️‼️‼️\n[Router] Best agent: {best_agent['name']} (score={best_score:.3f})\nPrompt: {user_input}\n‼️‼️‼️‼️"
        # )
        return best_agent["func"](user_input)


class ActionPlanningAgent:
    def __init__(self, openai_api_key, knowledge):
        self.openai_api_key = openai_api_key
        self.system_prompt = f"""Role: You are an action planning agent. Using your 
        knowledge, you extract from the user prompt the steps requested to 
        complete the action the user is asking for. You return the steps as a 
        list of tasks only. Constraint: Only return the steps in your knowledge, do not 
        preface the steps with any text. 
        Forget any previous context. This is your knowledge: {knowledge}\n
        Right example: 1. Task 1\n 2. Task 2\n 3. Task 3
        Wrong example: These tasks are usually used:\n 1. Task 1\n 2. Task 2\n 3. Task 3"""

    def extract_steps_from_prompt(self, prompt):
        # print(f"Prompt: {prompt}")
        client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=self.openai_api_key,
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        response_text = str(response.choices[0].message.content)

        steps = response_text.split("\n")

        return steps
