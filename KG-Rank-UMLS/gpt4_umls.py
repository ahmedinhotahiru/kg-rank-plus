from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from typing import List, Dict, Any

# choose different ranking techniques
from multi_hop_umls_mmr_new import get_umls_keys
# from new_multihop_expansion import get_umls_keys
# from bfs_multihop_expansion import get_umls_keys

from tqdm import tqdm
import pandas as pd
import datetime

#--------------- LOAD ENV VARIABLES ---------------------------

# import environment variables
from dotenv import load_dotenv
import os

# load the variables from .env into the environment
load_dotenv()

openai_api_key=os.getenv("OPENAI_API_KEY")
#--------------- LOAD ENV VARIABLES ---------------------------



llm = ChatOpenAI(model_name="gpt-4",
                 temperature=0,
                 openai_api_key=openai_api_key)

class ExtendedConversationBufferWindowMemory(ConversationBufferWindowMemory):
    extra_variables:List[str] = []

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables."""
        return [self.memory_key] + self.extra_variables

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return buffer with history and extra variables"""
        d = super().load_memory_variables(inputs)
        d.update({k:inputs.get(k) for k in self.extra_variables})
        return d

memory = ExtendedConversationBufferWindowMemory(k=0,
                                                ai_prefix="Physician",
                                                human_prefix="Patient",
                                                extra_variables=["context"])

template = """
Answer the question in conjunction with the following content.

Context:
{context}
Current conversation:
{history}
Patient: {input}
Physician:
"""

PROMPT = PromptTemplate(
    input_variables=["context", "history", "input"], template=template
)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=PROMPT,
    verbose=True,
)

PROMPT = """
Question: {question}

You are interacting with a knowledge graph that contains definitions and relational information of medical terminologies. To provide a precise and relevant answer to this question, you are expected to:

1. Understand the Question Thoroughly: Analyze the question deeply to identify which specific medical terminologies and their interrelations, as extracted from the knowledge graph, are crucial for formulating an accurate response.

2. Extract Key Terminologies: Return the 3-5 most relevant medical terminologies based on their significance to the question.

3. Format the Output : Return in a structured JSON format with the key as "medical terminologies". For example:

{"medical terminologies": ["term1", "term2", ...]}
"""



# ---------- DEFINE FINE NAMES AND PATHS OF INPUT AND OUTPUT -----------------

# Define input and output paths
input_file = "/ocean/projects/cis240101p/aissah/KG-Rank-main/evaluation_datasets/LiveQA.csv" # TODO
dataset_name = os.path.basename(input_file).split('.')[0]
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = "/ocean/projects/cis240101p/aissah/KG-Rank-main/results_multihop_best"
model_type = "gpt4" # TODO
ranking_type = "mmr" # TODO
output_file = f"{results_dir}/{dataset_name}_{model_type}_{ranking_type}_multihop_results_{timestamp}.csv"

# Create results directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

# -----------------------------------------------------------------------------



# data = pd.read_csv("path")
data = pd.read_csv("/ocean/projects/cis240101p/aissah/KG-Rank-main/evaluation_datasets/LiveQA.csv") # TODO
questions = data["Question"].tolist() 
reference_answers = data["Answer"].tolist()

# Initialize result storage
results = []
contexts = []

for i in tqdm(questions, desc="Processing Questions"):
    # This is only for Answer Expansion ranking method
    # pre_answer = llm.predict(i)
    # context = get_umls_keys(i, pre_answer, PROMPT, llm)

    context = get_umls_keys(i, PROMPT, llm)
    answer = conversation.predict(context=context, input=i)
    print(f"Answer {i}: {answer}")
    print("-" * 50)
        
    results.append(answer)




# Create results dataframe
results_df = pd.DataFrame({
    "Question": questions,
    "Generated_Answer": results,
})

# Add reference answers if available
results_df["Reference_Answer"] = reference_answers

# Save results
results_df.to_csv(output_file, index=False)
print(f"Results saved to: {output_file}")

# Optional: Save contexts (may be large) for debugging purposes
# contexts_file = f"{results_dir}/{dataset_name}_contexts_{timestamp}.csv"
# contexts_df = pd.DataFrame({
#     "Question": questions,
#     "Context": contexts
# })
# contexts_df.to_csv(contexts_file, index=False)
# print(f"Contexts saved to: {contexts_file}")

print(f"Processed {len(results)} questions successfully.")


# print(results)