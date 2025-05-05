import torch
import transformers
from typing import List, Dict, Any
from langchain.prompts import PromptTemplate
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from torch import cuda
from umls_rerank_cohere import get_umls_keys
# choose different ranking techniques
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
import accelerate
import datetime



#--------------- LOAD ENV VARIABLES ---------------------------

# import environment variables
from dotenv import load_dotenv
import os

# load the variables from .env into the environment
load_dotenv()

llama_token=os.getenv("LLAMA_TOKEN")
#--------------- LOAD ENV VARIABLES ---------------------------



# checkpoint = "/groups/gcb50389/pretrained/llama2-HF/Llama-2-13b-chat-hf"
checkpoint = "meta-llama/Llama-2-13b-chat-hf"

device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    trust_remote_code=True,
    quantization_config=nf4_config,
    device_map="auto",
    token=llama_token
)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model.eval()

stop_list = ["\nHuman:", "\n```\n"]
stop_token_ids = [tokenizer(x)["input_ids"] for x in stop_list]
stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

pipeline = transformers.pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="cuda",
    max_new_tokens=4096,
    return_full_text=True,
    stopping_criteria=stopping_criteria,
    repetition_penalty=1.1,
    do_sample=True,
)

llama = HuggingFacePipeline(pipeline=pipeline)
    
class ExtendedConversationBufferWindowMemory(ConversationBufferWindowMemory):
    extra_variables: List[str] = []

    @property
    def memory_variables(self) -> List[str]:
        """Will only return list of extra memory variables, not including self.memory_key."""
        return self.extra_variables

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return buffer with extra variables, but exclude the default 'history' memory_key."""
        d = super().load_memory_variables(inputs)
        d.pop("history", None)  
        d.update({k: inputs.get(k) for k in self.extra_variables})
        return d


memory = ExtendedConversationBufferWindowMemory(k=0,
                                                ai_prefix="Physician",
                                                human_prefix="Patient",
                                                extra_variables=["context"])

template = """
<s>[INST] <<SYS>>
Answer the question in conjunction with the following content.
<</SYS>>

Context:
{context}

Question: {input}
Answer: [/INST]
"""

PROMPT = PromptTemplate(
    input_variables=["context", "input"], template=template
)

conversation = ConversationChain(
    llm=llama,
    memory=memory,
    prompt=PROMPT,
    verbose=True,
)

PROMPT = """
[INST] Input: {question}

Our knowledge graph contains definitions and relational information for medical terminologies. To accurately answer this question, please follow these steps:

1. Analyze the question to determine which medical terminologies' definitions and their relationships, if extracted from the knowledge graph, would be particularly helpful in answering the question. 

2. Return 3-5 key medical terminologies in JSON format, as shown below:

{"medical terminologies": ["term1", "term2", ...]} [/INST]
"""



# ---------- DEFINE FINE NAMES AND PATHS OF INPUT AND OUTPUT -----------------

# Define input and output paths
input_file = "/ocean/projects/cis240101p/aissah/KG-Rank-main/evaluation_datasets/MedQA.csv"
dataset_name = os.path.basename(input_file).split('.')[0]
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = "/ocean/projects/cis240101p/aissah/KG-Rank-main/results"
model_type = "gpt4"
ranking_type = "rerank_cohere"
output_file = f"{results_dir}/{dataset_name}_{model_type}_{ranking_type}_results_{timestamp}.csv"

# Create results directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

# -----------------------------------------------------------------------------



# data = pd.read_csv("path")
data = pd.read_csv("/ocean/projects/cis240101p/aissah/KG-Rank-main/evaluation_datasets/LiveQA.csv")
questions = data["Question"].tolist() 
reference_answers = data["Answer"].tolist()

# Initialize result storage
results = []
contexts = []

for i in tqdm(questions, desc="Processing Questions"):
    context = get_umls_keys(i, PROMPT, llama)
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