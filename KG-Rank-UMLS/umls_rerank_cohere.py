import torch
import cohere
import numpy as np
import re
import json
import requests
import time
from collections import deque
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

#--------------- LOAD ENV VARIABLES ---------------------------
from dotenv import load_dotenv
import os

# load the variables from .env into the environment
load_dotenv()

# Load both API keys
cohere_api_key_1 = os.getenv("COHERE_API_KEY")
cohere_api_key_2 = os.getenv("COHERE_API_KEY_2")

# Initialize current API key and counter
current_api_key = cohere_api_key_1
api_call_counter = 0
max_calls_per_key = 999  # Maximum calls before switching keys
umls_api_key = os.getenv("UMLS_API_KEY")
#--------------- LOAD ENV VARIABLES ---------------------------

# Enhanced rate limiter with API key switching capability
class EnhancedRateLimiter:
    def __init__(self, max_calls=9, time_window=60):
        self.call_timestamps = deque(maxlen=max_calls)
        self.max_calls = max_calls  # Maximum calls in time window (using 9 to be safe)
        self.time_window = time_window  # Time window in seconds (60 = 1 minute)
        self.total_api_calls = 0  # Track total API calls for key switching
    
    def wait_if_needed(self):
        """Wait only if necessary to maintain rate limits"""
        current_time = time.time()
        
        # Remove timestamps older than our time window
        while self.call_timestamps and (current_time - self.call_timestamps[0] > self.time_window):
            self.call_timestamps.popleft()
        
        # If we've reached max calls within the time window, wait until the oldest expires
        if len(self.call_timestamps) >= self.max_calls:
            # Calculate minimum time needed to wait
            oldest_timestamp = self.call_timestamps[0]
            wait_time = oldest_timestamp + self.time_window - current_time
            
            if wait_time > 0:
                print(f"Rate limiting: Waiting {wait_time:.2f} seconds (made {len(self.call_timestamps)} calls in last minute)")
                time.sleep(wait_time + 0.1)  # Add a tiny buffer
                return True
        
        # Record this call
        self.call_timestamps.append(time.time())
        return False
    
    def record_api_call(self):
        """Record an API call and increment total count"""
        global api_call_counter, current_api_key, cohere_api_key_1, cohere_api_key_2
        
        # Increment the counter
        api_call_counter += 1
        self.total_api_calls += 1
        
        # Check if we need to switch API keys
        if current_api_key == cohere_api_key_1 and api_call_counter >= max_calls_per_key:
            print(f"⚠️ Reached {api_call_counter} API calls with first key. Switching to second API key.")
            current_api_key = cohere_api_key_2
            # Reset rate limiting queue when switching keys
            self.call_timestamps.clear()
            
            # Create new Cohere client with the new key
            global cohere_reranker
            cohere_reranker = UMLS_CohereReranker(current_api_key)
            
        print(f"📊 API call #{api_call_counter} recorded. Current usage: {len(self.call_timestamps)}/{self.max_calls} calls in last minute")
        return current_api_key

# Create global rate limiter
cohere_limiter = EnhancedRateLimiter(max_calls=9, time_window=60)

class UMLSBERT:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("GanjinZero/UMLSBert_ENG")
        self.model = AutoModel.from_pretrained("GanjinZero/UMLSBert_ENG")

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def batch_encode(self, texts, batch_size=16):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                model_output = self.model(**inputs)
                attention_mask = inputs["attention_mask"]
                batch_embeddings = self.mean_pooling(model_output, attention_mask)
            all_embeddings.extend(batch_embeddings)  
        return np.array(all_embeddings)

def get_similarity(query_vec, rel_vec):
    return cosine_similarity(query_vec, rel_vec)

class UMLS_CohereReranker:
    def __init__(self, api_key):
        self.co = cohere.Client(api_key)
        self.api_key = api_key
    
    def rerank(self, query, rels, top_n=20, max_retries=3):
        """Rate-limited rerank with retries"""
        global current_api_key
        
        for attempt in range(max_retries + 1):
            try:
                # Wait only if needed before making the API call
                cohere_limiter.wait_if_needed()
                
                # Check if API key has changed and update client if needed
                if self.api_key != current_api_key:
                    self.co = cohere.Client(current_api_key)
                    self.api_key = current_api_key
                    print("Updated Cohere client with new API key")
                
                # Make the API call
                results = self.co.rerank(query=query, documents=rels, top_n=top_n, model="rerank-english-v2.0")
                
                # Record the successful API call
                cohere_limiter.record_api_call()
                
                # Process results
                reranked_results = []
                for r in results:
                    reranked_results.append({
                        "rels": r.document["text"],  
                        "relevance_score": r.relevance_score
                    })
                return reranked_results
                
            except Exception as e:
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "trial key" in error_msg:
                    # For rate limit errors, wait a bit longer
                    wait_time = 62 if attempt == 0 else 62 * (attempt + 1)
                    print(f"Rate limit error: {e}")
                    print(f"Waiting {wait_time} seconds before retry... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                elif attempt < max_retries:
                    # For other errors, use shorter waits
                    wait_time = 10 * (attempt + 1)
                    print(f"API error: {e}")
                    print(f"Waiting {wait_time} seconds before retry... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"Failed after {max_retries} retries: {e}")
                    return []  # Return empty list as fallback

class UMLS_API:
    def __init__(self, apikey, version="current"):
        self.apikey = apikey
        self.version = version
        self.search_url = f"https://uts-ws.nlm.nih.gov/rest/search/{version}"
        self.content_url = f"https://uts-ws.nlm.nih.gov/rest/content/{version}"
        self.content_suffix = "/CUI/{}/{}?apiKey={}"

    def search_cui(self, query):
        cui_results = []

        try:
            page = 1
            size = 1
            query = {"string": query, "apiKey": self.apikey, "pageNumber": page, "pageSize": size}
            r = requests.get(self.search_url, params=query)
            r.raise_for_status()
            print(r.url)
            r.encoding = 'utf-8'
            outputs = r.json()

            items = outputs["result"]["results"]

            if len(items) == 0:
                print("No results found.\n")

            for result in items:
                cui_results.append((result["ui"], result["name"]))

        except Exception as except_error:
            print(except_error)

        return cui_results

    def get_definitions(self, cui):
        try:
            suffix = self.content_suffix.format(cui, "definitions", self.apikey)
            r = requests.get(self.content_url + suffix)
            r.raise_for_status()
            r.encoding = "utf-8"
            outputs = r.json()

            return outputs["result"]
        except Exception as except_error:
            print(except_error)

    def get_relations(self, cui, pages=20):
        all_relations = []

        try:
            for page in range(1, pages + 1):
                suffix = self.content_suffix.format(cui, "relations", self.apikey) + f"&pageNumber={page}"
                r = requests.get(self.content_url + suffix)
                r.raise_for_status()
                r.encoding = "utf-8"
                outputs = r.json()

                page_relations = outputs.get("result", [])
                all_relations.extend(page_relations)

        except Exception as except_error:
            print(except_error)

        return all_relations

# Initialize components
umls_api = UMLS_API(umls_api_key)
umlsbert = UMLSBERT()
cohere_reranker = UMLS_CohereReranker(current_api_key)

def get_umls_keys(query, prompt, llm):
    umls_res = {}
    prompt = prompt.replace("{question}", query)

    try:
        keys_text = llm.predict(prompt)
        print(keys_text)
        pattern = r"\{(.*?)\}"
        matches = re.findall(pattern, keys_text.replace("\n", ""))
        if not matches:
            raise ValueError("No medical terminologies returned by the model.")
        
        keys_dict = json.loads("{" + matches[0] + "}")
        if "medical terminologies" not in keys_dict or not keys_dict["medical terminologies"]:
            raise ValueError("Model did not return expected 'medical terminologies' key.")
    except Exception as e:
        print(f"Error during model processing: {e}")
        return "" 

    # Process each medical terminology
    for key in keys_dict["medical terminologies"][:]:
        print(f"Processing: {key}")
        
        # Get CUI information
        cuis = umls_api.search_cui(key)
        if len(cuis) == 0:
            continue
            
        cui = cuis[0][0]
        name = cuis[0][1]
        print(f"Found CUI: {cui} - {name}")

        # Get definitions
        defi = ""
        definitions = umls_api.get_definitions(cui)
        if definitions is not None:
            msh_def = None
            nci_def = None
            icf_def = None
            csp_def = None
            hpo_def = None

            for definition in definitions:
                source = definition["rootSource"]
                if source == "MSH":
                    msh_def = definition["value"]
                    break
                elif source == "NCI":
                    nci_def = definition["value"]
                elif source == "ICF":
                    icf_def = definition["value"]
                elif source == "CSP":
                    csp_def = definition["value"]
                elif source == "HPO":
                    hpo_def = definition["value"]

            defi = msh_def or nci_def or icf_def or csp_def or hpo_def

        # Process relations efficiently
        rels = []
        relations = umls_api.get_relations(cui)
        
        if relations and len(relations) > 0:
            print(f"Found {len(relations)} relations. Processing...")
            
            # Create embeddings
            relation_texts = [query] + [
                f"{rel.get('relatedFromIdName', '')} {rel.get('additionalRelationLabel', '').replace('_', ' ')} {rel.get('relatedIdName', '')}" 
                for rel in relations
            ]
            
            embeddings = umlsbert.batch_encode(relation_texts)
            query_embedding = embeddings[0]
            relation_embeddings = embeddings[1:]
            
            # Calculate similarity and sort
            relation_scores = []
            for rel_emb, rel in zip(relation_embeddings, relations):
                sim_score = get_similarity([query_embedding], [rel_emb])[0][0]
                relation_scores.append((sim_score, rel))
                
            relation_scores.sort(key=lambda x: x[0], reverse=True)
            
            # Use larger batches but still be cautious
            max_rels = min(60, len(relation_scores))  # Increased from 30 to 60
            top_rels = relation_scores[:max_rels]
            
            # Prepare texts for reranking
            top_rel_texts = [
                f"{rel[1].get('relatedFromIdName', '')} {rel[1].get('additionalRelationLabel', '').replace('_', ' ')} {rel[1].get('relatedIdName', '')}" 
                for rel in top_rels
            ]
            
            # Process in optimal batch sizes - larger but still safe
            if top_rel_texts:
                batch_size = 20  # Increased from 10 to 20
                reranked_results = []
                
                for i in range(0, len(top_rel_texts), batch_size):
                    batch = top_rel_texts[i:i+batch_size]
                    
                    # Use rate-limited reranker that only waits when necessary
                    print(f"Reranking batch {i//batch_size + 1} of {(len(top_rel_texts) + batch_size - 1)//batch_size}")
                    batch_results = cohere_reranker.rerank(query, batch, top_n=min(len(batch), 10))
                    
                    if batch_results:
                        reranked_results.extend(batch_results)
                
                # Take more top results to get better quality
                if reranked_results:
                    reranked_results.sort(key=lambda x: x["relevance_score"], reverse=True)
                    reranked_results = reranked_results[:10]  # Increased from 5 to 10
                    
                    for result in reranked_results:
                        rel = result["rels"]  
                        score = result["relevance_score"]
                        rels.append((rel, score))

        # Store results for this CUI
        umls_res[cui] = {"name": name, "definition": defi, "rels": rels}

    # Build the final context string
    context = "" 
    for k, v in umls_res.items():
        name = v["name"]
        definition = v["definition"]
        rels = v["rels"]
        
        rels_text = ""
        for rel in rels:
            relation_description, relevance_score = rel
            rels_text += f"({relation_description})\n"
            
        text = f"Name: {name}\nDefinition: {definition}\n"
        if rels_text != "":
            text += f"Relations: \n{rels_text}"
            
        context += text + "\n"
        
    if context != "":
        context = context[:-1]
        
    return context