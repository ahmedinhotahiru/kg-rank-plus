import torch
import cohere
import numpy as np
import re
import json
import requests
import time
import random
import os
import pickle
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Set, Any, Optional

#--------------- LOAD ENV VARIABLES ---------------------------
from dotenv import load_dotenv
load_dotenv()
cohere_api_key=os.getenv("COHERE_API_KEY")
umls_api_key=os.getenv("UMLS_API_KEY")
#--------------- LOAD ENV VARIABLES ---------------------------

# Persistent cache file paths
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)
EMBEDDING_CACHE_FILE = os.path.join(CACHE_DIR, 'embedding_cache.pkl')
CUI_CACHE_FILE = os.path.join(CACHE_DIR, 'cui_cache.pkl')
DEF_CACHE_FILE = os.path.join(CACHE_DIR, 'definition_cache.pkl')
REL_CACHE_FILE = os.path.join(CACHE_DIR, 'relation_cache.pkl')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_similarity(query_vec, rel_vec):
    return cosine_similarity(query_vec, rel_vec)

# Define a class for UMLS BERT model
class UMLSBERT:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.cache = self._load_cache(EMBEDDING_CACHE_FILE, {})
        self._model_loaded = False
        
    def _load_model(self):
        """Lazy-load the model only when needed"""
        if not self._model_loaded:
            print("Loading UMLSBert model...")
            self.tokenizer = AutoTokenizer.from_pretrained("GanjinZero/UMLSBert_ENG")
            self.model = AutoModel.from_pretrained("GanjinZero/UMLSBert_ENG")
            self._model_loaded = True
            
    def _load_cache(self, cache_file, default):
        """Load cache from disk if exists"""
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache {cache_file}: {e}")
        return default
        
    def _save_cache(self, cache_file, cache_dict):
        """Save cache to disk"""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_dict, f)
        except Exception as e:
            print(f"Error saving cache {cache_file}: {e}")

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def batch_encode(self, texts, batch_size=32):  # Increased batch size
        """Encode texts into embeddings with caching"""
        # Ensure model is loaded
        self._load_model()
        
        # Initialize embeddings list
        all_embeddings = []
        
        # Remove duplicates to avoid redundant processing
        unique_texts = list(set(texts))
        text_to_idx = {text: idx for idx, text in enumerate(texts)}
        
        # Check cache for all texts
        uncached_texts = []
        uncached_indices = []
        cached_embeddings = {}
        
        for i, text in enumerate(texts):
            cache_key = hash(text)
            if cache_key in self.cache:
                cached_embeddings[i] = self.cache[cache_key]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Process uncached texts
        if uncached_texts:
            batch_embeddings = []
            for i in range(0, len(uncached_texts), batch_size):
                batch_texts = uncached_texts[i:i+batch_size]
                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    model_output = self.model(**inputs)
                    attention_mask = inputs["attention_mask"]
                    embeddings = self.mean_pooling(model_output, attention_mask)
                
                # Process this batch
                for j, text in enumerate(batch_texts):
                    if i + j < len(uncached_indices):
                        idx = uncached_indices[i + j]
                        embedding = embeddings[j].cpu().numpy()
                        self.cache[hash(text)] = embedding
                        cached_embeddings[idx] = embedding
            
            # Save updated cache periodically
            if len(uncached_texts) > 10:  # Only save if we've added a significant number
                self._save_cache(EMBEDDING_CACHE_FILE, self.cache)
        
        # Construct result array using cached embeddings
        for i in range(len(texts)):
            if i in cached_embeddings:
                all_embeddings.append(cached_embeddings[i])
            
        return np.array(all_embeddings)

class UMLS_CohereReranker:
    def __init__(self, api_key):
        self.co = cohere.Client(api_key)
        self.cache = {}  # Add reranking cache
        self.last_call_time = 0
        self.call_interval = 7  # Increased to ~8 calls/minute
        self.rate_limit_hits = 0
        self.max_retries = 3
    
    def rerank(self, query, texts, top_n=20):
        """Rerank texts using Cohere reranking API with aggressive rate limiting and retries"""
        # Check if empty
        if not texts:
            return []
            
        # Create cache key based on query and documents
        cache_key = hash((query, tuple(sorted(texts))))
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # Format documents for Cohere API
        docs = [{"text": text} for text in texts]
        
        # Try multiple times with increasing backoff
        for retry in range(self.max_retries):
            try:
                # Apply aggressive rate limiting
                self._apply_rate_limit()
                
                # Perform reranking
                print(f"Calling Cohere reranking API for {len(texts)} texts (attempt {retry+1}/{self.max_retries})...")
                results = self.co.rerank(query=query, documents=docs, top_n=top_n, model="rerank-english-v2.0")
                
                # Process results
                reranked_results = []
                for r in results:
                    reranked_results.append({
                        "text": r.document["text"],  
                        "relevance_score": r.relevance_score
                    })
                
                # Reset rate limit hits on success
                self.rate_limit_hits = 0
                
                # Cache results
                self.cache[cache_key] = reranked_results
                return reranked_results
                
            except Exception as e:
                print(f"Error in Cohere reranking (attempt {retry+1}): {e}")
                
                # If rate limit error, increase waiting time
                if "rate limit" in str(e).lower() or "trial key" in str(e).lower():
                    self.rate_limit_hits += 1
                    wait_time = min(60, self.call_interval * (2 ** self.rate_limit_hits))  # Exponential backoff
                    print(f"Rate limit hit, waiting {wait_time}s before next attempt...")
                    time.sleep(wait_time)
                else:
                    # For other errors, break immediately
                    break
        
        # Fallback to embedding similarity rankings if all retries fail
        print("All Cohere reranking attempts failed, falling back to embedding scores...")
        # Return texts with descending confidence scores as fallback
        return [{"text": text, "relevance_score": 0.9 - (i * 0.02)} 
                for i, text in enumerate(texts[:top_n])]

    def _apply_rate_limit(self):
        """Apply aggressive rate limiting to API requests"""
        current_time = time.time()
        elapsed = current_time - self.last_call_time
        if elapsed < self.call_interval:
            sleep_time = self.call_interval - elapsed
            print(f"Rate limiting Cohere API, sleeping for {sleep_time:.2f}s...")
            time.sleep(sleep_time)
        self.last_call_time = time.time()
        
# Define a class for UMLS API
class UMLS_API:
    def __init__(self, apikey, version="current"):
        self.apikey = apikey
        self.version = version
        self.search_url = f"https://uts-ws.nlm.nih.gov/rest/search/{version}"
        self.content_url = f"https://uts-ws.nlm.nih.gov/rest/content/{version}"
        self.content_suffix = "/CUI/{}/{}?apiKey={}"
        
        # Add caching
        self.cui_cache = self._load_cache(CUI_CACHE_FILE, {})
        self.definition_cache = self._load_cache(DEF_CACHE_FILE, {})
        self.relation_cache = self._load_cache(REL_CACHE_FILE, {})
        self.path_cache = {}
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

    def _load_cache(self, cache_file, default):
        """Load cache from disk if exists"""
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache {cache_file}: {e}")
        return default
        
    def _save_cache(self, cache_file, cache_dict):
        """Save cache to disk"""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_dict, f)
        except Exception as e:
            print(f"Error saving cache {cache_file}: {e}")

    def _rate_limit(self):
        """Apply rate limiting to API requests"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def search_cui(self, query):
        # Check cache
        if query in self.cui_cache:
            return self.cui_cache[query]
            
        cui_results = []
        try:
            self._rate_limit()
            page = 1
            size = 1
            query_params = {"string": query, "apiKey": self.apikey, "pageNumber": page, "pageSize": size}
            r = requests.get(self.search_url, params=query_params)
            r.raise_for_status()
            print(r.url)
            r.encoding = 'utf-8'
            outputs = r.json()

            items = outputs["result"]["results"]
            if len(items) == 0:
                print("No results found.\n")
            else:
                for result in items:
                    cui_results.append((result["ui"], result["name"]))
        except Exception as e:
            print(f"Error searching for CUI: {e}")

        # Update cache
        self.cui_cache[query] = cui_results
        
        # Periodically save cache
        if len(self.cui_cache) % 20 == 0:
            self._save_cache(CUI_CACHE_FILE, self.cui_cache)
            
        return cui_results

    def get_definitions(self, cui):
        # Check cache
        if cui in self.definition_cache:
            return self.definition_cache[cui]
            
        try:
            self._rate_limit()
            suffix = self.content_suffix.format(cui, "definitions", self.apikey)
            r = requests.get(self.content_url + suffix)
            
            # Handle 404 errors gracefully
            if r.status_code == 404:
                self.definition_cache[cui] = []
                return []
                
            r.raise_for_status()
            r.encoding = "utf-8"
            outputs = r.json()
            
            # Update cache
            result = outputs["result"]
            self.definition_cache[cui] = result
            
            # Periodically save cache
            if len(self.definition_cache) % 20 == 0:
                self._save_cache(DEF_CACHE_FILE, self.definition_cache)
                
            return result
        except requests.exceptions.HTTPError as e:
            if "404" in str(e):
                print(f"No definitions found for {cui} (this is normal)")
                self.definition_cache[cui] = []
                return []
            else:
                print(f"Error retrieving definitions for {cui}: {e}")
                return []
        except Exception as e:
            print(f"Error retrieving definitions for {cui}: {e}")
            return []

    def get_relations(self, cui, pages=5):  # Reduced from 10 to 5 pages for speed
        # Check cache
        if cui in self.relation_cache:
            return self.relation_cache[cui]
            
        all_relations = []
        try:
            for page in range(1, pages + 1):
                self._rate_limit()
                suffix = self.content_suffix.format(cui, "relations", self.apikey) + f"&pageNumber={page}"
                r = requests.get(self.content_url + suffix)
                
                # Handle 404 errors gracefully
                if r.status_code == 404:
                    break
                    
                r.raise_for_status()
                r.encoding = "utf-8"
                outputs = r.json()

                page_relations = outputs.get("result", [])
                if not page_relations:
                    break  # No more results
                    
                all_relations.extend(page_relations)
        except Exception as e:
            print(f"Error retrieving relations for {cui}: {e}")

        # Update cache
        self.relation_cache[cui] = all_relations
        
        # Periodically save cache
        if len(self.relation_cache) % 10 == 0:
            self._save_cache(REL_CACHE_FILE, self.relation_cache)
            
        return all_relations

    def get_multi_hop_paths(self, cui, max_hops=3, max_relations_per_level=None):
        """
        Get multi-hop relations with unbiased selection
        """
        # Set default relations per level if not provided
        if max_relations_per_level is None:
            max_relations_per_level = {
                1: 20,  # Level 1: Expand top 20 concepts
                2: 7,   # Level 2: Expand top 7 concepts
                3: 3    # Level 3: Expand top 3 concepts
            }
            
        # Check cache
        cache_key = (cui, max_hops, str(max_relations_per_level))
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
            
        # Dictionary to store relation paths
        all_relation_paths = []
        
        # Set to track visited CUIs to avoid cycles
        visited_cuis = set([cui])
        
        # Queue for BFS with initial CUI and empty path
        queue = deque([(cui, [], 1)])  # (cui, path, hop_level)
        
        while queue:
            # Dequeue current CUI, path, and hop level
            current_cui, current_path, hop_level = queue.popleft()
            
            # Stop if we've reached the maximum hop level
            if hop_level > max_hops:
                continue
                
            # Get relations for this CUI
            relations = self.get_relations(current_cui)
            
            # If no relations found, continue to next item in queue
            if not relations:
                continue
            
            # Get max relations to explore at this hop level
            max_to_expand = max_relations_per_level.get(hop_level, 1)
            
            # Randomly sample relations to avoid biases - UNBIASED SELECTION
            if len(relations) > max_to_expand:
                pruned_relations = random.sample(relations, max_to_expand)
            else:
                pruned_relations = relations
            
            # Process each relation
            for relation in pruned_relations:
                # Get the related CUI
                related_cui = relation.get('relatedId')
                if not related_cui:
                    continue
                
                # Create relation text
                rel_text = f"{relation.get('relatedFromIdName', '')} {relation.get('additionalRelationLabel', '').replace('_', ' ')} {relation.get('relatedIdName', '')}"
                
                # Create a new path with this relation
                new_path = current_path + [(rel_text, relation, hop_level)]
                
                # Add the complete path for ranking
                if new_path:  # Only add non-empty paths
                    all_relation_paths.append(new_path)
                
                # Continue BFS only if this CUI hasn't been visited yet (avoid cycles)
                if related_cui not in visited_cuis:
                    visited_cuis.add(related_cui)
                    queue.append((related_cui, new_path, hop_level + 1))
        
        # Update cache
        self.path_cache[cache_key] = all_relation_paths
        return all_relation_paths

def extract_medical_entities(keys_text):
    """Extract medical entities with robust parsing"""
    try:
        # Try standard JSON parsing first
        pattern = r"\{(.*?)\}"
        matches = re.findall(pattern, keys_text.replace("\n", ""))
        if matches:
            entities_dict = json.loads("{" + matches[0] + "}")
            if "medical terminologies" in entities_dict:
                return entities_dict["medical terminologies"]
                
        # Fallback: Try to extract array directly
        array_pattern = r"\"medical terminologies\":\s*\[(.*?)\]"
        array_matches = re.findall(array_pattern, keys_text.replace("\n", ""))
        if array_matches:
            terms = re.findall(r"\"(.*?)\"", array_matches[0])
            return terms
            
        # Second fallback: Extract any quoted terms
        terms = re.findall(r"\"(.*?)\"", keys_text)
        return terms[:5]  # Limit to 5 terms
    except Exception as e:
        print(f"Error parsing entities: {e}")
        return []

def save_intermediate_results(path, umls_res, processed_entities, entities, query):
    """Save intermediate results to a file"""
    try:
        data = {
            "umls_res": umls_res,
            "processed_entities": processed_entities,
            "entities": entities,
            "query": query,
            "timestamp": time.time()
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save data
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Saved intermediate results to {path}")
    except Exception as e:
        print(f"Error saving intermediate results: {e}")

def get_umls_keys(query, prompt, llm, save_path=None, resume=False):
    """
    Get multi-hop knowledge from UMLS with Cohere reranking with resumable processing
    
    Args:
        query: The query string
        prompt: The prompt for entity extraction
        llm: The language model for entity extraction
        save_path: Path to save intermediate results (optional)
        resume: Whether to resume from saved results (optional)
    """
    start_time = time.time()
    
    # Set default save path if not provided
    if save_path is None:
        # Create a safe filename from the query
        safe_query = re.sub(r'[^\w\s-]', '', query).strip().replace(' ', '_')[:30]
        save_path = os.path.join(RESULTS_DIR, f'umls_progress_{safe_query}.json')
    
    # Try to load saved results if resuming
    if resume and os.path.exists(save_path):
        try:
            print(f"Resuming from saved results at {save_path}")
            with open(save_path, 'r') as f:
                saved_data = json.load(f)
                umls_res = saved_data.get("umls_res", {})
                processed_entities = saved_data.get("processed_entities", [])
                entities = saved_data.get("entities", [])
                saved_query = saved_data.get("query", "")
                
                # Verify we're resuming the same query
                if saved_query != query:
                    print(f"Warning: Saved query '{saved_query}' doesn't match current query '{query}'")
                    if input("Continue anyway? (y/n): ").lower() != 'y':
                        umls_res = {}
                        processed_entities = []
                        entities = []
        except Exception as e:
            print(f"Error loading saved results: {e}")
            umls_res = {}
            processed_entities = []
            entities = []
    else:
        umls_res = {}
        processed_entities = []
        entities = []
        
    # Extract entities if not resuming or resuming with no entities
    if not entities:
        prompt_text = prompt.replace("{question}", query)
        try:
            # Extract medical entities
            print("Extracting medical entities...")
            keys_text = llm.predict(prompt_text)
            print(keys_text)
            entities = extract_medical_entities(keys_text)
            
            if not entities:
                print("No medical entities found in query")
                return ""
                
            print(f"Found {len(entities)} medical entities: {', '.join(entities)}")
                
        except Exception as e:
            print(f"Error during model processing: {e}")
            return ""
        
        # Save entities
        save_intermediate_results(save_path, umls_res, processed_entities, entities, query)

    # Multi-hop parameters
    max_hops = 3
    max_relations_per_level = {
        1: 20,
        2: 7,
        3: 3
    }
    max_relations_to_keep = {
        1: 100,
        2: 50,
        3: 20
    }
    max_top_relations = 20

    print("Processing entities...")
    for key in entities:
        # Skip already processed entities
        if key in processed_entities:
            print(f"Skipping already processed entity: {key}")
            continue
            
        entity_start = time.time()
        cuis = umls_api.search_cui(key)

        if len(cuis) == 0:
            print(f"No CUI found for '{key}'")
            processed_entities.append(key)  # Mark as processed
            save_intermediate_results(save_path, umls_res, processed_entities, entities, query)
            continue
            
        cui = cuis[0][0]
        name = cuis[0][1]
        
        print(f"Processing entity: {name} ({cui})")

        # Get definition with priority to reliable sources
        defi = ""
        definitions = umls_api.get_definitions(cui)
        if definitions:
            # Source priority (highest to lowest)
            sources = ["MSH", "NCI", "ICF", "CSP", "HPO"]
            for source in sources:
                for definition in definitions:
                    if definition.get("rootSource") == source:
                        defi = definition.get("value", "")
                        break
                if defi:
                    break

        # Get multi-hop relations with UNBIASED exploration
        print(f"Exploring multi-hop paths for {name}...")
        path_start = time.time()
        relation_paths = umls_api.get_multi_hop_paths(cui, max_hops=max_hops, max_relations_per_level=max_relations_per_level)
        path_time = time.time() - path_start
        print(f"Found {len(relation_paths)} paths in {path_time:.2f}s")
        
        rels = []
        
        if relation_paths:
            # Extract all multi-hop relation texts and organize by hop level
            multi_hop_texts = []
            path_infos = []  # Store path information for later use
            
            # Group paths by hop level for balanced representation
            paths_by_hop = {}
            for path in relation_paths:
                # Get the max hop level in this path
                max_path_hop = max(rel[2] for rel in path) if path else 0
                
                if max_path_hop not in paths_by_hop:
                    paths_by_hop[max_path_hop] = []
                paths_by_hop[max_path_hop].append(path)
            
            # Process paths by hop level
            print(f"Processing paths by hop level...")
            for hop_level, paths in paths_by_hop.items():
                # Get max relations to keep for this hop level
                max_to_keep = max_relations_to_keep.get(hop_level, 20)
                
                # Determine how many to keep from this level (proportional allocation)
                hop_sample_size = min(len(paths), max_to_keep)
                
                # Randomly sample if needed to avoid biases
                if len(paths) > hop_sample_size:
                    sampled_paths = random.sample(paths, hop_sample_size)
                else:
                    sampled_paths = paths
                
                # Process each path
                for path in sampled_paths:
                    # For single-hop paths
                    if len(path) == 1:
                        rel_text, _, _ = path[0]
                        multi_hop_texts.append(rel_text)
                        path_infos.append(path)
                    # For multi-hop paths
                    else:
                        # Join relations with ' → ' to create a path description
                        path_text = ' → '.join([rel[0] for rel in path])
                        formatted_text = f"{hop_level}-hop: {path_text}"
                        multi_hop_texts.append(formatted_text)
                        path_infos.append(path)
            
            # Stage 1: UMLSBERT embedding similarity ranking
            if multi_hop_texts:
                print(f"Stage 1: Ranking {len(multi_hop_texts)} relations with UMLSBERT...")
                embedding_start = time.time()
                
                # Get embeddings for all texts
                all_texts = [query] + multi_hop_texts
                embeddings = umlsbert.batch_encode(all_texts)
                
                # Calculate similarity scores
                query_embedding = embeddings[0]
                relation_embeddings = embeddings[1:]
                
                # Get similarity scores
                relation_scores = []
                for i, (text, path) in enumerate(zip(multi_hop_texts, path_infos)):
                    if i < len(relation_embeddings):
                        sim_score = float(get_similarity([query_embedding], [relation_embeddings[i]])[0][0])
                        relation_scores.append((sim_score, text, path))
                
                # Sort by similarity score
                relation_scores.sort(key=lambda x: x[0], reverse=True)
                
                embedding_time = time.time() - embedding_start
                print(f"Stage 1 completed in {embedding_time:.2f}s")
                
                # Take top relations for Cohere reranking
                top_count = min(200, len(relation_scores))
                top_relations = relation_scores[:top_count]
                top_texts = [rel[1] for rel in top_relations]
                
                # Stage 2: Cohere Reranking
                if top_texts:
                    print(f"Stage 2: Reranking {len(top_texts)} relations with Cohere...")
                    rerank_start = time.time()
                    
                    # Get reranked results
                    reranked_results = cohere_reranker.rerank(query, top_texts, top_n=max_top_relations)
                    
                    rerank_time = time.time() - rerank_start
                    print(f"Stage 2 completed in {rerank_time:.2f}s")
                    
                    # Process reranked results
                    for result in reranked_results:
                        text = result["text"]
                        score = result["relevance_score"]
                        rels.append((text, score))

        entity_time = time.time() - entity_start
        print(f"Entity processed in {entity_time:.2f}s")
        
        # Store results and save progress after each entity
        umls_res[cui] = {"name": name, "definition": defi, "rels": rels}
        processed_entities.append(key)  # Mark as processed
        save_intermediate_results(save_path, umls_res, processed_entities, entities, query)

    # Format context for LLM consumption
    context = "" 
    for k, v in umls_res.items():
        name = v["name"]
        definition = v["definition"]
        rels = v["rels"]
        
        # Format relations with improved readability
        rels_text = ""
        for rel in rels:
            relation_description, relevance_score = rel
            # Add confidence score to help the LLM evaluate reliability
            rels_text += f"({relation_description}) [confidence: {relevance_score:.2f}]\n"
            
        # Format entity information
        text = f"Name: {name}\nDefinition: {definition}\n"
        if rels_text:
            text += f"Relations: \n{rels_text}"
            
        context += text + "\n"
        
    if context:
        context = context[:-1]  # Remove trailing newline
    else:
        print("Warning: No UMLS context was generated")
    
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f}s")
        
    return context

# Initialize the UMLS API and models
umls_api = UMLS_API(umls_api_key)
umlsbert = UMLSBERT()
cohere_reranker = UMLS_CohereReranker(cohere_api_key)

