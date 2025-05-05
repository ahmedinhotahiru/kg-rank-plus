import torch
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
from collections import deque
import concurrent.futures
from functools import lru_cache

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
umls_api_key = os.getenv("UMLS_API_KEY")

# Persistent cache file paths
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)
EMBEDDING_CACHE_FILE = os.path.join(CACHE_DIR, 'embedding_cache.pkl')
CUI_CACHE_FILE = os.path.join(CACHE_DIR, 'cui_cache.pkl')
DEF_CACHE_FILE = os.path.join(CACHE_DIR, 'definition_cache.pkl')
REL_CACHE_FILE = os.path.join(CACHE_DIR, 'relation_cache.pkl')

class UMLSBERT:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.embed_cache = self._load_cache(EMBEDDING_CACHE_FILE, {})
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
        self._load_model()  # Ensure model is loaded
        
        # Remove duplicates to avoid redundant processing
        unique_texts = list(set(texts))
        text_to_idx = {text: idx for idx, text in enumerate(texts)}
        
        # Check cache for all texts
        uncached_texts = []
        for text in unique_texts:
            if hash(text) not in self.embed_cache:
                uncached_texts.append(text)
        
        # Process uncached texts
        if uncached_texts:
            all_batch_embeddings = []
            
            for i in range(0, len(uncached_texts), batch_size):
                batch_texts = uncached_texts[i:i + batch_size]
                
                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    model_output = self.model(**inputs)
                    attention_mask = inputs["attention_mask"]
                    batch_embeddings = self.mean_pooling(model_output, attention_mask).numpy()
                    
                all_batch_embeddings.append(batch_embeddings)
                
                # Update cache
                for j, text in enumerate(batch_texts):
                    self.embed_cache[hash(text)] = batch_embeddings[j]
            
            # Save updated cache periodically
            if len(uncached_texts) > 10:  # Only save if we've added a significant number
                self._save_cache(EMBEDDING_CACHE_FILE, self.embed_cache)
        
        # Construct result array using cached embeddings
        result = np.zeros((len(texts), next(iter(self.embed_cache.values())).shape[0]))
        
        for i, text in enumerate(texts):
            result[i] = self.embed_cache[hash(text)]
            
        return result


class UMLS_API:
    def __init__(self, apikey, version="current"):
        self.apikey = apikey
        self.version = version
        self.search_url = f"https://uts-ws.nlm.nih.gov/rest/search/{version}"
        self.content_url = f"https://uts-ws.nlm.nih.gov/rest/content/{version}"
        self.content_suffix = "/CUI/{}/{}?apiKey={}"
        
        # Load persistent caches
        self.cui_cache = self._load_cache(CUI_CACHE_FILE, {})
        self.definition_cache = self._load_cache(DEF_CACHE_FILE, {})
        self.relation_cache = self._load_cache(REL_CACHE_FILE, {})
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1
        
        # Concurrent API calls
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self.pending_futures = []

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
        """Search for CUI by term with caching"""
        if query in self.cui_cache:
            return self.cui_cache[query]
            
        cui_results = []
        try:
            self._rate_limit()
            query_params = {"string": query, "apiKey": self.apikey, "pageNumber": 1, "pageSize": 1}
            response = requests.get(self.search_url, params=query_params)
            response.raise_for_status()
            items = response.json()["result"]["results"]
            if items:
                for result in items:
                    cui_results.append((result["ui"], result["name"]))
                    
            # Update cache
            self.cui_cache[query] = cui_results
            if len(self.cui_cache) % 20 == 0:  # Periodically save cache
                self._save_cache(CUI_CACHE_FILE, self.cui_cache)
                
        except Exception as e:
            print(f"Error searching for CUI '{query}': {e}")
            
        return cui_results

    def get_definitions(self, cui):
        """Get definitions for a CUI, handling cases where definitions don't exist"""
        if cui in self.definition_cache:
            return self.definition_cache[cui]
            
        try:
            self._rate_limit()
            suffix = self.content_suffix.format(cui, "definitions", self.apikey)
            response = requests.get(self.content_url + suffix)
            
            # If we get a 404, it means there are no definitions for this CUI
            if response.status_code == 404:
                self.definition_cache[cui] = []
                return []
                
            # For other errors, raise the exception
            response.raise_for_status()
            
            definitions = response.json()["result"]
            self.definition_cache[cui] = definitions
            
            # Periodically save cache
            if len(self.definition_cache) % 20 == 0:
                self._save_cache(DEF_CACHE_FILE, self.definition_cache)
                
            return definitions
            
        except requests.exceptions.HTTPError as e:
            if "404" in str(e):
                self.definition_cache[cui] = []
                return []
            else:
                print(f"Error retrieving definitions for {cui}: {e}")
                return []
        except Exception as e:
            print(f"Error retrieving definitions for {cui}: {e}")
            return []

    def get_relations(self, cui, max_pages=5):  # Reduced from 10 to 5 pages
        """Get relations for a CUI with caching"""
        if cui in self.relation_cache:
            return self.relation_cache[cui]
            
        all_relations = []
        try:
            for page in range(1, max_pages + 1):
                self._rate_limit()
                suffix = self.content_suffix.format(cui, "relations", self.apikey) + f"&pageNumber={page}"
                response = requests.get(self.content_url + suffix)
                response.raise_for_status()
                page_relations = response.json().get("result", [])
                if not page_relations:
                    break
                all_relations.extend(page_relations)
                
            # Update cache
            self.relation_cache[cui] = all_relations
            if len(self.relation_cache) % 10 == 0:  # Periodically save cache
                self._save_cache(REL_CACHE_FILE, self.relation_cache)
                
        except Exception as e:
            print(f"Error retrieving relations for {cui}: {e}")
            
        return all_relations
        
    def prefetch_batch(self, queries):
        """Prefetch multiple CUIs in parallel"""
        # Submit all queries as futures
        futures = []
        for query in queries:
            if query not in self.cui_cache:
                futures.append(self.executor.submit(self.search_cui, query))
                
        # Wait for all to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Just to handle any exceptions
            except Exception as e:
                print(f"Error in prefetch: {e}")


# Optimized similarity calculation
@lru_cache(maxsize=10000)
def compute_similarity_key(i, j):
    """Generates a unique key for caching similarity calculations"""
    return f"{hash(str(i))}-{hash(str(j))}"

class SimilarityCache:
    def __init__(self):
        self.cache = {}
        
    def get_similarity(self, vec1, vec2):
        """Get similarity with caching based on vector hashes"""
        key = compute_similarity_key(hash(vec1.tobytes()), hash(vec2.tobytes()))
        if key not in self.cache:
            self.cache[key] = float(cosine_similarity([vec1], [vec2])[0][0])
        return self.cache[key]

sim_cache = SimilarityCache()

def get_similarity(query_vec, rel_vec):
    """Wrapper for cosine similarity calculation"""
    return sim_cache.get_similarity(query_vec[0], rel_vec[0])

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

def precompute_similarities(query_embedding, relation_embeddings):
    """Precompute all similarities to avoid redundant calculations"""
    # Query-relation similarities
    query_similarities = []
    for i, rel_emb in enumerate(relation_embeddings):
        sim = float(cosine_similarity([query_embedding], [rel_emb])[0][0])
        query_similarities.append((i, sim))
    
    # Relation-relation similarities (only compute once, store in matrix)
    n = len(relation_embeddings)
    rel_similarities = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):  # Only compute upper triangle
            sim = float(cosine_similarity([relation_embeddings[i]], [relation_embeddings[j]])[0][0])
            rel_similarities[i, j] = sim
            rel_similarities[j, i] = sim  # Matrix is symmetric
    
    return query_similarities, rel_similarities

def apply_mmr_ranking(query_embedding, relation_embeddings, selected_indices=None, lambda_param=0.6, n=1, 
                     precomputed_sims=None):
    """
    Apply Maximal Marginal Relevance (MMR) for relation ranking with optimized similarity calculations
    """
    if selected_indices is None:
        selected_indices = []
        
    if len(relation_embeddings) == 0:
        return []
    
    # Use precomputed similarities if provided
    query_sims, rel_sims = precomputed_sims if precomputed_sims else (None, None)
    
    # For the first selection, just pick the most relevant
    if not selected_indices:
        if query_sims:
            # Use precomputed similarities
            idx = max(query_sims, key=lambda x: x[1])[0]
        else:
            # Calculate on-the-fly
            relevance_scores = [float(cosine_similarity([query_embedding], [rel_emb])[0][0]) 
                              for rel_emb in relation_embeddings]
            idx = np.argmax(relevance_scores)
        return [idx]
        
    # Calculate remaining candidates
    remaining_indices = [i for i in range(len(relation_embeddings)) if i not in selected_indices]
    
    # Apply MMR to select next n indices
    mmr_selected = []
    
    for _ in range(min(n, len(remaining_indices))):
        max_mmr_score = -float('inf')
        max_mmr_idx = -1
        
        for idx in remaining_indices:
            if idx in mmr_selected:
                continue
                
            # Calculate relevance (similarity to query)
            if query_sims:
                # Use precomputed query similarity
                relevance = next(sim for i, sim in query_sims if i == idx)
            else:
                # Calculate on-the-fly
                relevance = float(cosine_similarity([query_embedding], [relation_embeddings[idx]])[0][0])
            
            # Calculate diversity (dissimilarity to already selected items)
            if selected_indices + mmr_selected:
                if rel_sims is not None:
                    # Use precomputed relation similarities
                    max_similarity = max(rel_sims[idx, sel_idx] for sel_idx in selected_indices + mmr_selected)
                else:
                    # Calculate on-the-fly
                    max_similarity = max([float(cosine_similarity([relation_embeddings[idx]], 
                                                            [relation_embeddings[sel_idx]])[0][0])
                                       for sel_idx in selected_indices + mmr_selected])
            else:
                max_similarity = 0
                
            # Calculate MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
            
            if mmr_score > max_mmr_score:
                max_mmr_score = mmr_score
                max_mmr_idx = idx
        
        if max_mmr_idx >= 0:
            mmr_selected.append(max_mmr_idx)
            remaining_indices.remove(max_mmr_idx)
        else:
            break
            
    return mmr_selected

def get_umls_keys(query, prompt, llm):
    """
    Get multi-hop knowledge from UMLS with MMR ranking - optimized version
    """
    start_time = time.time()
    umls_res = {}
    prompt = prompt.replace("{question}", query)

    try:
        # Extract medical entities
        keys_text = llm.predict(prompt)
        print(keys_text)
        entities = extract_medical_entities(keys_text)
        
        if not entities:
            print("No medical entities found in query")
            return ""
            
    except Exception as e:
        print(f"Error during model processing: {e}")
        return "" 

    print(f"Found {len(entities)} medical entities: {', '.join(entities)}")
    
    # Maximum entities to process (performance optimization)
    entities = entities[:3]  # Only process top 3 entities
    
    # Get query embedding for MMR calculations
    query_embedding = umlsbert.batch_encode([query])[0]

    # Configurable parameters with optimal defaults
    max_hops = 3  # Maximum hop level to explore
    max_relations_per_level = {
        1: 20,  # Level 1: Expand top 20 concepts
        2: 7,   # Level 2: Expand top 7 concepts
        3: 3    # Level 3: Expand top 3 concepts
    }
    max_relations_to_keep = {
        1: 100,  # Keep top 100 1-hop relations
        2: 50,   # Keep top 50 2-hop relations
        3: 20    # Keep top 20 3-hop relations
    }
    max_top_relations = 20  # Maximum final relations to return after MMR ranking

    print(f"Processing entities...")
    
    # Prefetch CUIs for all entities to parallelize API calls
    umls_api.prefetch_batch(entities)

    for key in entities:
        process_start_time = time.time()
        cuis = umls_api.search_cui(key)

        if len(cuis) == 0:
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
                    if definition["rootSource"] == source:
                        defi = definition["value"]
                        break
                if defi:
                    break

        # Multi-hop relation implementation with MMR ranking
        rels = []
        all_relations = []        # Store all relations with their hop levels
        relation_texts = []       # Store relation texts for embedding
        
        # Track visited concepts to avoid cycles
        visited_concepts = set([name.lower()])
        
        # Queue for BFS traversal: (concept_name, cui, hop_level, path)
        queue = deque([(name, cui, 1, [])])
        
        # Continue BFS until queue is empty or max hop level is reached
        print(f"Exploring knowledge graph...")
        exploration_start = time.time()
        while queue:
            concept_name, concept_cui, hop_level, path = queue.popleft()
            
            # Stop if we've reached the maximum hop level
            if hop_level > max_hops:
                continue
                
            # Early stopping if we have enough relations already
            if len(all_relations) > 300:  # Added early stopping
                print(f"Early stopping: collected {len(all_relations)} relations")
                break
                
            # Get relations for current concept
            relations = umls_api.get_relations(concept_cui)
            
            # Skip if no relations found
            if not relations:
                continue
            
            # MODIFICATION: Limit relations based on hop level with relevance-based selection
            max_to_expand = max_relations_per_level.get(hop_level, 1)
            if len(relations) > max_to_expand:
                # Get relation texts for embedding
                relation_texts_to_score = []
                for rel in relations:
                    related_from = rel.get('relatedFromIdName', '')
                    relation_label = rel.get('additionalRelationLabel', '').replace('_', ' ')
                    related_to = rel.get('relatedIdName', '')
                    relation_texts_to_score.append(f"{related_from} {relation_label} {related_to}")
                
                # Get embeddings and score against query
                relation_embeddings = umlsbert.batch_encode(relation_texts_to_score)
                
                # Score relations
                scores = []
                for i, embedding in enumerate(relation_embeddings):
                    sim_score = float(cosine_similarity([query_embedding], [embedding])[0][0])
                    scores.append((i, sim_score))
                
                # Select top relations by relevance
                scores.sort(key=lambda x: x[1], reverse=True)
                selected_indices = [idx for idx, _ in scores[:max_to_expand]]
                pruned_relations = [relations[i] for i in selected_indices]
            else:
                pruned_relations = relations
            
            # Process relations from this hop level
            for rel in pruned_relations:
                related_from = rel.get('relatedFromIdName', '')
                relation_label = rel.get('additionalRelationLabel', '').replace('_', ' ')
                related_to = rel.get('relatedIdName', '')
                
                # Skip if missing essential information
                if not related_to:
                    continue
                
                # Skip if we've already visited this concept
                if related_to.lower() in visited_concepts:
                    continue
                    
                # Mark as visited
                visited_concepts.add(related_to.lower())
                
                # Create current relation tuple
                current_rel = (related_from, relation_label, related_to)
                
                # Create full path including this relation
                full_path = path + [current_rel]
                
                # Format text representation of this relation
                if hop_level == 1:
                    # Direct 1-hop relation
                    rel_text = f"{related_from} {relation_label} {related_to}"
                else:
                    # Multi-hop relation with path
                    path_segments = []
                    for p in full_path:
                        path_segments.append(f"{p[0]} {p[1]} {p[2]}")
                    rel_text = f"{hop_level}-hop: " + " → ".join(path_segments)
                
                # Store relation text
                relation_texts.append(rel_text)
                
                # Store relation info with hop level and full path
                all_relations.append((hop_level, full_path, rel, rel_text))
                
                # Continue exploration from this concept
                target_cuis = umls_api.search_cui(related_to)
                if target_cuis:
                    target_cui = target_cuis[0][0]  # Use first match
                    queue.append((related_to, target_cui, hop_level + 1, full_path))
        
        exploration_time = time.time() - exploration_start
        print(f"Exploration completed in {exploration_time:.2f}s. Found {len(all_relations)} relations.")
        
        # Now apply MMR on gathered relations
        if all_relations:
            # Group relations by hop level for more balanced selection
            relations_by_hop = {}
            for i, rel_info in enumerate(all_relations):
                hop_level = rel_info[0]
                if hop_level not in relations_by_hop:
                    relations_by_hop[hop_level] = []
                relations_by_hop[hop_level].append((i, rel_info[3]))  # (index, text)
            
            # MODIFICATION: Filter top relations by relevance for each hop level
            filtered_indices = []
            
            print(f"Filtering relations by relevance...")
            filtering_start = time.time()
            for hop_level, rel_infos in relations_by_hop.items():
                max_to_keep = max_relations_to_keep.get(hop_level, 20)
                
                if len(rel_infos) > max_to_keep:
                    # Get embeddings for this hop level
                    hop_texts = [text for _, text in rel_infos]
                    hop_embeddings = umlsbert.batch_encode(hop_texts)
                    
                    # Calculate relevance scores
                    relevance_scores = [float(cosine_similarity([query_embedding], [emb])[0][0]) 
                                      for emb in hop_embeddings]
                    
                    # Sort by relevance
                    hop_indices_scores = [(rel_infos[i][0], score) for i, score in enumerate(relevance_scores)]
                    hop_indices_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    # Keep top relations for this hop level
                    filtered_indices.extend([idx for idx, _ in hop_indices_scores[:max_to_keep]])
                else:
                    # Keep all if fewer than max
                    filtered_indices.extend([idx for idx, _ in rel_infos])
            
            filtering_time = time.time() - filtering_start
            print(f"Filtering completed in {filtering_time:.2f}s. Kept {len(filtered_indices)} relations.")
            
            # Filter relations and texts
            if filtered_indices:  # Check if we have filtered indices
                filtered_relations = [all_relations[i] for i in filtered_indices]
                filtered_texts = [relation_texts[i] for i in filtered_indices]
            else:
                filtered_relations = all_relations
                filtered_texts = relation_texts
                
            # Get embeddings for filtered relations
            print(f"Computing embeddings for {len(filtered_texts)} relations...")
            embedding_start = time.time()
            relation_embeddings = umlsbert.batch_encode(filtered_texts)
            embedding_time = time.time() - embedding_start
            print(f"Embeddings computed in {embedding_time:.2f}s.")
            
            # Precompute similarities for MMR (optimization)
            print(f"Precomputing similarities...")
            precompute_start = time.time()
            precomputed_sims = precompute_similarities(query_embedding, relation_embeddings)
            precompute_time = time.time() - precompute_start
            print(f"Similarities precomputed in {precompute_time:.2f}s.")
            
            # Apply MMR to select relations iteratively
            print(f"Applying MMR to select final relations...")
            mmr_start = time.time()
            
            # Initialize with empty selection
            selected_indices = []
            lambda_decay = 0.05  # How much to decrease lambda each iteration
            lambda_param = 0.8   # Start with high relevance weight
            
            # Select top relations with MMR
            batch_size = 3  # Process in batches of 3 for efficiency
            
            while len(selected_indices) < max_top_relations and len(selected_indices) < len(relation_embeddings):
                # Decrease lambda slightly each iteration to increase diversity
                current_lambda = max(0.5, lambda_param - len(selected_indices) * lambda_decay)
                
                # Select next relation with MMR in batches
                next_indices = apply_mmr_ranking(
                    query_embedding, 
                    relation_embeddings, 
                    selected_indices,
                    lambda_param=current_lambda,
                    n=batch_size,
                    precomputed_sims=precomputed_sims
                )
                
                if not next_indices:
                    break
                    
                selected_indices.extend(next_indices)
            
            mmr_time = time.time() - mmr_start
            print(f"MMR completed in {mmr_time:.2f}s. Selected {len(selected_indices)} relations.")
            
            # Process selected relations
            print(f"Formatting final relations...")
            for idx in selected_indices:
                if idx < len(filtered_relations):  # Safety check
                    hop_level, full_path, _, _ = filtered_relations[idx]
                    
                    if hop_level == 1:
                        # Process 1-hop relation directly
                        from_name, relation_label, to_name = full_path[0]
                        rels.append((from_name, relation_label, to_name))
                        
                    else:
                        # For multi-hop relations, create a condensed representation
                        start_name = full_path[0][0]
                        end_name = full_path[-1][2]
                        path_description = f"{hop_level}-hop: "
                        
                        # Add relation types in the path
                        relation_types = []
                        for path_segment in full_path:
                            relation_types.append(path_segment[1])
                            
                        path_description += " → ".join(relation_types)
                        
                        rels.append((start_name, path_description, end_name))

        entity_process_time = time.time() - process_start_time
        print(f"Entity processed in {entity_process_time:.2f}s.")
        umls_res[cui] = {"name": name, "definition": defi, "rels": rels}

    # Format context for LLM consumption
    context = "" 
    for k, v in umls_res.items():
        name = v["name"]
        definition = v["definition"]
        rels = v["rels"]
        
        # Format relations with improved readability
        rels_text = ""
        for rel in rels:
            rel_0 = rel[0] if rel[0] is not None else ""
            rel_1 = rel[1] if rel[1] is not None else ""
            rel_2 = rel[2] if rel[2] is not None else ""
            
            # Format based on hop count
            if "hop:" in rel_1:
                # Multi-hop relation
                hop_count = rel_1.split("-")[0]
                rels_text += f"({rel_0} through {hop_count} steps to {rel_2}: {rel_1})\n"
            else:
                # Direct relation
                rels_text += f"({rel_0} {rel_1} {rel_2})\n"
                
        # Format entity information
        text = f"Name: {name}\nDefinition: {definition}\n"
        if rels_text:
            text += f"Relations: \n{rels_text}"

        context += text + "\n"
        
    if context:
        context = context[:-1]  # Remove trailing newline
    
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f}s.")
    
    return context

umls_api = UMLS_API(umls_api_key)
umlsbert = UMLSBERT()