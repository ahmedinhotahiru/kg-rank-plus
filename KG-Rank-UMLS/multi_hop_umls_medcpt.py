import torch
import numpy as np
import re
import json
import requests
import time
import random
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque
from sentence_transformers.cross_encoder import CrossEncoder
from typing import List, Dict, Tuple, Set, Any, Optional

#--------------- LOAD ENV VARIABLES ---------------------------
from dotenv import load_dotenv
import os

# load the variables from .env into the environment
load_dotenv()
umls_api_key=os.getenv("UMLS_API_KEY")
#--------------- LOAD ENV VARIABLES ---------------------------

def get_similarity(query_vec, rel_vec):
    return cosine_similarity(query_vec, rel_vec)

class UMLSBERT:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("GanjinZero/UMLSBert_ENG")
        self.model = AutoModel.from_pretrained("GanjinZero/UMLSBert_ENG")
        self.cache = {}  # Add embedding cache

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def batch_encode(self, texts, batch_size=16):
        all_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            cache_key = hash(text)
            if cache_key in self.cache:
                all_embeddings.append(self.cache[cache_key])
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
                batch_embeddings.extend(embeddings)
            
            # Update cache and restore original order
            full_embeddings = [None] * len(texts)
            for i, orig_idx in enumerate(uncached_indices):
                embedding = batch_embeddings[i].cpu().numpy()
                self.cache[hash(uncached_texts[i])] = embedding
                if orig_idx < len(full_embeddings):
                    full_embeddings[orig_idx] = embedding
            
            # Merge cached and new embeddings
            for i in range(len(texts)):
                if i < len(full_embeddings) and full_embeddings[i] is not None:
                    all_embeddings.append(full_embeddings[i])
        
        return np.array(all_embeddings)

    def encode(self, text):
        """Encode a single text with caching"""
        cache_key = hash(text)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            model_output = self.model(**inputs)
            attention_mask = inputs["attention_mask"]
            embedding = self.mean_pooling(model_output, attention_mask)[0].cpu().numpy()
        
        self.cache[cache_key] = embedding
        return embedding

class MedCPT_CrossEncoder:
    def __init__(self):
        # Initialize MedCPT cross-encoder model
        self.model = CrossEncoder('ncbi/MedCPT-Cross-Encoder')
        self.cache = {}  # Cache for reranking results
    
    def score(self, query, texts):
        """Score multiple texts against a query with caching"""
        if not texts:
            return []
            
        # Create a cache key based on query and documents
        cache_key = hash((query, tuple(sorted(texts))))
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # Create pairs of query and texts for cross-encoder scoring
        pairs = [[query, text] for text in texts]
        
        # Score all pairs
        scores = self.model.predict(pairs)
        
        # Cache results
        self.cache[cache_key] = scores
        return scores

class UMLS_API:
    def __init__(self, apikey, version="current"):
        self.apikey = apikey
        self.version = version
        self.search_url = f"https://uts-ws.nlm.nih.gov/rest/search/{version}"
        self.content_url = f"https://uts-ws.nlm.nih.gov/rest/content/{version}"
        self.content_suffix = "/CUI/{}/{}?apiKey={}"
        
        # Add caching
        self.cui_cache = {}
        self.definition_cache = {}
        self.relation_cache = {}
        self.path_cache = {}
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

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
        return cui_results

    def get_definitions(self, cui):
        # Check cache
        if cui in self.definition_cache:
            return self.definition_cache[cui]
            
        try:
            self._rate_limit()
            suffix = self.content_suffix.format(cui, "definitions", self.apikey)
            r = requests.get(self.content_url + suffix)
            r.raise_for_status()
            r.encoding = "utf-8"
            outputs = r.json()
            
            # Update cache
            result = outputs["result"]
            self.definition_cache[cui] = result
            return result
        except Exception as e:
            print(f"Error retrieving definitions for {cui}: {e}")
            return None

    def get_relations(self, cui, pages=10):
        # Check cache
        if cui in self.relation_cache:
            return self.relation_cache[cui]
            
        all_relations = []
        try:
            for page in range(1, pages + 1):
                self._rate_limit()
                suffix = self.content_suffix.format(cui, "relations", self.apikey) + f"&pageNumber={page}"
                r = requests.get(self.content_url + suffix)
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
        return all_relations

    def get_multi_hop_paths(self, cui, max_hops=3, max_relations_per_hop=20):
        """
        Get multi-hop relations without biased prioritization
        """
        # Check cache
        cache_key = (cui, max_hops, max_relations_per_hop)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
            
        # Dictionary to store relation paths
        all_relation_paths = []
        
        # Set to track visited CUIs to avoid cycles
        visited_cuis = set([cui])
        
        # Queue for BFS with initial CUI and empty path
        queue = deque([(cui, [])])
        
        # Current hop level
        current_hop = 0
        current_level_size = 1
        next_level_size = 0
        
        while queue and current_hop < max_hops:
            # Dequeue current CUI and its path so far
            current_cui, current_path = queue.popleft()
            current_level_size -= 1
            
            # Get relations for this CUI
            relations = self.get_relations(current_cui)
            
            # If no relations found, continue to next item in queue
            if not relations:
                continue
            
            # Take random sample to avoid biases and limit growth - UNBIASED SELECTION
            if len(relations) > max_relations_per_hop:
                pruned_relations = random.sample(relations, max_relations_per_hop)
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
                new_path = current_path + [(rel_text, relation)]
                
                # Add the complete path for ranking
                if new_path:  # Only add non-empty paths
                    all_relation_paths.append(new_path)
                
                # Continue BFS only if this CUI hasn't been visited yet (avoid cycles)
                if related_cui not in visited_cuis:
                    visited_cuis.add(related_cui)
                    queue.append((related_cui, new_path))
                    next_level_size += 1
            
            # Check if we've completed the current level
            if current_level_size == 0:
                current_hop += 1
                current_level_size = next_level_size
                next_level_size = 0
        
        # Update cache
        self.path_cache[cache_key] = all_relation_paths
        return all_relation_paths

umls_api = UMLS_API(umls_api_key)
umlsbert = UMLSBERT()
medcpt_cross_encoder = MedCPT_CrossEncoder()

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

def get_umls_keys(query, prompt, llm):
    """
    Get multi-hop knowledge from UMLS with two-stage reranking
    """
    start_time = time.time()
    umls_res = {}
    prompt = prompt.replace("{question}", query)

    try:
        # Extract medical entities
        print("Extracting medical entities...")
        keys_text = llm.predict(prompt)
        print(keys_text)
        entities = extract_medical_entities(keys_text)
        
        if not entities:
            print("No medical entities found in query")
            return ""
            
        print(f"Found {len(entities)} medical entities: {', '.join(entities)}")
            
    except Exception as e:
        print(f"Error during model processing: {e}")
        return "" 

    # Multi-hop parameters
    max_hops = 3  # Maximum number of hops
    max_relations_per_hop = 20  # Maximum relations per concept at each hop
    max_initial_filter = 200  # Maximum paths to consider for first-stage ranking
    max_top_relations = 20  # Maximum relations to keep after final reranking

    print("Processing entities...")
    for key in entities:
        print(f"Processing entity: {key}")
        entity_start = time.time()
        cuis = umls_api.search_cui(key)

        if len(cuis) == 0:
            print(f"No CUI found for {key}")
            continue
            
        cui = cuis[0][0]
        name = cuis[0][1]
        print(f"Found CUI: {cui} ({name})")

        # Get definition with priority to reliable sources
        defi = ""
        definitions = umls_api.get_definitions(cui)
        if definitions:
            for source_priority in ["MSH", "NCI", "ICF", "CSP", "HPO"]:
                for definition in definitions:
                    if definition["rootSource"] == source_priority:
                        defi = definition["value"]
                        break
                if defi:
                    break

        # Get multi-hop relations with UNBIASED exploration
        print(f"Exploring multi-hop paths for {name}...")
        paths_start = time.time()
        relation_paths = umls_api.get_multi_hop_paths(cui, max_hops=max_hops, max_relations_per_hop=max_relations_per_hop)
        print(f"Found {len(relation_paths)} paths in {time.time() - paths_start:.2f}s")
        
        rels = []
        
        if relation_paths:
            # Extract all multi-hop relation texts and organize by hop level
            multi_hop_texts = []
            multi_hop_paths = []  # Store original paths for later processing
            
            # Group paths by hop level for balanced representation
            paths_by_hop = {}
            for path in relation_paths:
                hop_level = len(path)
                if hop_level not in paths_by_hop:
                    paths_by_hop[hop_level] = []
                paths_by_hop[hop_level].append(path)
            
            # Process paths by hop level
            for hop_level, paths in paths_by_hop.items():
                hop_texts = []
                for path in paths:
                    # For single relation in path
                    if len(path) == 1:
                        hop_texts.append((path[0][0], path))  # (text, path)
                    # For multi-hop paths (2+ relations)
                    else:
                        # Join relations with ' → ' to create a path description
                        path_text = ' → '.join([step[0] for step in path])
                        hop_text = f"{hop_level}-hop: {path_text}"
                        hop_texts.append((hop_text, path))
                
                # Take a balanced sample of paths from each hop level
                sample_size = min(len(hop_texts), 100 // len(paths_by_hop))
                if len(hop_texts) > sample_size:
                    sampled_texts = random.sample(hop_texts, sample_size)
                else:
                    sampled_texts = hop_texts
                
                multi_hop_texts.extend([text for text, _ in sampled_texts])
                multi_hop_paths.extend([path for _, path in sampled_texts])
            
            if multi_hop_texts:
                print(f"Processing {len(multi_hop_texts)} relation texts...")
                
                # STAGE 1: Initial ranking with UMLSBERT embeddings
                print("Stage 1: Embedding-based ranking...")
                embedding_start = time.time()
                
                # Encode the query and all multi-hop relation texts
                relation_texts = [query] + multi_hop_texts
                embeddings = umlsbert.batch_encode(relation_texts)
                
                # Calculate similarity scores
                query_embedding = embeddings[0]
                relation_embeddings = embeddings[1:]
                
                # Calculate similarity and pair with relation texts and original paths
                relation_scores = [(get_similarity([query_embedding], [rel_emb])[0][0], text, path) 
                                  for rel_emb, text, path in zip(relation_embeddings, multi_hop_texts, multi_hop_paths)]
                
                # Sort by similarity score (descending)
                relation_scores.sort(key=lambda x: x[0], reverse=True)
                
                embedding_time = time.time() - embedding_start
                print(f"Stage 1 completed in {embedding_time:.2f}s")
                
                # Take top relations for second-stage reranking
                top_relations = relation_scores[:max_initial_filter]
                top_texts = [rel[1] for rel in top_relations]
                
                # STAGE 2: MedCPT CrossEncoder reranking
                if top_texts:
                    print(f"Stage 2: MedCPT reranking of {len(top_texts)} relations...")
                    rerank_start = time.time()
                    
                    # Get CrossEncoder scores
                    cross_encoder_scores = medcpt_cross_encoder.score(query, top_texts)
                    
                    # Combine with original data
                    reranked_relations = []
                    for i, (score, (_, text, path)) in enumerate(zip(cross_encoder_scores, top_relations)):
                        if i < len(top_texts):  # Safety check
                            reranked_relations.append((score, text, path))
                    
                    # Sort by cross-encoder score
                    reranked_relations.sort(key=lambda x: x[0], reverse=True)
                    
                    rerank_time = time.time() - rerank_start
                    print(f"Stage 2 completed in {rerank_time:.2f}s")
                    
                    # Take top results after reranking
                    final_relations = reranked_relations[:max_top_relations]
                    
                    # Process final relations for output
                    for score, text, path in final_relations:
                        # For 1-hop relations
                        if len(path) == 1:
                            # Extract components from the relation
                            relation = path[0][1]  # Get the original relation object
                            from_name = relation.get("relatedFromIdName", "")
                            relation_label = relation.get("additionalRelationLabel", "").replace("_", " ")
                            to_name = relation.get("relatedIdName", "")
                            
                            # Add to results with score
                            rels.append((text, score))
                        # For multi-hop relations
                        else:
                            # Get start and end points
                            start_relation = path[0][1]
                            end_relation = path[-1][1]
                            start_entity = start_relation.get("relatedFromIdName", "")
                            end_entity = end_relation.get("relatedIdName", "")
                            
                            # Format as path with score
                            rels.append((text, score))
        
        entity_time = time.time() - entity_start
        print(f"Entity processed in {entity_time:.2f}s")
        umls_res[cui] = {"name": name, "definition": defi, "rels": rels}

    # Format context
    context = "" 
    for k, v in umls_res.items():
        name = v["name"]
        definition = v["definition"]
        rels = v["rels"]
        
        # Format relations
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
    
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f}s")
        
    return context