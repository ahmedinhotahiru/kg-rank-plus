import torch
import numpy as np
import re
import json
import requests
import time
import random
import os
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
umls_api_key = os.getenv("UMLS_API_KEY")

class UMLSBERT:
    """
    UMLS-specific BERT model for encoding medical texts.
    Includes caching for efficient reuse of embeddings.
    """
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("GanjinZero/UMLSBert_ENG")
        self.model = AutoModel.from_pretrained("GanjinZero/UMLSBert_ENG")
        self.embed_cache = {}  # Simple memory cache

    def mean_pooling(self, model_output, attention_mask):
        """Calculate mean pooling from token embeddings with attention mask."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def batch_encode(self, texts, batch_size=16):
        """
        Encode texts in batches with caching for efficiency.
        
        Args:
            texts: List of strings to encode
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings
        """
        all_embeddings = []
        # Check cache first
        cached_texts = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = hash(text)
            if cache_key in self.embed_cache:
                cached_texts.append((i, self.embed_cache[cache_key]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Process all new texts in batches
        if uncached_texts:
            for i in range(0, len(uncached_texts), batch_size):
                batch_texts = uncached_texts[i:i+batch_size]
                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    model_output = self.model(**inputs)
                    attention_mask = inputs["attention_mask"]
                    batch_embeddings = self.mean_pooling(model_output, attention_mask)
                
                # Cache the results
                for j, text in enumerate(batch_texts):
                    cache_key = hash(text)
                    embedding = batch_embeddings[j].numpy()
                    self.embed_cache[cache_key] = embedding
                    cached_texts.append((uncached_indices[i+j], embedding))
        
        # Sort by original index and extract embeddings
        cached_texts.sort(key=lambda x: x[0])
        all_embeddings = [embedding for _, embedding in cached_texts]
            
        return np.array(all_embeddings)

class UMLS_API:
    """API client for the Unified Medical Language System."""
    def __init__(self, apikey, version="current"):
        self.apikey = apikey
        self.version = version
        self.search_url = f"https://uts-ws.nlm.nih.gov/rest/search/{version}"
        self.content_url = f"https://uts-ws.nlm.nih.gov/rest/content/{version}"
        self.content_suffix = "/CUI/{}/{}?apiKey={}"
        self.cui_cache = {}
        self.definition_cache = {}
        self.relation_cache = {}
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

    def _rate_limit(self):
        """Apply rate limiting to API requests."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def search_cui(self, query):
        """
        Search for a concept's CUI by text query.
        
        Args:
            query: String query for concept lookup
            
        Returns:
            List of (CUI, name) tuples
        """
        # Check cache
        if query in self.cui_cache:
            return self.cui_cache[query]
            
        cui_results = []
        try:
            # self._rate_limit()
            query_params = {"string": query, "apiKey": self.apikey, "pageNumber": 1, "pageSize": 1}
            response = requests.get(self.search_url, params=query_params)
            response.raise_for_status()
            print(response.url)
            items = response.json()["result"]["results"]
            
            if items:
                for result in items:
                    cui_results.append((result["ui"], result["name"]))
                    
            # Update cache
            self.cui_cache[query] = cui_results
                
        except Exception as e:
            print(f"Error searching for CUI: {e}")
            
        return cui_results

    def get_definitions(self, cui):
        """
        Get definitions for a concept by CUI.
        
        Args:
            cui: Concept Unique Identifier 
            
        Returns:
            List of definition objects
        """
        # Check cache
        if cui in self.definition_cache:
            return self.definition_cache[cui]
            
        try:
            # self._rate_limit()
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
            return definitions
            
        except requests.exceptions.HTTPError as e:
            if "404" in str(e):
                print(f"No definitions found for {cui} (normal for some concepts)")
                self.definition_cache[cui] = []
                return []
            else:
                print(f"Error retrieving definitions for {cui}: {e}")
                return []
        except Exception as e:
            print(f"Error retrieving definitions for {cui}: {e}")
            return []

    def get_relations(self, cui, max_pages=5):
        """
        Get relations for a concept by CUI.
        
        Args:
            cui: Concept Unique Identifier
            max_pages: Maximum pages to retrieve (default: 20 as per original paper)
            
        Returns:
            List of relation objects
        """
        # Check cache
        if cui in self.relation_cache:
            return self.relation_cache[cui]
            
        all_relations = []
        try:
            for page in range(1, max_pages + 1):
                # self._rate_limit()
                suffix = self.content_suffix.format(cui, "relations", self.apikey) + f"&pageNumber={page}"
                response = requests.get(self.content_url + suffix)
                response.raise_for_status()
                page_relations = response.json().get("result", [])
                if not page_relations:
                    break
                all_relations.extend(page_relations)
                
            # Update cache
            self.relation_cache[cui] = all_relations
                
        except Exception as e:
            print(f"Error retrieving relations for {cui}: {e}")
            
        return all_relations

def get_similarity(query_vec, rel_vec):
    """Calculate cosine similarity between vectors."""
    return cosine_similarity(query_vec, rel_vec)

def extract_medical_entities(keys_text):
    """Extract medical entities with JSON parsing."""
    try:
        pattern = r"\{(.*?)\}"
        matches = re.findall(pattern, keys_text.replace("\n", ""))
        if not matches:
            raise ValueError("No medical terminologies returned by the model.")
        
        keys_dict = json.loads("{" + matches[0] + "}")
        if "medical terminologies" not in keys_dict or not keys_dict["medical terminologies"]:
            raise ValueError("Model did not return expected 'medical terminologies' key.")
        
        return keys_dict["medical terminologies"]
    except Exception as e:
        print(f"Error parsing medical entities: {e}")
        return []

def get_umls_keys(query, answer, prompt, llm):
    """
    Multi-hop knowledge retrieval from UMLS with answer expansion.
    
    Args:
        query: The user's medical question
        answer: The preliminary answer generated by the LLM
        prompt: Prompt for entity extraction
        llm: Language model for entity extraction
        
    Returns:
        String context with medical entity definitions and relations
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
        
        # Create expanded query with the answer - key aspect of answer expansion method
        extended_query = query + " " + answer
        print(f"Created expanded query ({len(extended_query.split())} words)")
        
    except Exception as e:
        print(f"Error during model processing: {e}")
        return "" 

    # Get expanded query embedding
    print("Computing embedding for expanded query...")
    query_embedding_start = time.time()
    expanded_embedding = umlsbert.batch_encode([extended_query])[0]
    print(f"Query embedding computed in {time.time() - query_embedding_start:.2f}s")

    # Multi-hop parameters - INCREASED to reduce excessive pruning
    max_hops = 3  # Maximum hop level to explore
    max_relations_per_level = {
        1: 20,  # Level 1: Expand top 50 concepts (increased from 20)
        2: 7,  # Level 2: Expand top 20 concepts (increased from 7)
        3: 3   # Level 3: Expand top 10 concepts (increased from 3)
    }
    max_relations_to_keep = {
        2: 50,   # Keep top 50 2-hop relations
        3: 20    # Keep top 30 3-hop relations
    }
    max_top_relations = 20  # Maximum final relations to return per entity (as in original paper)

    # Process each entity
    for key in entities:
        process_start = time.time()
        cuis = umls_api.search_cui(key)

        if len(cuis) == 0:
            print(f"No CUI found for '{key}'")
            continue
        
        cui = cuis[0][0]
        name = cuis[0][1]
        
        print(f"Processing entity: {name} ({cui})")

        # Get definition with source prioritization
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
            
            # If no matching source found, use the first definition
            if not defi and definitions:
                defi = definitions[0].get("value", "")

        # Multi-hop implementation
        all_relations = []  # Store all relations across all hops
        
        # Track visited concepts to avoid cycles
        visited_concepts = set([name.lower()])
        
        # Queue for BFS traversal: (concept_name, cui, hop_level, path)
        queue = deque([(name, cui, 1, [])])
        
        # Begin exploration
        print(f"Exploring knowledge graph...")
        while queue:
            concept_name, concept_cui, hop_level, path = queue.popleft()
            
            # Stop if we've reached the maximum hop level
            if hop_level > max_hops:
                continue
                
            # Get relations for current concept
            relations = umls_api.get_relations(concept_cui, max_pages=5)
            
            # Skip if no relations found
            if not relations:
                continue
            
            # Process relations based on hop level
            if hop_level == 1:
                # For first hop, process ALL relations
                relation_texts = []
                valid_relations = []
                
                for rel in relations:
                    related_from = rel.get('relatedFromIdName', '')
                    relation_label = rel.get('additionalRelationLabel', '').replace('_', ' ')
                    related_to = rel.get('relatedIdName', '')
                    
                    # Skip if missing essential information
                    if not related_from or not relation_label or not related_to:
                        continue
                    
                    # Create text representation
                    rel_text = f"{related_from} {relation_label} {related_to}"
                    
                    # Add to first hop relations
                    relation_texts.append(rel_text)
                    valid_relations.append(rel)
                    
                    # Create path for this relation
                    relation_path = [(related_from, relation_label, related_to)]
                    
                    # Store ALL first-hop relations
                    all_relations.append((hop_level, relation_path, rel_text))
                
                # Score first-hop relations to determine which to expand
                if relation_texts:
                    relation_embeddings = umlsbert.batch_encode(relation_texts)
                    
                    # Score based on similarity to expanded query
                    relation_scores = []
                    for i, embedding in enumerate(relation_embeddings):
                        sim_score = float(get_similarity([expanded_embedding], [embedding])[0][0])
                        relation_scores.append((i, sim_score))
                    
                    # Sort by score
                    relation_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    # Select top relations to expand
                    top_to_expand = min(len(relation_scores), max_relations_per_level[hop_level])
                    for idx, _ in relation_scores[:top_to_expand]:
                        rel = valid_relations[idx]
                        related_from = rel.get('relatedFromIdName', '')
                        relation_label = rel.get('additionalRelationLabel', '').replace('_', ' ')
                        related_to = rel.get('relatedIdName', '')
                        
                        # Skip if we've already visited this concept
                        if related_to.lower() in visited_concepts:
                            continue
                        
                        # Mark as visited
                        visited_concepts.add(related_to.lower())
                        
                        # Get CUI for the target concept for further exploration
                        target_cuis = umls_api.search_cui(related_to)
                        if target_cuis:
                            target_cui = target_cuis[0][0]  # Use first match
                            # Create path for this relation
                            next_path = [(related_from, relation_label, related_to)]
                            # Add to queue for next hop
                            queue.append((related_to, target_cui, hop_level + 1, next_path))
            else:
                # For higher hops, process and select only top relations
                relation_texts = []
                valid_relations = []
                paths = []
                
                for rel in relations:
                    related_from = rel.get('relatedFromIdName', '')
                    relation_label = rel.get('additionalRelationLabel', '').replace('_', ' ')
                    related_to = rel.get('relatedIdName', '')
                    
                    # Skip if missing essential information or already visited
                    if (not related_from or not relation_label or not related_to or
                        related_to.lower() in visited_concepts):
                        continue
                    
                    # Create text representation with path
                    path_segments = []
                    for p in path:
                        path_segments.append(f"{p[0]} {p[1]} {p[2]}")
                    path_segments.append(f"{related_from} {relation_label} {related_to}")
                    
                    rel_text = f"{hop_level}-hop: " + " → ".join(path_segments)
                    
                    # Store for scoring
                    relation_texts.append(rel_text)
                    valid_relations.append(rel)
                    
                    # Create full path including this relation
                    new_path = path + [(related_from, relation_label, related_to)]
                    paths.append(new_path)
                
                # Score higher-hop relations
                if relation_texts:
                    relation_embeddings = umlsbert.batch_encode(relation_texts)
                    
                    # Score with similarity to expanded query
                    relation_scores = []
                    for i, embedding in enumerate(relation_embeddings):
                        sim_score = float(get_similarity([expanded_embedding], [embedding])[0][0])
                        
                        # Apply hop discount
                        hop_discount = 0.98 ** (hop_level - 1)
                        final_score = sim_score * hop_discount
                        
                        relation_scores.append((i, final_score))
                    
                    # Sort by score
                    relation_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    # For higher hops, only STORE top relations
                    top_to_keep = min(len(relation_scores), max_relations_to_keep.get(hop_level, 20))
                    for i in range(top_to_keep):
                        idx, _ = relation_scores[i]
                        # Add to all relations
                        all_relations.append((hop_level, paths[idx], relation_texts[idx]))
                    
                    # For higher hops, only EXPAND top relations
                    top_to_expand = min(len(relation_scores), max_relations_per_level.get(hop_level, 10))
                    for i in range(top_to_expand):
                        idx, _ = relation_scores[i]
                        related_to = paths[idx][-1][2]
                        
                        # Mark as visited
                        visited_concepts.add(related_to.lower())
                        
                        # Get CUI for further exploration
                        target_cuis = umls_api.search_cui(related_to)
                        if target_cuis:
                            target_cui = target_cuis[0][0]  # Use first match
                            # Add to queue for next hop
                            queue.append((related_to, target_cui, hop_level + 1, paths[idx]))
        
        # Score all relations for final selection
        all_scores = []
        
        # Group relations by hop level
        relations_by_hop = {1: [], 2: [], 3: []}
        for i, (hop_level, path, rel_text) in enumerate(all_relations):
            relations_by_hop[hop_level].append((i, rel_text))
        
        # Score and select from each hop level
        for hop_level, rel_infos in relations_by_hop.items():
            if not rel_infos:
                continue
                
            # For hop 1, consider ALL relations
            # For higher hops, we've already filtered them
            
            # Extract relation texts for scoring
            rel_indices = [idx for idx, _ in rel_infos]
            rel_texts = [text for _, text in rel_infos]
            
            # Compute embeddings
            rel_embeddings = umlsbert.batch_encode(rel_texts)
            
            # Score relations
            for i, embedding in enumerate(rel_embeddings):
                sim_score = float(get_similarity([expanded_embedding], [embedding])[0][0])
                
                # Apply hop discount
                hop_discount = 0.98 ** (hop_level - 1)
                final_score = sim_score * hop_discount
                
                # Store relation index in all_relations and score
                all_scores.append((rel_indices[i], final_score))
        
        # Sort by score
        all_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top relations overall
        top_relations = all_scores[:max_top_relations]  # 20 relations per entity as in original paper
        
        # Process selected relations for output
        rels = []
        for rel_idx, _ in top_relations:
            hop_level, path, _ = all_relations[rel_idx]
            
            if hop_level == 1:
                # Process 1-hop relation directly
                from_name, relation_label, to_name = path[0]
                rels.append((from_name, relation_label, to_name))
                
            else:
                # For multi-hop relations, create a condensed representation
                start_name = path[0][0]
                end_name = path[-1][2]
                
                # Collect relation types from the path
                relation_types = []
                for path_segment in path:
                    relation_types.append(path_segment[1])
                
                # Create path description
                path_description = f"{hop_level}-hop path: " + " → ".join(relation_types)
                
                rels.append((start_name, path_description, end_name))

        process_time = time.time() - process_start
        print(f"Entity processed in {process_time:.2f}s")
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
            if "hop" in rel_1:
                # Multi-hop relation
                rels_text += f"({rel_0} to {rel_2} via {rel_1})\n"
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
    print(f"Total processing time: {total_time:.2f}s")
        
    return context

# Initialize the UMLS API and UMLSBERT
umls_api = UMLS_API(umls_api_key)
umlsbert = UMLSBERT()