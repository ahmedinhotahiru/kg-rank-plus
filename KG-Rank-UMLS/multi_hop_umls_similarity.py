import torch
import numpy as np
import re
import json
import requests
import time
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque

#--------------- LOAD ENV VARIABLES ---------------------------
from dotenv import load_dotenv
import os

# load the variables from .env into the environment
load_dotenv()
umls_api_key=os.getenv("UMLS_API_KEY")
#--------------- LOAD ENV VARIABLES ---------------------------

class UMLSBERT:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("GanjinZero/UMLSBert_ENG")
        self.model = AutoModel.from_pretrained("GanjinZero/UMLSBert_ENG")
        self.embed_cache = {}  # Add embedding cache

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def batch_encode(self, texts, batch_size=16):
        all_embeddings = []
        cached_indices = {}
        uncached_texts = []
        
        # Check cache first
        for i, text in enumerate(texts):
            cache_key = hash(text)
            if cache_key in self.embed_cache:
                cached_indices[i] = self.embed_cache[cache_key]
            else:
                uncached_texts.append((i, text, cache_key))
        
        # Process uncached texts
        if uncached_texts:
            for i in range(0, len(uncached_texts), batch_size):
                batch_indices = []
                batch_texts = []
                batch_keys = []
                
                # Prepare batch
                for j in range(i, min(i + batch_size, len(uncached_texts))):
                    idx, text, key = uncached_texts[j]
                    batch_indices.append(idx)
                    batch_texts.append(text)
                    batch_keys.append(key)
                
                # Encode batch
                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    model_output = self.model(**inputs)
                    attention_mask = inputs["attention_mask"]
                    batch_embeddings = self.mean_pooling(model_output, attention_mask)
                
                # Cache results
                for j, idx in enumerate(batch_indices):
                    embedding = batch_embeddings[j].numpy()
                    self.embed_cache[batch_keys[j]] = embedding
                    cached_indices[idx] = embedding
        
        # Assemble final embeddings in correct order
        for i in range(len(texts)):
            all_embeddings.append(cached_indices[i])
            
        return np.array(all_embeddings)

def get_similarity(query_vec, rel_vec):
    return cosine_similarity(query_vec, rel_vec)

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
            # self._rate_limit()
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
            # self._rate_limit()
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

    def get_relations(self, cui, pages=5):
        # Check cache
        if cui in self.relation_cache:
            return self.relation_cache[cui]
            
        all_relations = []
        try:
            for page in range(1, pages + 1):
                # self._rate_limit()
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

umls_api = UMLS_API(umls_api_key)
umlsbert = UMLSBERT()

def get_umls_keys(query, prompt, llm):
    umls_res = {}
    prompt = prompt.replace("{question}", query)

    # Get query embedding once for later similarity comparisons
    query_embedding = umlsbert.batch_encode([query])[0]

    try:
        # Extract medical entities
        keys_text = llm.predict(prompt)
        print(keys_text)
        
        # Extract entities with robust parsing
        pattern = r"\{(.*?)\}"
        matches = re.findall(pattern, keys_text.replace("\n", ""))
        
        entities = []
        if matches:
            try:
                entities_dict = json.loads("{" + matches[0] + "}")
                entities = entities_dict.get("medical terminologies", [])
            except:
                # Fallback extraction methods
                pattern = r"\"medical terminologies\":\s*\[(.*?)\]"
                matches = re.findall(pattern, keys_text.replace("\n", ""))
                if matches:
                    entities = re.findall(r"\"(.*?)\"", matches[0])
                else:
                    # Just extract quoted terms as a last resort
                    entities = re.findall(r"\"(.*?)\"", keys_text)[:5]
        
        if not entities:
            print("No medical entities found")
            return ""
            
    except Exception as e:
        print(f"Error during model processing: {e}")
        return "" 

    # Configurable parameters with optimal defaults
    max_hops = 3
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
    max_top_relations = 20

    for key in entities:
        cuis = umls_api.search_cui(key)

        if len(cuis) == 0:
            continue
        cui = cuis[0][0]
        name = cuis[0][1]

        # Get definition with improved source prioritization
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

        # N-hop relation implementation with optimizations
        rels = []
        relation_texts = [query]  # Start with query embedding for comparison
        all_relations = []        # Store all relations with their hop levels
        
        # Store relations by hop level for later filtering
        hop_relations = {1: [], 2: [], 3: []}
        
        # Track visited concepts to avoid cycles
        visited_concepts = set([name.lower()])
        
        # Queue for BFS traversal: (concept_name, cui, hop_level, path)
        queue = deque([(name, cui, 1, [])])
        
        # Continue BFS until queue is empty or max hop level is reached
        while queue:
            concept_name, concept_cui, hop_level, path = queue.popleft()
            
            # Stop if we've reached the maximum hop level
            if hop_level > max_hops:
                continue
                
            # Get relations for current concept
            relations = umls_api.get_relations(concept_cui)
            
            # Skip if no relations found
            if not relations:
                continue
            
            # CHANGE 1: Score and select relations based on similarity
            
            # Create text representations for each relation
            rel_texts = []
            valid_relations = []
            
            for rel in relations:
                related_from = rel.get('relatedFromIdName', '')
                relation_label = rel.get('additionalRelationLabel', '').replace('_', ' ')
                related_to = rel.get('relatedIdName', '')
                
                # Skip if missing essential information
                if not related_to:
                    continue
                
                # Skip if we've already visited this concept
                if related_to.lower() in visited_concepts:
                    continue
                
                # Create text representation for scoring
                rel_text = f"{related_from} {relation_label} {related_to}"
                rel_texts.append(rel_text)
                valid_relations.append(rel)
            
            # Score relations if we have any valid ones
            if valid_relations:
                # Get embeddings
                rel_embeddings = umlsbert.batch_encode(rel_texts)
                
                # Calculate similarity scores
                rel_scores = []
                for i, embedding in enumerate(rel_embeddings):
                    sim_score = float(get_similarity([query_embedding], [embedding])[0][0])
                    rel_scores.append((sim_score, i))
                
                # Sort by similarity score (highest first)
                rel_scores.sort(reverse=True)
                
                # Get top N relations to expand
                max_to_expand = max_relations_per_level.get(hop_level, 1)
                pruned_indices = [idx for _, idx in rel_scores[:max_to_expand]]
                pruned_relations = [valid_relations[idx] for idx in pruned_indices]
            else:
                pruned_relations = []
            
            # Process relations from this hop level
            for rel in pruned_relations:
                related_from = rel.get('relatedFromIdName', '')
                relation_label = rel.get('additionalRelationLabel', '').replace('_', ' ')
                related_to = rel.get('relatedIdName', '')
                
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
                
                # Add relation text to our list for similarity comparison
                relation_texts.append(rel_text)
                
                # Store relation info with hop level and full path
                all_relations.append((hop_level, full_path, rel, rel_text))
                
                # CHANGE 2: Store by hop level for later filtering
                hop_relations[hop_level].append((full_path, rel, rel_text))
                
                # Continue exploration from this concept
                target_cuis = umls_api.search_cui(related_to)
                if target_cuis:
                    target_cui = target_cuis[0][0]  # Use first match
                    queue.append((related_to, target_cui, hop_level + 1, full_path))
        
        # CHANGE 3: Filter relations to keep by hop level based on similarity
        filtered_relations = []
        
        # Process each hop level separately
        for level in range(1, max_hops + 1):
            level_relations = hop_relations.get(level, [])
            
            if level_relations:
                # Extract text representations for scoring
                level_texts = [rel_info[2] for rel_info in level_relations]
                
                # Get embeddings and score against query
                level_embeddings = umlsbert.batch_encode(level_texts)
                
                # Score relations
                level_scores = []
                for i, embedding in enumerate(level_embeddings):
                    sim_score = float(get_similarity([query_embedding], [embedding])[0][0])
                    level_scores.append((sim_score, i))
                
                # Sort by similarity score
                level_scores.sort(reverse=True)
                
                # Keep top N relations for this level
                max_to_keep = max_relations_to_keep.get(level, 20)
                top_indices = [idx for _, idx in level_scores[:max_to_keep]]
                
                # Add to filtered relations
                for idx in top_indices:
                    full_path, rel, rel_text = level_relations[idx]
                    filtered_relations.append((level, full_path, rel, rel_text))
        
        # Score all filtered relations against query for final selection
        if filtered_relations:
            filtered_texts = [rel_info[3] for rel_info in filtered_relations]
            filtered_embeddings = umlsbert.batch_encode(filtered_texts)
            
            # Score with small hop discount
            final_scores = []
            for i, embedding in enumerate(filtered_embeddings):
                sim_score = float(get_similarity([query_embedding], [embedding])[0][0])
                hop_level = filtered_relations[i][0]
                hop_discount = 0.95 ** (hop_level - 1)  # Small discount for multi-hop
                final_scores.append((sim_score * hop_discount, i))
            
            # Sort by final score
            final_scores.sort(reverse=True)
            
            # Select top relations
            top_indices = [idx for _, idx in final_scores[:max_top_relations]]
            
            # Process top relations
            for idx in top_indices:
                level, full_path, _, _ = filtered_relations[idx]
                
                if level == 1:
                    # Process 1-hop relation directly
                    from_name, relation_label, to_name = full_path[0]
                    rels.append((from_name, relation_label, to_name))
                else:
                    # For multi-hop relations, create a condensed representation
                    start_name = full_path[0][0]
                    end_name = full_path[-1][2]
                    path_description = f"{level}-hop: "
                    
                    # Add relation types in the path
                    relation_types = []
                    for path_segment in full_path:
                        relation_types.append(path_segment[1])
                        
                    path_description += " → ".join(relation_types)
                    
                    rels.append((start_name, path_description, end_name))

        umls_res[cui] = {"name": name, "definition": defi, "rels": rels}

    # Format context for LLM consumption
    context = "" 
    for k, v in umls_res.items():
        name = v["name"]
        definition = v["definition"]
        rels = v["rels"]
        
        # Format relations as before
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
        
    return context