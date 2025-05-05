import random
import re
import json
import requests
import time
from collections import deque

#--------------- LOAD ENV VARIABLES ---------------------------

# import environment variables
from dotenv import load_dotenv
import os

# load the variables from .env into the environment
load_dotenv()

umls_api_key=os.getenv("UMLS_API_KEY")

#--------------- LOAD ENV VARIABLES ---------------------------

class UMLS_API:
    def __init__(self, apikey, version="current"):
        self.apikey = apikey
        self.version = version
        self.search_url = f"https://uts-ws.nlm.nih.gov/rest/search/{version}"
        self.content_url = f"https://uts-ws.nlm.nih.gov/rest/content/{version}"
        self.content_suffix = "/CUI/{}/{}?apiKey={}"
        
        # Add caching to improve performance
        self.cui_cache = {}
        self.definition_cache = {}
        self.relation_cache = {}
        
        # Rate limiting to avoid API errors
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
        # Check cache first
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
                    
            # Update cache
            self.cui_cache[query] = cui_results

        except Exception as e:
            print(f"Error searching for CUI: {e}")

        return cui_results

    def get_definitions(self, cui):
        # Check cache first
        if cui in self.definition_cache:
            return self.definition_cache[cui]
            
        try:
            self._rate_limit()
            suffix = self.content_suffix.format(cui, "definitions", self.apikey)
            r = requests.get(self.content_url + suffix)
            
            # Handle 404 errors (no definitions)
            if r.status_code == 404:
                self.definition_cache[cui] = []
                return []
                
            r.raise_for_status()
            r.encoding = "utf-8"
            outputs = r.json()
            
            # Update cache
            result = outputs["result"]
            self.definition_cache[cui] = result
            return result
            
        except requests.exceptions.HTTPError as e:
            if "404" in str(e):
                # No definitions found (this is normal for some concepts)
                self.definition_cache[cui] = []
                return []
            else:
                print(f"Error retrieving definitions for {cui}: {e}")
                return []
        except Exception as e:
            print(f"Error retrieving definitions for {cui}: {e}")
            return []

    def get_relations(self, cui, pages=10):
        # Check cache first
        if cui in self.relation_cache:
            return self.relation_cache[cui]
            
        all_relations = []

        try:
            for page in range(1, pages + 1):
                self._rate_limit()
                suffix = self.content_suffix.format(cui, "relations", self.apikey) + f"&pageNumber={page}"
                r = requests.get(self.content_url + suffix)
                
                # Handle 404 errors
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
        return all_relations
        
    def get_random_relations(self, cui, sample_size=20):
        """Get a random sample of relations for a CUI"""
        relations = self.get_relations(cui)
        
        # Take a random sample to avoid biases
        if len(relations) > sample_size:
            return random.sample(relations, sample_size)
        else:
            return relations
            
    def get_multi_hop_relations(self, cui, max_hops=3, max_relations_per_level=None):
        """
        Get multi-hop relations with random selection at each level
        
        Args:
            cui: The starting concept CUI
            max_hops: Maximum number of hops to explore
            max_relations_per_level: Dictionary specifying max relations per hop level
        """
        # Set default relations per level if not provided
        if max_relations_per_level is None:
            max_relations_per_level = {
                1: 20,  # Level 1: Expand top 20 concepts
                2: 7,   # Level 2: Expand top 7 concepts
                3: 3    # Level 3: Expand top 3 concepts
            }
            
        # All paths found
        all_paths = []
        
        # Track visited CUIs to avoid cycles
        visited_cuis = set([cui])
        
        # Queue for BFS: (cui, path, hop_level)
        queue = deque([(cui, [], 1)])
        
        # Process queue
        while queue:
            current_cui, current_path, hop_level = queue.popleft()
            
            # Stop if we've reached the maximum hop level
            if hop_level > max_hops:
                continue
                
            # Get relations for this CUI
            relations = self.get_relations(current_cui)
            
            # Get max relations to explore at this hop level
            max_to_expand = max_relations_per_level.get(hop_level, 1)
            
            # Randomly sample relations to avoid biases
            if len(relations) > max_to_expand:
                sampled_relations = random.sample(relations, max_to_expand)
            else:
                sampled_relations = relations
                
            # Process each relation
            for relation in sampled_relations:
                # Get the related CUI
                related_cui = relation.get('relatedId')
                if not related_cui:
                    continue
                    
                # Get relation text
                related_from = relation.get('relatedFromIdName', '')
                relation_label = relation.get('additionalRelationLabel', '').replace('_', ' ')
                related_to = relation.get('relatedIdName', '')
                
                # Create text representation
                rel_text = f"{related_from} {relation_label} {related_to}"
                
                # Create new path
                new_path = current_path + [(rel_text, relation, hop_level)]
                
                # Add to all paths
                all_paths.append(new_path)
                
                # Continue exploration if not visited and within hop limit
                if related_cui not in visited_cuis and hop_level < max_hops:
                    visited_cuis.add(related_cui)
                    queue.append((related_cui, new_path, hop_level + 1))
                    
        return all_paths
    
umls_api = UMLS_API(umls_api_key)

def get_umls_keys(query, prompt, llm):
    """Get multi-hop knowledge from UMLS with random selection"""
    start_time = time.time()
    umls_res = {}
    prompt = prompt.replace("{question}", query)

    try:
        # Extract medical entities
        print("Extracting medical entities...")
        keys_text = llm.predict(prompt)
        print(keys_text)
        pattern = r"\{(.*?)\}"
        matches = re.findall(pattern, keys_text.replace("\n", ""))
        if not matches:
            raise ValueError("No medical terminologies returned by the model.")
        
        keys_dict = json.loads("{" + matches[0] + "}")
        if "medical terminologies" not in keys_dict or not keys_dict["medical terminologies"]:
            raise ValueError("Model did not return expected 'medical terminologies' key.")
            
        print(f"Found medical entities: {', '.join(keys_dict['medical terminologies'])}")
            
    except Exception as e:
        print(f"Error during model processing: {e}")
        return "" 

    # Multi-hop parameters
    max_hops = 3
    max_relations_per_level = {
        1: 20,  # Level 1: Expand top 20 concepts
        2: 7,   # Level 2: Expand top 7 concepts
        3: 3    # Level 3: Expand top 3 concepts
    }
    max_relations_to_sample = 20  # Maximum relations to include in final output

    for key in keys_dict["medical terminologies"][:]:
        print(f"Processing entity: {key}")
        entity_start = time.time()
        cuis = umls_api.search_cui(key)

        if len(cuis) == 0:
            continue
        cui = cuis[0][0]
        name = cuis[0][1]

        # Get definition with priority to reliable sources
        defi = ""
        definitions = umls_api.get_definitions(cui)
        if definitions:
            msh_def = None
            nci_def = None
            icf_def = None
            csp_def = None
            hpo_def = None

            for definition in definitions:
                source = definition.get("rootSource")
                if source == "MSH":
                    msh_def = definition.get("value")
                    break
                elif source == "NCI":
                    nci_def = definition.get("value")
                elif source == "ICF":
                    icf_def = definition.get("value")
                elif source == "CSP":
                    csp_def = definition.get("value")
                elif source == "HPO":
                    hpo_def = definition.get("value")

            defi = msh_def or nci_def or icf_def or csp_def or hpo_def

        # Get multi-hop relations with random selection
        print(f"Exploring multi-hop relations for {name}...")
        relation_paths = umls_api.get_multi_hop_relations(cui, max_hops=max_hops, 
                                                        max_relations_per_level=max_relations_per_level)
        print(f"Found {len(relation_paths)} paths")
        
        rels = []
        
        if relation_paths:
            # Group paths by hop level
            paths_by_hop = {}
            for path in relation_paths:
                if not path:
                    continue
                # Get the maximum hop level in this path
                max_hop = max(rel[2] for rel in path)
                if max_hop not in paths_by_hop:
                    paths_by_hop[max_hop] = []
                paths_by_hop[max_hop].append(path)
            
            # Sample paths from each hop level for balanced representation
            sampled_paths = []
            for hop, paths in paths_by_hop.items():
                # Number to sample depends on hop level - more from lower hops
                sample_size = min(len(paths), max_relations_to_sample // len(paths_by_hop))
                sampled_paths.extend(random.sample(paths, sample_size))
            
            # Process sampled paths
            for path in sampled_paths:
                if len(path) == 1:
                    # Single hop relation
                    rel_text, relation, _ = path[0]
                    related_from = relation.get('relatedFromIdName', '')
                    relation_label = relation.get('additionalRelationLabel', '').replace('_', ' ')
                    related_to = relation.get('relatedIdName', '')
                    rels.append((related_from, relation_label, related_to))
                else:
                    # Multi-hop relation
                    hop_count = max(rel[2] for rel in path)
                    start_relation = path[0][1]
                    end_relation = path[-1][1]
                    
                    # Path description
                    path_segments = []
                    for rel_text, _, _ in path:
                        path_segments.append(rel_text)
                    
                    path_description = f"{hop_count}-hop path: " + " → ".join(path_segments)
                    
                    # Include as a special relation
                    from_name = start_relation.get('relatedFromIdName', '')
                    to_name = end_relation.get('relatedIdName', '')
                    rels.append((from_name, path_description, to_name))

        # Ensure we don't have too many relations
        if len(rels) > max_relations_to_sample:
            rels = random.sample(rels, max_relations_to_sample)
            
        entity_time = time.time() - entity_start
        print(f"Entity processed in {entity_time:.2f}s")

        umls_res[cui] = {"name": name, "definition": defi, "rels": rels}

    # Format context
    context = ""
    for k, v in umls_res.items():
        name = v["name"]
        definition = v["definition"]
        rels = v["rels"]
        rels_text = ""
        
        for rel in rels:
            # Format based on whether it's a multi-hop path
            if "hop path:" in rel[1]:
                rels_text += f"({rel[0]} to {rel[2]} via {rel[1]})\n"
            else:
                rels_text += f"({rel[0]} {rel[1]} {rel[2]})\n"
                
        text = f"Name: {name}\nDefinition: {definition}\n"
        if rels_text:
            text += f"Relations: \n{rels_text}"

        context += text + "\n"
        
    if context:
        context = context[:-1]
        
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f}s")
    
    return context