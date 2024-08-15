def extract_keywords(text, top_n=20, embedding_model=None,split_sentences=False):
    lines = text.split('\n', 1)
    title = lines[0]
    content = lines[1] if len(lines) > 1 else ''
    title_tokens = embedding_model.encode(title)
    content_tokens = embedding_model.encode(content)
    

    #if len(text )>2000:top_n=min(40,int(len(text)/200))
    #elif len(text )>20000:top_n=min(90,int(len(text)/200))
    

    title_words = embedding_model.decode(title_tokens).split()
    content_words=split_into_sentences(content) if split_sentences else embedding_model.decode(content_tokens).split()  #
    top_n=min ( 100 ,max(20,int( len(content_words)/33)+len(title_words) ) )
    

    word_freq = {}
    for word in title_words:
        word_freq[word] = word_freq.get(word, 0) + 3  
    for word in content_words:
        word_freq[word] = word_freq.get(word, 0) + 1

 
    total_words = len(title_words) * 3 + len(content_words)
    word_scores = []
    for word, freq in word_freq.items():
        tf = freq / total_words
        idf = 1  
        score = tf * idf
        word_scores.append((word, score))


    word_scores.sort(key=lambda x: x[1], reverse=True)
    l=int(len (word_scores)*4/10)
    return [word for word, score in word_scores][:top_n]



def build_relationship_tree(keywords, embeddings):
    tree = {keyword: [] for keyword in keywords}
    all_similarities = []
    
    # 計算所有關鍵詞對之間的相似度
    for i, keyword1 in enumerate(keywords):
        for j, keyword2 in enumerate(keywords[i+1:], start=i+1):
            similarity = cosine_similarity([embeddings(keyword1)], [embeddings(keyword2)])[0][0]
            all_similarities.append((keyword1, keyword2, similarity))
    
    # 計算相似度閾值
    all_similarities.sort(key=lambda x: x[2], reverse=True)
    num_relations = len(all_similarities)
    num_to_keep = min(num_relations, min(80, max(int(num_relations / 50), 10)))
    
    if num_to_keep > 0:
        similarity_threshold = all_similarities[num_to_keep-1][2]
    else:
        similarity_threshold = 0  # 如果沒有關係，設置閾值為0
    
    # 根據閾值添加關係，保持原始順序
    for i, keyword1 in enumerate(keywords):
        for j, keyword2 in enumerate(keywords[i+1:], start=i+1):
            similarity = cosine_similarity([embeddings(keyword1)], [embeddings(keyword2)])[0][0]
            if similarity >= similarity_threshold:
                tree[keyword1].append((keyword2, similarity))
                tree[keyword2].append((keyword1, similarity))
    
    # 對每個關鍵詞的關係列表進行排序
    for keyword in tree:
        tree[keyword].sort(key=lambda x: x[1], reverse=True)
    
    return tree

def _format_concept_tree( tree, max_depth=3):
    def _format_tree(tree, depth=0, prefix=""):
        if depth > max_depth:
            return ""
        
        output = ""
        for i, (concept, relations) in enumerate(tree.items()):
            if i == len(tree) - 1:
                branch = "└─ "
                new_prefix = prefix + "   "
            else:
                branch = "├─ "
                new_prefix = prefix + "│  "

            output += f"{prefix}{branch}{concept}\n"

            for j, (related_concept, similarity) in enumerate(relations[:3]):  # 只显示前3个相关概念
                if j == len(relations) - 1 or j == 2:
                    rel_branch = "└─ "
                else:
                    rel_branch = "├─ "

                output += f"{new_prefix}{rel_branch}{related_concept} ({similarity:.2f})\n"

            if len(relations) > 3:
                output += f"{new_prefix}└─ ...\n"

        return output

    return _format_tree(tree)

def _format_concept_tree( tree, max_depth=3):
    def _format_tree(tree, depth=0, prefix=""):
        if depth > max_depth:
            return ""
        
        output = ""
        for i, (concept, relations) in enumerate(tree.items()):
            if i == len(tree) - 1:
                branch = "└─ "
                new_prefix = prefix + "   "
            else:
                branch = "├─ "
                new_prefix = prefix + "│  "

            output += f"{prefix}{branch}{concept}\n"

            for j, (related_concept, similarity) in enumerate(relations[:3]):  # 只显示前3个相关概念
                if j == len(relations) - 1 or j == 2:
                    rel_branch = "└─ "
                else:
                    rel_branch = "├─ "

                output += f"{new_prefix}{rel_branch}{related_concept} ({similarity:.2f})\n"

            if len(relations) > 3:
                output += f"{new_prefix}└─ ...\n"

        return output

    return _format_tree(tree)

class ConceptNetwork(nn.Module):
    def __init__(self, embedding_model, num_concepts, hidden_dim, device=device):
        super(ConceptNetwork, self).__init__()
        self.embedding_model = embedding_model
        self.num_concepts = num_concepts
        self.concept_extractor = nn.Sequential(
            nn.Linear(embedding_model.embedding.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_concepts)
        )
        self.predictor = nn.Linear(num_concepts, embedding_model.embedding.embedding_dim)
        self.device = device
   
    def embeds(self, text):
        with torch.no_grad():
            ids = self.embedding_model.tokenizer(text, return_tensors='pt')['input_ids'].to(self.device)
            embedding = self.embedding_model.embedding(ids).mean(dim=1)
            concept_scores = self.concept_extractor(embedding)
            return self.predictor(concept_scores)
    
    def forward(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        concept_scores_list = []
        predictions_list = []
        
        for text in texts:
            with torch.no_grad():
                ids = self.embedding_model.tokenizer(text, return_tensors='pt')['input_ids'].to(self.device)
                embedding = self.embedding_model.embedding(ids).mean(dim=1)
            embeddings.append(embedding)
            
            concept_scores = self.concept_extractor(embedding)
            concept_scores_list.append(concept_scores)
            
            predictions = self.predictor(concept_scores)
            predictions_list.append(predictions)
        
        if embeddings:embeddings = torch.cat(embeddings, dim=0)
        else:embeddings = torch.empty(0)
        if concept_scores_list:concept_scores = torch.cat(concept_scores_list, dim=0)
        else:concept_scores = torch.empty(0)
        if predictions_list:predictions = torch.cat(predictions_list, dim=0)
        else:predictions = torch.empty(0)
        
        return concept_scores, predictions, embeddings

    def generate_concept_tree(self, text, top_n=64 , split_sentences=True):
        device = next(self.parameters()).device  # Get the device of the model

        # Extract keywords using the provided function
        keywords = extract_keywords(text, top_n=top_n, embedding_model=self.embedding_model, split_sentences=split_sentences)

        # Get concept scores, predictions, and embeddings for all keywords
        concept_scores, predictions, embeddings = self(keywords)

        # Print shapes for debugging
        #print(f"concept_scores shape: {concept_scores.shape}")
        #print(f"predictions shape: {predictions.shape}")
        #print(f"embeddings shape: {embeddings.shape}")

        # Reshape tensors to 2D
        concept_scores = concept_scores.view(concept_scores.size(0), -1)
        predictions = predictions.view(predictions.size(0), -1)
        embeddings = embeddings.view(embeddings.size(0), -1)

        tree = {}
        for i, keyword in enumerate(keywords):
            tree[keyword] = []
            for j, other_keyword in enumerate(keywords):
                if i != j:
                    # Calculate similarity based on embeddings
                    embedding_similarity = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0))

                    # Calculate similarity based on concept scores
                    concept_similarity = F.cosine_similarity(concept_scores[i].unsqueeze(0), concept_scores[j].unsqueeze(0))

                    # Calculate similarity based on predictions
                    prediction_similarity = F.cosine_similarity(predictions[i].unsqueeze(0), predictions[j].unsqueeze(0))

                    # Combine all factors
                    combined_score = (embedding_similarity + concept_similarity + prediction_similarity) / 3

                    # Ensure combined_score is a scalar
                    combined_score = combined_score.item()

                    tree[keyword].append((other_keyword, combined_score))

            # Sort relations by combined score
            tree[keyword].sort(key=lambda x: x[1], reverse=True)

        return tree

    def generate_tree(self, text):
        keywords = extract_keywords(text, embedding_model=self.embedding_model, split_sentences=True)
        tree = build_relationship_tree(keywords, self.embedding_model.embeds)
        return tree
