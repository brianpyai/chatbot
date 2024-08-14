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

class ConceptNetwork(nn.Module):
    def __init__(self, embedding_model, num_concepts, hidden_dim):
        super(ConceptNetwork, self).__init__()
        self.embedding_model = embedding_model
        self.num_concepts = num_concepts
        self.concept_extractor = nn.Sequential(
            nn.Linear(embedding_model.embedding.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_concepts)
        )
        self.predictor = nn.Linear(num_concepts, embedding_model.embedding.embedding_dim)
        
    def forward(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            with torch.no_grad():
                embedding = self.embedding_model.embeds(text)
            embeddings.append(embedding)
        embeddings = torch.stack(embeddings)
        concept_scores = self.concept_extractor(embeddings)
        predictions = self.predictor(concept_scores)
        return concept_scores, predictions, embeddings

    def extract_keywords(self, text, top_n=20):
        concept_scores, _, _ = self(text)
        _, top_indices = torch.topk(concept_scores, min(top_n, concept_scores.size(1)))
        keywords = [self.embedding_model.decode([idx.item()])[0] for idx in top_indices[0]]
        return keywords

    def generate_concept_tree(self, text, top_n=20):
        concept_scores, predictions, _ = self(text)
        
        
        _, top_indices = torch.topk(concept_scores[0], min(top_n, concept_scores.size(1)))
       
        tree = {}
        for i, idx in enumerate(top_indices):
            keyword = self.embedding_model.decode([idx.item()])[0]
            tree[keyword] = []
          
            for j, other_idx in enumerate(top_indices):
                if i != j:
                    other_keyword = self.embedding_model.decode([other_idx.item()])[0]
                    similarity = F.cosine_similarity(predictions[0, idx].unsqueeze(0), predictions[0, other_idx].unsqueeze(0))
                    tree[keyword].append((other_keyword, similarity.item()))
            
           
            tree[keyword].sort(key=lambda x: x[1], reverse=True)
        
        return tree

    def generate_tree(self, text):
        keywords = self.extract_keywords(text)
        tree = build_relationship_tree(keywords, self.embedding_model.embeds)
        return tree

class ConceptTreeDataset(Dataset):
    def __init__(self, items_iterator, max_len):
        self.items_iterator = items_iterator
        self.max_len = max_len

    def __getitem__(self, idx):
        title, content = next(self.items_iterator)
        text = title + "\n" + content
        return text

    def __len__(self):
        return self.max_len