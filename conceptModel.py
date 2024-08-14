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

    def generate_concept_tree(self, text, top_n=64):
        device = next(self.parameters()).device  # Get the device of the model

        # Extract keywords using the provided function
        keywords = extract_keywords(text, top_n=top_n, embedding_model=self.embedding_model, split_sentences=True)

        # Get concept scores, predictions, and embeddings for all keywords
        concept_scores, predictions, embeddings = self(keywords)

        # Print shapes for debugging
        print(f"concept_scores shape: {concept_scores.shape}")
        print(f"predictions shape: {predictions.shape}")
        print(f"embeddings shape: {embeddings.shape}")

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

class ConceptTreeDataset(Dataset):
    def __init__(self, items_iterator, max_len):
        self.items_iterator = items_iterator
        self.max_len = max_len

    def __getitem__(self, idx):
        title, content = next(self.items_iterator)
        return title, content

    def __len__(self):
        return self.max_len