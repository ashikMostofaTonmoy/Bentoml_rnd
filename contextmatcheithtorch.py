from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F

# Load model and tokenizer
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Encode sentences into embeddings
sentences = ["i like the weather today.", "the weather seems preety good"]
encoded_inputs = tokenizer(sentences, padding=True,
                           truncation=True, return_tensors="pt")
with torch.no_grad():
    embeddings = model(**encoded_inputs).pooler_output

# Calculate cosine similarity using PyTorch
normalized_embeddings = F.normalize(
    embeddings, p=2, dim=1)  # Normalize embeddings
cosine_similarity = torch.mm(normalized_embeddings[0].unsqueeze(
    0), normalized_embeddings[1].unsqueeze(0).T)

# Print the cosine similarity
print("Cosine Similarity between the sentences:")
print(cosine_similarity.item())
print(type(cosine_similarity))
