import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import bentoml


@bentoml.env(pip_packages=["transformers"])
@bentoml.artifacts([bentoml.PickleArtifact("model")])
class SentenceSimilarityService(bentoml.BentoService):
    def preprocess(self, sentences):
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        encoded_inputs = tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt")
        return encoded_inputs

    def compute_similarity(self, embeddings):
        model = self.artifacts.model
        with torch.no_grad():
            embeddings = model(**embeddings).pooler_output
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
            cosine_similarity = torch.mm(normalized_embeddings[0].unsqueeze(
                0), normalized_embeddings[1].unsqueeze(0).T)
            return cosine_similarity.item()

    @bentoml.api(input=bentoml.PickleArtifact(), output=bentoml.Float())
    def similarity(self, input_data):
        similarity_score = self.compute_similarity(input_data)
        return similarity_score


# Load the pre-trained model
model = AutoModel.from_pretrained(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# Create a BentoService instance and save it
bento_service = SentenceSimilarityService()
bento_service.pack("model", model)
saved_path = bento_service.save()
