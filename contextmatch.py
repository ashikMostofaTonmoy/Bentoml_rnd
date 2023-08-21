import bentoml

# Load the SentenceTransformer model
model = bentoml.load("paraphrase-multilingual-mpnet-base-v2")

def match_context(answer: dict) -> dict:
    start_time = time.time()

    # Encode the preset answer and applicant answer
    embedding1 = model.encode(str(answer["preset_answer"]).lower(), convert_to_tensor=True)
    embedding2 = model.encode(str(answer["applicant_answer"]).lower(), convert_to_tensor=True)

    # Calculate the similarity score
    similarity_score = util.pytorch_cos_sim(embedding1, embedding2).item()

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time

    result = {
        "answer_id": answer["answer_id"],
        "score": similarity_score,
        "model_name": model.name,
        "time_taken": elapsed_time,
    }

    return result

# Create a BentoService
service = bentoml.Service(
    name="match_context", description="A service for matching context", version="1.0.0"
)

# Add the match_context function to the service
service.add_function(match_context)

# Save the service
service.save()
