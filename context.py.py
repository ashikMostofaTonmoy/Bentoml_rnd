import time
import sentence_transformers
from sentence_transformers import SentenceTransformer, util


def matchContext(answer: dict) -> dict:

    start_time = time.time()  # Record start time
    model_name = "paraphrase-multilingual-mpnet-base-v2"

    # name of pre-trained sentence transformer model
    model = SentenceTransformer(model_name)

    embedding1 = model.encode(
        str(answer['preset_answer']).lower(), convert_to_tensor=True)
    embedding2 = model.encode(
        str(answer['applicant_answer']).lower(), convert_to_tensor=True)

    similarity_score = util.pytorch_cos_sim(embedding1, embedding2).item()

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time

    result = dict({
        'answer_Id': answer['answer_id'],
        'score': similarity_score,
        'model_name': model_name,
        'time_taken': elapsed_time
    })

    return result


if __name__ == "__main__":
    print(sentence_transformers.__version__)
    #    print(
    #     matchContext([
    #     {
    #         "answer_id": 1,
    #         "preset_answer": "i kjsdfa you ",
    #         "applicant_answer": "I kjbgahgl you"
    #     }
    #     ])
    #    )
