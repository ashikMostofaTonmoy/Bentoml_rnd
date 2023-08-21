from __future__ import annotations

import time
import typing as t
from sentence_transformers import SentenceTransformer, util
from typing import TYPE_CHECKING

import bentoml
from bentoml.io import JSON

if TYPE_CHECKING:
    from bentoml._internal.runner.runner import RunnerMethod

    class RunnerImpl(bentoml.Runner):
        match_context: RunnerMethod

inference_duration = bentoml.metrics.Histogram(
    name="inference_duration",
    documentation="Duration of inference",
    labelnames=["model_name"],
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        7.5,
        10.0,
        float("inf"),
    ),
)


class ContextMatchingRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self):
        self.model_name = "paraphrase-multilingual-mpnet-base-v2"
        self.model = SentenceTransformer(self.model_name)

    @bentoml.Runnable.method(batchable=False)
    def match_context(self, answer: dict) -> dict:
        start_time = time.time()
        embedding1 = self.model.encode(
            str(answer['preset_answer']).lower(), convert_to_tensor=True)
        embedding2 = self.model.encode(
            str(answer['applicant_answer']).lower(), convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(embedding1, embedding2).item()
        end_time = time.time()
        elapsed_time = end_time - start_time
        result = {
            'answer_Id': answer['answer_id'],
            'score': similarity_score,
            'model_name': self.model_name,
            'time_taken': elapsed_time
        }
        inference_duration.labels(
            model_name=self.model_name).observe(elapsed_time)
        return result


context_matching_runner = t.cast(
    "RunnerImpl", bentoml.Runner(
        ContextMatchingRunnable, name="context_matching")
)

svc = bentoml.Service("context_matcher", runners=[context_matching_runner])


@svc.api(input=JSON(), output=JSON())
async def context_matching(answer: dict) -> dict:
    result = await context_matching_runner.match_context.async_run(answer)
    return result

if __name__ == "__main__":
    example_answer = {
        "answer_id": 1,
        "preset_answer": "i kjsdfa you ",
        "applicant_answer": "I kjbgahgl you"
    }
    result = context_matching(example_answer)
    print(result)
