from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import DataLoader
from colpali_engine.models.paligemma_colbert_architecture import ColPali
from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator
from colpali_engine.utils.colpali_processing_utils import (
    process_images,
    process_queries,
)
from tqdm import tqdm
from transformers import AutoProcessor
from colpali_engine.utils.image_from_page_utils import load_from_dataset
from PIL import Image


class DocumentEmbeddingModel:
    def __init__(self):
        self.model = ColPali.from_pretrained(
            "google/paligemma-3b-mix-448",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            low_cpu_mem_usage=True,
        ).eval()
        adapter_name = "vidore/colpali"
        self.model.load_adapter(adapter_name)
        self.processor = AutoProcessor.from_pretrained(adapter_name)

    def embed_docs(self, images: list) -> list[torch.Tensor]:
        dataloader = DataLoader(
            images,
            batch_size=8,
            shuffle=False,
            collate_fn=lambda x: process_images(self.processor, x),
        )

        embeddings = []
        for docs in tqdm(dataloader):
            with torch.inference_mode():
                embeds = self.model(
                    **{k: v.to(self.model.device) for k, v in docs.items()}
                )
            embeddings.extend(list(torch.unbind(embeds.to("cpu"))))

        return embeddings

    def embed_queries(self, queries: list[str]) -> list[torch.Tensor]:
        dataloader = DataLoader(
            queries,
            batch_size=8,
            shuffle=False,
            collate_fn=lambda x: process_queries(
                self.processor, x, Image.new("RGB", (448, 448), (255, 255, 255))
            ),
        )

        embeddings = []
        for queries in tqdm(dataloader):
            with torch.inference_mode():
                embed = self.model(
                    **{k: v.to(self.model.device) for k, v in queries.items()}
                )
            embeddings.extend(list(torch.unbind(embed.to("cpu"))))

        return embeddings


@dataclass
class SearchResult:
    doc_path: str
    doc_image: Image
    score: float


def search_top_k(
    doc_paths: list[str], queries: list[str], k: int = 20
) -> list[list[SearchResult]]:
    model = DocumentEmbeddingModel()
    docs = [Image.open(path) for path in doc_paths]
    doc_embeds = model.embed_docs(docs)
    query_embeds = model.embed_queries(queries)

    retriever_evaluator = CustomEvaluator(is_multi_vector=True)
    scores = retriever_evaluator.evaluate(query_embeds, doc_embeds)

    print(scores)
    print(scores.shape)

    results = []
    for query_scores in scores:
        sorted_indices = np.argsort(query_scores)[::-1]

        top_k_results = [
            SearchResult(
                doc_path=doc_paths[i], doc_image=docs[i], score=float(query_scores[i])
            )
            for i in sorted_indices[:k]
        ]

        results.append(top_k_results)

    return results


if __name__ == "__main__":
    import json
    import random
    from PIL import Image

    single_test = False
    multi_test = True

    with open("../datasets/mp-docvqa/val.json") as f:
        data = json.load(f)["data"]
    data = [d for d in data if len(d["page_ids"]) == 20]
    sample = random.choice(data)
    question = sample["question"]
    answer_page_id = sample["page_ids"][sample["answer_page_idx"]]

    print(sample)

    if multi_test:
        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)

        doc_paths = [f"/mnt/ssd/images/{page_id}.jpg" for page_id in sample["page_ids"]]

        queries = [
            question,
            question,
        ]

        results = search_top_k(doc_paths, queries)

        print("question:", question)
        print("groundtruth answer page id:", answer_page_id)

        for pair in zip(queries, results):
            print("question:", pair[0])
            for i, result in enumerate(pair[1]):
                print(
                    f"{i+1}. result page id: {result.doc_path.split('/')[-1].split('.')[0]}, score: {result.score:.4f}"
                )

    if single_test:
        images = []
        for page_id in sample["page_ids"]:
            image = Image.open(f"/mnt/ssd/images/{page_id}.jpg")
            images.append(image)

        model = DocumentEmbeddingModel()

        doc_embeds = model.embed_docs(images)
        query_embed = model.embed_queries([question])

        retriever_evaluator = CustomEvaluator(is_multi_vector=True)
        scores = retriever_evaluator.evaluate(query_embed, doc_embeds)

        k = len(sample["page_ids"])

        top_k_indices = scores.argsort(axis=1)[0][-k:][::-1]

        print("question:", question)

        results = []
        for idx in top_k_indices:
            results.append((sample["page_ids"][idx], scores[0][idx]))

        for i, (result, score) in enumerate(results):
            print(f"{i+1}. result page id: {result}, score: {score:.4f}")

        print("groundtruth answer page id:", answer_page_id)
