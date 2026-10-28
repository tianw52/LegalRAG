import torch
from benchmark_rag.components.base import BaseReranker, RetrievedChunk

_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
    'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_DEFAULT_INSTRUCTION = (
    "Given a legal question or situation described by a person in Canada, retrieve passages "
    "from Canadian court decisions or legislation that help explain the applicable law"
)

class Qwen3Reranker(BaseReranker):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        device: str = "cuda:0",
        batch_size: int = 8,
        max_length: int = 8192,
        instruction: str | None = None,
        quantization: str | None = None,
    ):
        self.quantization = quantization
        self.model_name = model_name
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.instruction = instruction or _DEFAULT_INSTRUCTION
        self._bundle = None

    def _load(self):
        if self._bundle is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        tok = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        if self.quantization == "8bit":
            from transformers import BitsAndBytesConfig

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=dtype, device_map={"": self.device.index or 0},
                quantization_config=BitsAndBytesConfig(load_in_8bit=True)).eval()
        else:
            model = (
                AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=dtype)
                .to(self.device)
                .eval()
            )
        self._bundle = (
            tok,
            model,
            tok.convert_tokens_to_ids("yes"),
            tok.convert_tokens_to_ids("no"),
            tok.encode(_PREFIX, add_special_tokens=False),
            tok.encode(_SUFFIX, add_special_tokens=False),
        )

    def _score(self, texts: list[str]) -> list[float]:
        tok, model, yes_id, no_id, prefix, suffix = self._bundle
        enc = tok(
            texts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(prefix) - len(suffix),
        )
        enc["input_ids"] = [prefix + ids + suffix for ids in enc["input_ids"]]
        batch = tok.pad(enc, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():

            last = model(**batch, logits_to_keep=1).logits[:, -1, :]
        pair = torch.stack([last[:, no_id], last[:, yes_id]], dim=1).float()
        return torch.nn.functional.log_softmax(pair, dim=1)[:, 1].exp().tolist()

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        self._load()
        texts = [
            f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {c.text}" for c in chunks
        ]

        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        scores = [0.0] * len(texts)
        for s in range(0, len(order), self.batch_size):
            idx = order[s : s + self.batch_size]
            for i, sc in zip(idx, self._score([texts[i] for i in idx])):
                scores[i] = sc
        for c, sc in zip(chunks, scores):
            c.score = float(sc)
        return sorted(chunks, key=lambda c: c.score, reverse=True)
