"""Local checkpoint inference using the same prompt as training."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .prompting import MAX_TOKENS, make_prompt


class Converter:
    def __init__(self, model_path, *, cpu=False, max_new_tokens=128):
        if max_new_tokens <= 0:
            raise ValueError("max-new-tokensは正の整数を指定してください。")
        if not cpu and not torch.cuda.is_available():
            raise ValueError(
                "CUDA GPUが必要です。CPUで評価する場合は--cpuを指定してください。"
            )
        if not model_path.is_dir():
            raise ValueError(
                "--modelには保存済みモデルのディレクトリを指定してください。"
            )
        self.device = torch.device("cpu" if cpu else "cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        if self.tokenizer.eos_token_id is None:
            raise ValueError("終了記号のあるtokenizerが必要です。")
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                dtype=torch.float32,
                attn_implementation="sdpa",
            )
            .to(self.device)
            .eval()
        )
        self.limit = min(MAX_TOKENS, self.model.config.max_position_embeddings)
        # Do not inherit sampling, beam search, or penalties from generation_config.json.
        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id,
            use_cache=True,
        )
        # Transformers fills unset fields from model.generation_config even when
        # generate() receives an explicit config. Reset both to prevent leakage.
        self.model.generation_config = self.generation_config

    def prepare_inputs(self, items):
        inputs = []
        for item in items:
            encoded = self.tokenizer(
                make_prompt(item["input"], item["context_text"]),
                return_tensors="pt",
                return_token_type_ids=False,
            )
            length = encoded.input_ids.shape[-1]
            if length + self.generation_config.max_new_tokens > self.limit:
                raise ValueError(
                    f"index={item['index']}: 入力{length}＋生成上限"
                    f"{self.generation_config.max_new_tokens}が{self.limit}トークンを超えます。"
                    "--max-new-tokensを減らしてください。入力は切り詰めません。"
                )
            inputs.append(encoded)
        return inputs

    @torch.inference_mode()
    def predict(self, inputs):
        inputs = inputs.to(self.device)
        tokens = self.model.generate(**inputs, generation_config=self.generation_config)
        generated = tokens[0, inputs.input_ids.shape[-1] :].tolist()
        ended = generated[-1] == self.tokenizer.eos_token_id
        return {
            "prediction": self.tokenizer.decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ),
            "prompt_tokens": inputs.input_ids.shape[-1],
            "generated_tokens": len(generated),
            "truncated": not ended,
        }
