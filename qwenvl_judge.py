"""Evaluate generated images on VIF-Bench with Qwen3-VL-32B-Instruct, the open-weight judge of the paper.

No API key is needed. Two backends:
  * transformers (default): loads the model on the local GPU(s) (device_map="auto").
    Qwen3-VL-32B in bf16 needs about 70 GB of GPU memory in total.
  * an OpenAI-compatible server (e.g. vLLM) via --api_base:
        vllm serve Qwen/Qwen3-VL-32B-Instruct --max-model-len 32768
        python qwenvl_judge.py ... --api_base http://localhost:8000/v1

Example:
  python qwenvl_judge.py --data_dir ./data --generated_dir ./generations/my_model

Outputs follow judge.py: results/<model_name>/qwen3vl/responses/<task_id>.txt and
results/<model_name>/qwen3vl/scores.jsonl. Decoding is greedy for reproducibility.
"""
import argparse
import base64
import mimetypes
import os
from pathlib import Path

from tqdm import tqdm

from common import extract_scores, find_generated, load_tasks, response_path, write_scores
from prompts import PROMPT_CRITERIA, PROMPT_HEADER, build_middle

DEFAULT_MODEL = "Qwen/Qwen3-VL-32B-Instruct"
JUDGE_NAME = "qwen3vl"


def build_content(image_paths, instruction, generated_path, image_item):
    content = [{"type": "text", "text": PROMPT_HEADER}]
    content += [image_item(p) for p in image_paths]
    content.append({"type": "text", "text": build_middle(instruction)})
    content.append(image_item(generated_path))
    content.append({"type": "text", "text": PROMPT_CRITERIA})
    return content


class TransformersJudge:
    def __init__(self, model_id, attn_implementation, max_new_tokens):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.torch = torch
        kwargs = dict(dtype=torch.bfloat16, device_map="auto")
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.max_new_tokens = max_new_tokens

    def __call__(self, image_paths, instruction, generated_path):
        content = build_content(image_paths, instruction, generated_path,
                                lambda p: {"type": "image", "image": str(p)})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


class ServerJudge:
    def __init__(self, model_id, api_base, max_new_tokens):
        from openai import OpenAI
        self.client = OpenAI(base_url=api_base, api_key=os.getenv("QWEN_API_KEY", "EMPTY"))
        self.model_id, self.max_new_tokens = model_id, max_new_tokens

    @staticmethod
    def _item(p):
        mime = mimetypes.guess_type(str(p))[0] or "image/png"
        b64 = base64.b64encode(Path(p).read_bytes()).decode("utf-8")
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}

    def __call__(self, image_paths, instruction, generated_path):
        content = build_content(image_paths, instruction, generated_path, self._item)
        r = self.client.chat.completions.create(model=self.model_id, messages=[{"role": "user", "content": content}],
                                                max_tokens=self.max_new_tokens, temperature=0.0)
        return r.choices[0].message.content or ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="VIF-Bench dataset root (contains metadata.jsonl and tasks/)")
    ap.add_argument("--generated_dir", required=True, help="directory with <task_id>.png generated images")
    ap.add_argument("--model_name", default=None, help="name used in the output path (default: generated_dir name)")
    ap.add_argument("--output_dir", default="results")
    ap.add_argument("--subset", default="all", choices=["all", "vi_vs_ti"])
    ap.add_argument("--tasks", nargs="*", default=None, help="evaluate only these task ids, e.g. v6_n2_00")
    ap.add_argument("--judge_model", default=DEFAULT_MODEL, help=f"HF model id (default: {DEFAULT_MODEL})")
    ap.add_argument("--api_base", default=None, help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1")
    ap.add_argument("--attn_implementation", default=None, help='e.g. "flash_attention_2" (transformers backend)')
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    model_name = args.model_name or Path(args.generated_dir).resolve().name
    tasks = load_tasks(args.data_dir, args.subset, args.tasks)
    work, missing = [], 0
    for t in tasks:
        g = find_generated(args.generated_dir, t["id"])
        if g is None:
            missing += 1
            continue
        out = response_path(args.output_dir, model_name, JUDGE_NAME, t["id"])
        if out.exists() and not args.overwrite and extract_scores(out.read_text(encoding="utf-8")) is not None:
            continue
        work.append((t, g, out))
    print(f"{len(tasks)} tasks | {missing} without a generated image | {len(work)} to judge with {args.judge_model}")

    if work:
        judge = (ServerJudge(args.judge_model, args.api_base, args.max_new_tokens) if args.api_base
                 else TransformersJudge(args.judge_model, args.attn_implementation, args.max_new_tokens))
        failures = 0
        for t, g, out in tqdm(work, desc="judge[qwen3vl]"):
            for _ in range(args.max_retries):
                text = judge(t["images"], t["instruction"], g)
                if extract_scores(text) is not None:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(text, encoding="utf-8")
                    break
            else:
                failures += 1
                tqdm.write(f"{t['id']}: unparsable response")
        if failures:
            print(f"{failures} tasks failed; re-run the same command to retry them.")

    write_scores(args.output_dir, model_name, JUDGE_NAME, tasks)


if __name__ == "__main__":
    main()
