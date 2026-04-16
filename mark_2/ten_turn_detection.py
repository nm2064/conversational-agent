"""
TEN Turn Detection — wrapper around TEN-framework/TEN_Turn_Detection (Qwen 7B).
Returns one of: "finished", "unfinished", "wait".
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "TEN-framework/TEN_Turn_Detection"


class TENTurnDetector:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None

    def load(self):
        print("Loading TEN Turn Detection model (Qwen 7B)...", flush=True)

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            dtype = torch.float16
            print("  Device: MPS (Apple Silicon GPU)")
        else:
            self.device = torch.device("cpu")
            dtype = torch.float32
            print("  Device: CPU")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, trust_remote_code=True, dtype=dtype
        ).to(self.device)
        self.model.eval()
        print("TEN Turn Detection model loaded.", flush=True)

    def detect(self, text):
        """
        Classify the user's speech buffer.
        Returns: "finished" | "unfinished" | "wait"
        """
        if not text or not text.strip():
            return "unfinished"

        messages = [{"role": "user", "content": text.strip()}]
        encoded = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        # transformers ≥5 returns BatchEncoding; older returns a tensor
        if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
            input_ids = encoded["input_ids"].to(self.device)
        else:
            input_ids = encoded.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=1,
                do_sample=True,
                top_p=0.1,
                temperature=0.1,
            )

        result = self.tokenizer.decode(
            outputs[0][input_ids.shape[-1]:], skip_special_tokens=True
        ).strip().lower()

        if result in ("finished", "unfinished", "wait"):
            return result
        # Default to finished if model returns something unexpected
        return "finished"
