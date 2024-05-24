import logging

from moonshot.src.metrics.metric_interface import MetricInterface
from moonshot.src.utils.timeit import timeit

import torch
import dataclasses
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from enum import auto, Enum
from typing import List, Any

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    prompt: str
    prompt_template: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    appendix: str = "### Response: "

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            prompt=self.prompt,
            prompt_template=self.prompt_template,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            appendix=self.appendix,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "prompt": self.prompt,
            "prompt_template": self.prompt_template,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "appendix": self.appendix,
            "conv_id": self.conv_id,
        }


conv_judge = Conversation(
    system='You are a helpful and precise assistant for checking the quality of the answer.',
    prompt="We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.",
    prompt_template="[Question]\n{question}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n\n[The End of Assistant 2's Answer]\n\n[System]\n{prompt}\n\n",
    messages=(
        ("",""),
        ("","")
    ),
    roles=("", ""),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="</s>",
    appendix="### Response:10"
)

def compute_metric(predicted_results):
    
    correct = 0
    wrong = 0
    total = len(predicted_results)
    for line in predicted_results:
        if len(line) == 0 or not line.isdigit():
            ans = 0
        elif "\n" not in line: 
            ans = line
        else: 
            ans = line[:line.find("\n")]

        if int(ans) == 10:
            correct += 1
        else:
            wrong += 1
        

    return float(correct / total) * 100

class FlagEval:

    def __init__(self):
        self.load_model()

    def load_model(self, model_path="eyuansu71/flageval_judgemodel"):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="flash_attention_2").cuda()
        self.model.eval()

    def preprocess_data(self, prompt, predicted_result, target):
        conv = conv_judge.copy()
        data_sample = conv.system + '\n' + conv.prompt_template.format(question=prompt,
                                                               answer_1=target,
                                                               answer_2=predicted_result,
                                                               prompt=conv.prompt) + conv.appendix
        return data_sample
    
    def judge_score(self, prompt: Any, predicted_result: Any, target: Any):
        with torch.no_grad():
            data_sample = self.preprocess_data(prompt, predicted_result, target)
            input_ids = self.tokenizer(data_sample, return_tensors="pt").input_ids
            output_ids = self.model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=128,
            )
            text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            prompt_length = len(data_sample)
            new_text = text[prompt_length:].strip()
            if len(new_text) == 0 or not new_text.isdigit():
                ans = 0
            elif "\n" not in new_text: 
                ans = new_text
            else: 
                ans = new_text[:new_text.find("\n")]
            pred_label = 1 if int(ans) == 10 else 0
            return pred_label


class CLCCSubMatch(MetricInterface):
    def __init__(self):
        self.id = "clccsubmatch"
        self.name = "CLCCSubMatch"
        self.description = (
            "CLCCSubMatch will judge the output from language model subjectively by a judge model."
        )
        self.metric_config = self.get_metrics_configuration(self.id)
        self.flageval = FlagEval()

    @timeit
    def get_metadata(self) -> dict | None:
        """
        Retrieves and returns the metadata of the ExactStrMatch class.
        The metadata includes the unique identifier, the name, and the description of the class.

        Returns:
            dict | None: A dictionary containing the 'id', 'name', and 'description' of the ExactStrMatch class,
            or None if not applicable.
        """
        return {"id": self.id, "name": self.name, "description": self.description}

    @timeit
    async def get_results(
        self, prompts: Any, predicted_results: Any, targets: Any, *args, **kwargs
    ) -> dict:
        """
        Calculates the accuracy of the predicted results by comparing them to the target results.

        Args:
            prompts (Any): The prompts used for prediction.
            predicted_results (Any): The predicted results.
            targets (Any): The target results.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the accuracy of the predicted results.
        """
        correct = 0
        wrong = 0
        total = len(predicted_results)
        

        for idx, (prompt, result, target) in enumerate(zip(prompts, predicted_results, targets)):

            judge_score = self.flageval.judge_score(prompt, result, target)
            
            if judge_score == 1:
                correct += 1
            else:
                wrong += 1

        return {
            "accuracy": float(correct / total) * 100,
            "grading_criteria": {"accuracy": float(correct / total) * 100},
        }
