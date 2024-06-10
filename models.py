from transformers import (
    AutoTokenizer,
    pipeline,
)
import torch
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset


class Aya:
    def __init__(self) -> None:
        self.model = pipeline(
            "text2text-generation",
            model="CohereForAI/aya-101",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def respond(self, dataset):
        responses = []
        for response in tqdm(
            self.model(
                KeyDataset(dataset, "question"),
                batch_size=8,
                max_new_tokens=128,
            ),
            total=len(dataset),
        ):
            responses.append(response[0]["generated_text"])
        return responses


class Falcon:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            "tiiuae/falcon-7b-instruct"
        )
        self.model = pipeline(
            "text-generation",
            model="tiiuae/falcon-7b-instruct",
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.tokenizer.pad_token_id = (
            self.model.model.config.eos_token_id
        )

    def process(self, example):
        example["question"] = self.model.tokenizer.apply_chat_template(
            [{"role": "user", "content": example["question"]}], tokenize=False
        )
        return example

    def respond(self, dataset):
        dataset = dataset.map(self.process)
        responses = []
        for response in tqdm(
            self.model(
                KeyDataset(dataset, "question"),
                batch_size=16,
                max_length=200,
                do_sample=False,
                num_beams=1,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
            ),
            total=len(dataset),
        ):
            responses.append(
                response[0]["generated_text"]
                .split("\n\nAssistant:")[1]
                .split("\nUser")[0]
            )
        return responses


class Llama2:
    def __init__(self) -> None:
        self.model = pipeline(
            "text-generation",
            model="meta-llama/Llama-2-7b-chat-hf",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.tokenizer.pad_token_id = (
            self.model.model.config.eos_token_id
        )

    def process(self, example):
        example["question"] = self.model.tokenizer.apply_chat_template(
            [{"role": "user", "content": example["question"]}], tokenize=False
        )
        return example

    def respond(self, dataset):
        dataset = dataset.map(self.process)
        responses = []
        for response in tqdm(
            self.model(
                KeyDataset(dataset, "question"),
                batch_size=4,
                do_sample=False,
                num_beams=1,
            ),
            total=len(dataset),
        ):
            responses.append(response[0]["generated_text"].split("[/INST]")[1])
        return responses


class Mistral:
    def __init__(self) -> None:
        self.model = pipeline(
            "text-generation",
            model="mistralai/Mistral-7B-Instruct-v0.2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.tokenizer.pad_token_id = (
            self.model.model.config.eos_token_id
        )

    def process(self, example):
        example["question"] = self.model.tokenizer.apply_chat_template(
            [{"role": "user", "content": example["question"]}], tokenize=False
        )
        return example

    def respond(self, dataset):
        dataset = dataset.map(self.process)
        responses = []
        for response in tqdm(
            self.model(
                KeyDataset(dataset, "question"),
                batch_size=16,
                max_new_tokens=1000,
                do_sample=False,
                num_beams=1,
            ),
            total=len(dataset),
        ):
            try:
                responses.append(
                    response[0]["generated_text"].split("[/INST] ")[1]
                )
            except:
                responses.append(
                    response[0]["generated_text"].split("[/INST]")[1]
                )
        return responses


class Wizard:
    def __init__(self) -> None:
        self.model = pipeline(
            "text-generation",
            model="wizardlm-7b",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def process(self, example):
        example["question"] = self.model.tokenizer.apply_chat_template(
            [{"role": "user", "content": example["question"]}], tokenize=False
        )
        return example

    def respond(self, dataset):
        dataset = dataset.map(self.process)
        responses = []
        for response in tqdm(
            self.model(
                KeyDataset(dataset, "question"),
                batch_size=4,
                do_sample=False,
                num_beams=1,
                max_new_tokens=2048,
            ),
            total=len(dataset),
        ):
            responses.append(
                response[0]["generated_text"]
                .split("Response:\n")[1]
                .split("\n\n###")[0]
            )
        return responses


class Zephyr:
    def __init__(self) -> None:
        self.model = pipeline(
            "text-generation",
            model="HuggingFaceH4/zephyr-7b-beta",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def process(self, example):
        example["question"] = self.model.tokenizer.apply_chat_template(
            [{"role": "user", "content": example["question"]}], tokenize=False
        )
        return example

    def respond(self, dataset):
        dataset = dataset.map(self.process)
        responses = []
        for response in tqdm(
            self.model(
                KeyDataset(dataset, "question"),
                batch_size=32,
                max_new_tokens=256,
                do_sample=False,
                num_beams=1,
            ),
            total=len(dataset),
        ):
            responses.append(
                response[0]["generated_text"].split("<|assistant|>\n")[1]
            )
        return responses
