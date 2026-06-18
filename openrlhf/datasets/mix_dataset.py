import copy

from torch.utils.data import Dataset
from tqdm import tqdm


def _str_to_content_list(text: str):
    """Convert a string with ``<image>`` tags to a VLM content list.

    E.g. ``"<image>Find x."`` → ``[{"type": "image"}, {"type": "text", "text": "Find x."}]``
    Returns the original string unchanged when no ``<image>`` tags are present.
    """
    if "<image>" not in text:
        return text
    parts = text.split("<image>")
    content = []
    for i, part in enumerate(parts):
        if i > 0:
            content.append({"type": "image"})
        stripped = part.strip()
        if stripped:
            content.append({"type": "text", "text": stripped})
    return content


def preprocess_data(data, input_template=None, input_key_proxy="input_proxy", input_key_full="input_full", label_key=None, apply_chat_template=None) -> str:
    if apply_chat_template:
        chat_proxy = data[input_key_proxy]
        chat_full = data[input_key_full]

        if isinstance(chat_proxy, str):
            chat_proxy = [{"role": "user", "content": _str_to_content_list(chat_proxy)}]
        elif isinstance(chat_proxy, list):
            # Deep copy to avoid mutating the original dataset entries.
            chat_proxy = copy.deepcopy(chat_proxy)
            for msg in chat_proxy:
                if isinstance(msg.get("content"), str):
                    msg["content"] = _str_to_content_list(msg["content"])
        prompt_proxy = apply_chat_template(chat_proxy, tokenize=False, add_generation_prompt=True)

        if isinstance(chat_full, str):
            chat_full = [{"role": "user", "content": _str_to_content_list(chat_full)}]
        elif isinstance(chat_full, list):
            # Deep copy to avoid mutating the original dataset entries.
            chat_full = copy.deepcopy(chat_full)
            for msg in chat_full:
                if isinstance(msg.get("content"), str):
                    msg["content"] = _str_to_content_list(msg["content"])
        prompt_full = apply_chat_template(chat_full, tokenize=False, add_generation_prompt=True)
    else:
        prompt_proxy = data[input_key_proxy]
        prompt_full = data[input_key_full]
        if input_template:
            prompt_proxy = input_template.format(prompt_proxy)
            prompt_full = input_template.format(prompt_full)

    # for Reinforced Fine-tuning
    label = "" if label_key is None else data[label_key]
    return prompt_proxy, prompt_full, label


class MixDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key_proxy = getattr(self.strategy.args.data, "input_key_proxy", None)
        input_key_full = getattr(self.strategy.args.data, "input_key_full", None)
        label_key = getattr(self.strategy.args.data, "label_key", None)
        apply_chat_template = getattr(self.strategy.args.data, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.image_key = getattr(self.strategy.args.data, "image_key", "images")

        self.prompts_proxy = []
        self.prompts_full = []
        self.labels = []
        self.images = []
        self.datasources = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt_proxy, prompt_full, label = preprocess_data(data, input_template, input_key_proxy, input_key_full, label_key, apply_chat_template)
            self.prompts.append(prompt_proxy)
            self.prompts.append(prompt_full)
            self.labels.append(label)
            self.images.append(data.get(self.image_key, None))
            self.datasources.append(data.get("datasource", "default"))

    def __len__(self):
        length = len(self.labels)
        return length

    def __getitem__(self, idx):
        return self.datasources[idx], self.prompts_proxy[idx], self.prompts_full[idx], self.labels[idx], self.images[idx]

    def collate_fn(self, item_list):
        datasources = []
        prompts_proxy = []
        prompts_full = []
        labels = []
        images = []
        for datasource, prompt_proxy, prompt_full, label, img in item_list:
            datasources.append(datasource)
            prompts_proxy.append(prompt_proxy)
            prompts_full.append(prompt_full)
            labels.append(label)
            images.append(img)

        return datasources, prompts_proxy, prompts_full, labels, images
