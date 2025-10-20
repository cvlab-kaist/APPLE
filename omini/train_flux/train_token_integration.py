import torch
import os
import random

from PIL import Image, ImageDraw

from datasets import load_dataset

from .trainer import OminiModel, get_config, train
from ..pipeline.flux_omini import Condition, generate
from .train_spatial_alignment import ImageConditionDataset


class TokenIntergrationDataset(ImageConditionDataset):
    def __getitem__(self, idx):
        image = self.base_dataset[idx]["jpg"]
        image = image.resize(self.target_size).convert("RGB")
        description = self.base_dataset[idx]["json"]["prompt"]

        assert self.condition_type == "token_intergration"
        assert (
            image.size[0] % 16 == 0 and image.size[1] % 16 == 0
        ), "Condition size must be divisible by 16"

        # Randomly drop text or image (for training)
        description = "" if random.random() < self.drop_text_prob else description

        # Generate a latent mask
        w, h = image.size[0] // 16, image.size[1] // 16
        while True:
            x1, x2 = sorted([random.randint(0, w), random.randint(0, w)])
            y1, y2 = sorted([random.randint(0, h), random.randint(0, h)])
            is_zero = x1 == x2 or y1 == y2
            is_full = x1 == 0 and y1 == 0 and x2 == w and y2 == h
            if not (is_zero or is_full):
                break
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([x1, y1, x2, y2], fill=255)
        if random.random() > 0.5:
            mask = Image.eval(mask, lambda a: 255 - a)
        mask = self.to_tensor(mask).to(bool).reshape(-1)

        return {
            "image": self.to_tensor(image),
            "image_latent_mask": torch.logical_not(mask),
            "condition_0": self.to_tensor(image),
            "condition_type_0": self.condition_type,
            "condition_latent_mask_0": mask,
            "description": description,
        }


@torch.no_grad()
def test_function(model, save_path, file_name):
    target_size = model.training_config["dataset"]["target_size"]

    condition_type = model.training_config["condition_type"]
    test_list = []

    # Generate two masks to test inpainting and outpainting.
    mask1 = torch.ones((32, 32), dtype=bool)
    mask1[8:24, 8:24] = False
    mask2 = torch.logical_not(mask1)

    image = Image.open("assets/vase_hq.jpg").resize(target_size)
    condition1 = Condition(
        image, model.adapter_names[2], latent_mask=mask1, is_complement=True
    )
    condition2 = Condition(
        image, model.adapter_names[2], latent_mask=mask2, is_complement=True
    )
    test_list.append((condition1, "A beautiful vase on a table.", mask2))
    test_list.append((condition2, "A beautiful vase on a table.", mask1))

    os.makedirs(save_path, exist_ok=True)
    for i, (condition, prompt, latent_mask) in enumerate(test_list):
        generator = torch.Generator(device=model.device)
        generator.manual_seed(42)

        res = generate(
            model.flux_pipe,
            prompt=prompt,
            conditions=[condition],
            height=target_size[0],
            width=target_size[1],
            generator=generator,
            model_config=model.model_config,
            kv_cache=model.model_config.get("independent_condition", False),
            latent_mask=latent_mask,
        )
        file_path = os.path.join(save_path, f"{file_name}_{condition_type}_{i}.jpg")
        res.images[0].save(file_path)


def main():
    # Initialize
    config = get_config()
    training_config = config["train"]
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    # Load dataset text-to-image-2M
    dataset = load_dataset(
        "webdataset",
        data_files={"train": training_config["dataset"]["urls"]},
        split="train",
        cache_dir="cache/t2i2m",
        num_proc=32,
    )
    dataset = TokenIntergrationDataset(
        dataset,
        condition_size=training_config["dataset"]["condition_size"],
        target_size=training_config["dataset"]["target_size"],
        condition_type=training_config["condition_type"],
        drop_text_prob=training_config["dataset"]["drop_text_prob"],
        drop_image_prob=training_config["dataset"]["drop_image_prob"],
        position_scale=training_config["dataset"].get("position_scale", 1.0),
    )

    # Initialize model
    trainable_model = OminiModel(
        flux_pipe_id=config["flux_path"],
        lora_config=training_config["lora_config"],
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        adapter_names=[None, None, "default"],
    )

    train(dataset, trainable_model, config, test_function)


if __name__ == "__main__":
    main()
