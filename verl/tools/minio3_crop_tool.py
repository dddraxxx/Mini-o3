# Copyright 2026 Mini-o3 contributors
#
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from typing import Any

from PIL import Image

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    ToolResponse,
)


class MiniO3CropTool(BaseTool):
    """Mini-o3 visual grounding crop tool for official verl AgentLoop."""

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="tool_crop",
                description=(
                    "Zoom into a specific rectangular region of the original image or a previous "
                    "observation image and return the cropped zoom-in image. Coordinates are "
                    "relative by default, matching Mini-o3 grounding format."
                ),
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "bbox_2d": OpenAIFunctionPropertySchema(
                            type="array",
                            description="Bounding box [x0, y0, x1, y1].",
                        ),
                        "source": OpenAIFunctionPropertySchema(
                            type="string",
                            description="original_image or observation_i.",
                        ),
                    },
                    required=["bbox_2d"],
                ),
            ),
        )

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        model_images = getattr(agent_data, "image_data", None)
        raw_images = getattr(agent_data, "raw_image_data", None)
        use_raw_image_bank = bool(self.config.get("use_raw_image_bank", True))
        raw_count = len(raw_images or [])
        model_count = len(model_images or [])
        num_images = max(raw_count if use_raw_image_bank else 0, model_count)
        if num_images <= 0:
            return ToolResponse(text="ERROR occurs during grounding. No image is available."), 0.0, {}

        try:
            image_index = self._resolve_image_index(parameters, num_images)
            if use_raw_image_bank and raw_images and image_index < raw_count:
                images = raw_images
                source_space = "raw"
            else:
                images = model_images
                source_space = "runtime"
            if not images or image_index >= len(images):
                raise ValueError(f"img_idx {image_index} is out of range for {len(images or [])} image(s).")
            image = images[image_index]
            if not isinstance(image, Image.Image):
                raise TypeError(f"Expected PIL.Image.Image, got {type(image).__name__}")

            bbox = self._resolve_bbox(parameters.get("bbox_2d"), image.size)
            crop = self._crop_image(image, bbox)
        except Exception as exc:
            return (
                ToolResponse(text=f"ERROR occurs during grounding. Error Information: {exc}.\n"),
                0.0,
                {"minio3_crop/error": str(exc)},
            )

        metrics = {
            "minio3_crop/source_index": image_index,
            "minio3_crop/source_space": source_space,
            "minio3_crop/raw_bank_enabled": int(use_raw_image_bank),
            "minio3_crop/raw_bank_count": raw_count,
            "minio3_crop/runtime_bank_count": model_count,
            "minio3_crop/source_w": image.width,
            "minio3_crop/source_h": image.height,
            "minio3_crop/x0": bbox[0],
            "minio3_crop/y0": bbox[1],
            "minio3_crop/x1": bbox[2],
            "minio3_crop/y1": bbox[3],
            "minio3_crop/crop_w": crop.width,
            "minio3_crop/crop_h": crop.height,
        }
        return ToolResponse(image=[crop]), 0.0, metrics

    def _resolve_image_index(self, parameters: dict[str, Any], num_images: int) -> int:
        if "img_idx" in parameters:
            try:
                image_index = int(float(parameters["img_idx"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"img_idx must be a number, got {parameters['img_idx']!r}") from exc
            if image_index < 0 or image_index >= num_images:
                raise ValueError(f"img_idx {image_index} is out of range for {num_images} image(s).")
            return image_index
        return self._resolve_source(parameters.get("source", "original_image"), num_images)

    @staticmethod
    def _resolve_source(source: Any, num_images: int) -> int:
        if source in (None, "", "original_image"):
            return 0
        if isinstance(source, int):
            image_index = source
        elif isinstance(source, str) and source.startswith("observation_"):
            image_index = int(source.removeprefix("observation_"))
        else:
            raise ValueError(f"Unsupported source: {source!r}")

        if image_index < 0 or image_index >= num_images:
            raise ValueError(f"Source {source!r} is out of range for {num_images} image(s).")
        return image_index

    def _resolve_bbox(self, bbox: Any, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
        if not isinstance(bbox, list | tuple) or len(bbox) != 4:
            raise ValueError(f"bbox_2d must be a list of four numbers, got {bbox!r}")

        w, h = image_size
        coords = [float(v) for v in bbox]
        if self.config.get("use_relative_coordinates", True):
            coordinate_scale = float(self.config.get("coordinate_scale", 1.0))
            if coordinate_scale <= 0:
                raise ValueError(f"coordinate_scale must be positive, got {coordinate_scale}")
            coords = [
                coords[0] / coordinate_scale * w,
                coords[1] / coordinate_scale * h,
                coords[2] / coordinate_scale * w,
                coords[3] / coordinate_scale * h,
            ]

        x0, y0, x1, y1 = coords
        x0 = max(0, min(int(x0), w - 1))
        y0 = max(0, min(int(y0), h - 1))
        x1 = max(1, min(int(x1), w))
        y1 = max(1, min(int(y1), h))
        if x0 >= x1 or y0 >= y1:
            raise ValueError(f"Invalid bounding box after clamping: {[x0, y0, x1, y1]}")

        width = x1 - x0
        height = y1 - y0
        max_aspect_ratio = float(self.config.get("max_aspect_ratio", 200.0))
        if width / height > max_aspect_ratio or height / width > max_aspect_ratio:
            raise ValueError(f"Bounding box aspect ratio is too large: {[x0, y0, x1, y1]}")

        return x0, y0, x1, y1

    def _crop_image(self, image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
        crop = image.crop(bbox)
        resize = float(self.config.get("resize", 1.0))
        min_crop_size = int(self.config.get("min_crop_size", 28))
        target_w = max(int(crop.width * resize), min_crop_size)
        target_h = max(int(crop.height * resize), min_crop_size)
        if target_w == crop.width and target_h == crop.height:
            return crop

        return crop.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
