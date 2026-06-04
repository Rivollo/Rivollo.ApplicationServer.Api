from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, model_validator


class PointPrompt(BaseModel):
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    label: Literal[0, 1] = 1
    frame_index: Optional[int] = Field(default=None, ge=0)


class BoxPrompt(BaseModel):
    x_min: int = Field(..., ge=0)
    y_min: int = Field(..., ge=0)
    x_max: int = Field(..., ge=0)
    y_max: int = Field(..., ge=0)
    frame_index: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_box_bounds(self) -> "BoxPrompt":
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")
        if self.y_max <= self.y_min:
            raise ValueError("y_max must be greater than y_min")
        return self


class Sam2ImageSegmentRequest(BaseModel):
    image_url: HttpUrl
    prompts: list[PointPrompt] = Field(default_factory=list)
    box_prompts: list[BoxPrompt] = Field(default_factory=list)
    apply_mask: Optional[bool] = None
    sync_mode: Optional[bool] = None
    output_format: Literal["jpeg", "png", "webp"] = "png"

    @model_validator(mode="after")
    def require_segmentation_prompt(self) -> "Sam2ImageSegmentRequest":
        if not self.prompts and not self.box_prompts:
            raise ValueError("At least one point prompt or box prompt is required")
        return self


class SegmentedImage(BaseModel):
    url: str
    content_type: Optional[str] = None
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None


class Sam2ImageSegmentResponse(BaseModel):
    image: SegmentedImage


class Sam2AutoSegmentRequest(BaseModel):
    image_url: HttpUrl
    sync_mode: Optional[bool] = None
    output_format: Literal["jpeg", "png"] = "png"
    points_per_side: int = Field(default=32, ge=1)
    pred_iou_thresh: float = Field(default=0.88, ge=0.0, le=1.0)
    stability_score_thresh: float = Field(default=0.95, ge=0.0, le=1.0)
    min_mask_region_area: int = Field(default=100, ge=0)


class Sam2AutoSegmentResponse(BaseModel):
    combined_mask: SegmentedImage
    individual_masks: list[SegmentedImage] = Field(default_factory=list)
