import base64
from abc import ABC, abstractmethod
from typing import List, Union, Dict
from dataclasses import dataclass
from openai import OpenAI


@dataclass
class Image:
    """Class to represent an image input"""

    content: str  # Either URL or base64 string
    is_url: bool = True


class VLMResponse:
    """Class to represent VLM response"""

    def __init__(self, raw_response: any, text: str):
        self.raw_response = raw_response
        self.text = text

    def __str__(self):
        return self.text


class VLMBase(ABC):
    """Abstract base class for VLM implementations"""

    @abstractmethod
    def analyze(self, prompt: str, images: List[Image]) -> VLMResponse:
        """Analyze images with given prompt"""
        pass


class OpenAIVLM(VLMBase):
    """OpenAI's VLM implementation"""

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def analyze(self, prompt: str, images: List[Image]) -> VLMResponse:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ]

        # Add images to the message
        for image in images:
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": (
                        image.content
                        if image.is_url
                        else f"data:image/jpeg;base64,{image.content}"
                    )
                },
            }
            messages[0]["content"].append(image_content)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=300,
        )

        return VLMResponse(
            raw_response=response, text=response.choices[0].message.content
        )


class VLM:
    """Main VLM interface"""

    @staticmethod
    def from_url(url: str) -> Image:
        """Create Image instance from URL"""
        return Image(content=url, is_url=True)

    @staticmethod
    def from_file(file_path: str) -> Image:
        """Create Image instance from local file"""
        with open(file_path, "rb") as image_file:
            base64_string = base64.b64encode(image_file.read()).decode("utf-8")
        return Image(content=base64_string, is_url=False)
