import base64
import os
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Optional
from openai import OpenAI
from anthropic import Anthropic


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
    def analyze(
        self, prompt: str, images: List[Dict], system_prompt: Optional[str] = None
    ) -> VLMResponse:
        """Analyze images with given prompt"""
        pass


class OpenAIVLM(VLMBase):
    """OpenAI's VLM implementation"""

    def __init__(self, api_key: str = None, model: str = "gpt-4-vision-preview"):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def analyze(
        self, prompt: str, images: List[Dict], system_prompt: Optional[str] = None
    ) -> VLMResponse:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_message = {"role": "user", "content": [{"type": "text", "text": prompt}]}

        for image in images:
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": image.get("url")
                    or f"data:image/jpeg;base64,{image['base64']}"
                },
            }
            user_message["content"].append(image_content)

        messages.append(user_message)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1024,
        )

        return VLMResponse(
            raw_response=response, text=response.choices[0].message.content
        )


class AnthropicVLM(VLMBase):
    """Anthropic's Claude Vision implementation"""

    def __init__(self, api_key: str = None, model: str = "claude-3-5-sonnet-20241022"):
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_images = 20

    def analyze(
        self, prompt: str, images: List[Dict], system_prompt: Optional[str] = None
    ) -> VLMResponse:
        if len(images) > self.max_images:
            raise ValueError(f"Maximum {self.max_images} images allowed per request")

        message_content = []

        # Add numbered image labels if multiple images
        if len(images) > 1:
            for idx, image in enumerate(images, 1):
                message_content.extend(
                    [
                        {"type": "text", "text": f"Image {idx}:"},
                        self._format_image(image),
                    ]
                )
        else:
            message_content.append(self._format_image(images[0]))

        # Add the prompt text after images
        message_content.append({"type": "text", "text": prompt})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": message_content}],
        )

        return VLMResponse(raw_response=response, text=response.content[0].text)

    def _format_image(self, image: Dict) -> Dict:
        """Format image for Anthropic API"""
        if "url" in image:
            return {"type": "image", "source": {"type": "url", "url": image["url"]}}
        else:
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.get("media_type", "image/jpeg"),
                    "data": image["base64"],
                },
            }


class VLM:
    """Helper class for creating image inputs"""

    @staticmethod
    def from_url(url: str) -> Dict:
        """Create image input from URL"""
        return {"url": url}

    @staticmethod
    def from_file(file_path: str) -> Dict:
        """Create image input from local file"""
        # Determine media type from file extension
        if file_path.lower().endswith(".png"):
            media_type = "image/png"
        elif file_path.lower().endswith((".jpg", ".jpeg")):
            media_type = "image/jpeg"
        elif file_path.lower().endswith(".webp"):
            media_type = "image/webp"
        elif file_path.lower().endswith(".gif"):
            media_type = "image/gif"
        else:
            media_type = "image/jpeg"  # default

        with open(file_path, "rb") as image_file:
            base64_string = base64.b64encode(image_file.read()).decode("utf-8")

        return {"base64": base64_string, "media_type": media_type}
