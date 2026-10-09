"""
Ollama nodes for ComfyUI
"""
import requests
import json
import base64
from io import BytesIO
import numpy as np
from PIL import Image
import torch


class OllamaTextGenerator:
    """
    Generate or enhance text using Ollama models.
    Perfect for prompt enhancement, story generation, etc.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Enhance this prompt for image generation: a cat"}),
                "model": (["qwen2.5:7b", "llama3.1:8b", "gemma2", "mistral", "qwen2.5", "llama3.2"], ),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful AI assistant that enhances image generation prompts."}),
                "ollama_host": ("STRING", {"default": "http://localhost:11434"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "Ollama/Kindred"

    def generate(self, prompt, model, temperature, max_tokens, system_prompt="", ollama_host="http://localhost:11434"):
        try:
            url = f"{ollama_host}/api/generate"

            payload = {
                "model": model,
                "prompt": prompt,
                "system": system_prompt if system_prompt else None,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }

            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()

            result = response.json()
            generated_text = result.get("response", "")

            print(f"[Ollama] Generated {len(generated_text)} characters")
            return (generated_text,)

        except Exception as e:
            error_msg = f"Ollama error: {str(e)}"
            print(error_msg)
            return (error_msg,)


class OllamaVisionAnalyzer:
    """
    Analyze images using Ollama vision models (qwen3-vl, llama3.2-vision).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "Describe this image in detail."}),
                "model": (["qwen3-vl:32b", "llama3.2-vision:11b", "llama3.2-vision"], ),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "ollama_host": ("STRING", {"default": "http://localhost:11434"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "analyze"
    CATEGORY = "Ollama/Kindred"

    def analyze(self, image, prompt, model, temperature, ollama_host="http://localhost:11434"):
        try:
            # Convert ComfyUI image tensor to PIL Image
            image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)

            # Convert to base64
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            url = f"{ollama_host}/api/generate"

            payload = {
                "model": model,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False,
                "options": {
                    "temperature": temperature,
                }
            }

            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()

            result = response.json()
            description = result.get("response", "")

            print(f"[Ollama Vision] Generated description: {len(description)} characters")
            return (description,)

        except Exception as e:
            error_msg = f"Ollama Vision error: {str(e)}"
            print(error_msg)
            return (error_msg,)


class OllamaPromptEnhancer:
    """
    Specialized node for enhancing prompts for image generation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "simple_prompt": ("STRING", {"multiline": True, "default": "a cat"}),
                "style": (["detailed", "artistic", "photorealistic", "cinematic", "anime", "concept art"], ),
                "model": (["qwen2.5:7b", "llama3.1:8b", "mistral"], ),
            },
            "optional": {
                "additional_instructions": ("STRING", {"multiline": True, "default": ""}),
                "ollama_host": ("STRING", {"default": "http://localhost:11434"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "original_prompt")
    FUNCTION = "enhance"
    CATEGORY = "Ollama/Kindred"

    def enhance(self, simple_prompt, style, model, additional_instructions="", ollama_host="http://localhost:11434"):
        try:
            system_prompt = f"""You are an expert at enhancing prompts for AI image generation.
Given a simple prompt, expand it into a detailed, {style} prompt that will produce stunning images.

Guidelines:
- Add relevant artistic details, lighting, composition
- Include technical terms (8k, highly detailed, etc.) when appropriate
- Keep the core concept intact
- Make it concise but descriptive (50-150 words)
- Do NOT include explanations, just output the enhanced prompt

{additional_instructions}"""

            url = f"{ollama_host}/api/generate"

            payload = {
                "model": model,
                "prompt": f"Simple prompt: {simple_prompt}\n\nEnhanced prompt:",
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "num_predict": 256,
                }
            }

            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()

            result = response.json()
            enhanced = result.get("response", "").strip()

            print(f"[Ollama Enhancer] {simple_prompt} -> {enhanced[:100]}...")
            return (enhanced, simple_prompt)

        except Exception as e:
            error_msg = f"Enhancement error: {str(e)}"
            print(error_msg)
            return (simple_prompt, simple_prompt)


class OllamaChat:
    """
    Interactive chat node with conversation history.
    Perfect for building storylines and iterative creative work.
    """

    def __init__(self):
        self.history = []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "message": ("STRING", {"multiline": True, "default": "Tell me a story"}),
                "model": (["qwen2.5:7b", "llama3.1:8b", "gemma2", "mistral", "nemotron:70b"], ),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a creative storyteller and worldbuilding expert."}),
                "reset_history": ("BOOLEAN", {"default": False}),
                "ollama_host": ("STRING", {"default": "http://localhost:11434"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "full_conversation")
    FUNCTION = "chat"
    CATEGORY = "Ollama/Kindred"

    def chat(self, message, model, temperature, system_prompt="", reset_history=False, ollama_host="http://localhost:11434"):
        try:
            if reset_history:
                self.history = []

            url = f"{ollama_host}/api/chat"

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.extend(self.history)
            messages.append({"role": "user", "content": message})

            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                }
            }

            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()

            result = response.json()
            assistant_message = result.get("message", {}).get("content", "")

            # Update history
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": assistant_message})

            # Create full conversation string
            full_conv = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in self.history])

            print(f"[Ollama Chat] History length: {len(self.history)} messages")
            return (assistant_message, full_conv)

        except Exception as e:
            error_msg = f"Chat error: {str(e)}"
            print(error_msg)
            return (error_msg, error_msg)


class OllamaModelList:
    """
    Fetch available models from Ollama server dynamically.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "ollama_host": ("STRING", {"default": "http://localhost:11434"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("models_list",)
    FUNCTION = "list_models"
    CATEGORY = "Ollama/Kindred"
    OUTPUT_NODE = True

    def list_models(self, ollama_host="http://localhost:11434"):
        try:
            url = f"{ollama_host}/api/tags"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            result = response.json()
            models = result.get("models", [])

            model_list = "\n".join([f"- {m['name']} ({m.get('size', 'unknown')} bytes)" for m in models])

            print(f"[Ollama] Found {len(models)} models")
            return (model_list,)

        except Exception as e:
            error_msg = f"Model list error: {str(e)}"
            print(error_msg)
            return (error_msg,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "OllamaTextGenerator": OllamaTextGenerator,
    "OllamaVisionAnalyzer": OllamaVisionAnalyzer,
    "OllamaPromptEnhancer": OllamaPromptEnhancer,
    "OllamaChat": OllamaChat,
    "OllamaModelList": OllamaModelList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaTextGenerator": "Ollama Text Generator",
    "OllamaVisionAnalyzer": "Ollama Vision Analyzer",
    "OllamaPromptEnhancer": "Ollama Prompt Enhancer",
    "OllamaChat": "Ollama Chat",
    "OllamaModelList": "Ollama Model List",
}
