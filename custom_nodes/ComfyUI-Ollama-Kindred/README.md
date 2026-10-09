# ComfyUI-Ollama-Kindred

A comprehensive Ollama integration for ComfyUI, designed for creative workflows, storytelling, and multi-modal AI applications.

## Features

### 🎨 Current Nodes

1. **Ollama Text Generator**
   - Generate or enhance text using any Ollama model
   - Customizable temperature and token limits
   - System prompts for different personas

2. **Ollama Vision Analyzer**
   - Analyze images using vision models (qwen3-vl, llama3.2-vision)
   - Image captioning and description
   - Visual question answering

3. **Ollama Prompt Enhancer**
   - Specialized prompt enhancement for image generation
   - Multiple style presets (artistic, photorealistic, cinematic, etc.)
   - Maintains core concept while adding details

4. **Ollama Chat**
   - Conversational interface with history
   - Perfect for storytelling and iterative creation
   - Maintains context across multiple exchanges

5. **Ollama Model List**
   - Dynamically fetch available models from your Ollama server
   - Useful for debugging and model management

## Installation

1. The custom node is already in: `~/Documents/ComfyUI/custom_nodes/ComfyUI-Ollama-Kindred/`
2. Restart ComfyUI
3. Nodes will appear under "Ollama/Kindred" category

## Requirements

- Ollama running locally (default: http://localhost:11434)
- Installed models (you have 30+ already!)

## Usage Examples

### Simple Prompt Enhancement
```
OllamaPromptEnhancer -> CLIPTextEncode -> KSampler
```

### Image Analysis Workflow
```
LoadImage -> OllamaVisionAnalyzer -> Display Text
```

### Iterative Storytelling
```
OllamaChat -> OllamaPromptEnhancer -> Image Generation
```

## Roadmap (Future Development)

- [ ] **Storyline Generator**: Multi-panel comic/storyboard creation
- [ ] **Video Frame Sequencing**: Text-to-video workflows
- [ ] **World Model Integration**: Physics simulation and prediction
- [ ] **Batch Processing**: Process multiple prompts/images
- [ ] **Memory System**: Long-term context across sessions
- [ ] **Custom Model Training**: Fine-tune on your data
- [ ] **Multi-agent Systems**: Collaborative AI workflows

## Your Ollama Models

Currently available:
- qwen2.5:7b, llama3.1:8b, gemma2, mistral
- nemotron:70b (large model)
- qwen3-vl:32b, llama3.2-vision:11b (vision models)
- Many custom models (kaelen, watson, aion, etc.)

## Development

Built with extensibility in mind. Easy to add new nodes for:
- Custom model workflows
- Specialized creative tasks
- Integration with other ComfyUI nodes
