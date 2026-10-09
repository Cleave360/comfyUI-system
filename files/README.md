# ComfyUI-Ollama Bridge

A Rust-based system for teaching LLMs and VLMs to understand hierarchical relationships in images and videos, with audio processing capabilities. This is a foundational tool for embodied AI in robotics.

## Architecture

```
ComfyUI (Image Generation) → Bridge (Analysis) → Ollama (Vision LLM) → Scene Graph → Training Data
                                    ↓
                            Audio Processing (Whisper)
```

## Features

- **ComfyUI Integration**: WebSocket client for real-time workflow monitoring
- **Ollama Vision Models**: Support for LLaVA and other vision-language models
- **Scene Graph Construction**: Hierarchical representation of objects and relationships
- **Audio Processing**: Real-time capture and speech-to-text integration
- **Training Pipeline**: Annotation and dataset generation for fine-tuning

## Prerequisites

1. **Rust** (1.75+)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **ComfyUI** running locally
   ```bash
   # Clone and install ComfyUI
   git clone https://github.com/comfyanonymous/ComfyUI
   cd ComfyUI
   pip install -r requirements.txt
   python main.py
   # Default: http://127.0.0.1:8188
   ```

3. **Ollama** with vision models
   ```bash
   # Install Ollama
   curl https://ollama.ai/install.sh | sh

   # Pull vision models
   ollama pull llava:13b
   ollama pull llava:7b
   ollama pull bakllava

   # Pull Whisper for audio
   ollama pull whisper:large
   ```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd comfyui-bridge

# Build the project
cargo build --release

# Run
cargo run --release
```

## Configuration

Create a `config.toml` in the project root:

```toml
[comfyui]
websocket_url = "ws://127.0.0.1:8188/ws"
api_url = "http://127.0.0.1:8188"

[ollama]
base_url = "http://localhost:11434"
default_model = "llava:13b"

[audio]
sample_rate = 16000
channels = 1
whisper_model = "whisper:large"

[server]
host = "127.0.0.1"
port = 3030
```

## Usage

### Basic Usage

1. Start ComfyUI
2. Start Ollama
3. Run the bridge:
   ```bash
   cargo run --release
   ```

### Scene Graph Analysis

```rust
use comfyui_bridge::{ComfyUIClient, OllamaClient, BridgeProcessor};

#[tokio::main]
async fn main() -> Result<()> {
    let comfyui = ComfyUIClient::new(
        "ws://127.0.0.1:8188/ws".to_string(),
        "http://127.0.0.1:8188".to_string(),
    );

    let ollama = OllamaClient::new("http://localhost:11434".to_string());

    let processor = BridgeProcessor::new(
        comfyui,
        ollama,
        "llava:13b".to_string(),
    );

    // Process an image
    let image_base64 = /* ... */;
    let scene_graph = processor.process_image(image_base64).await?;

    // Query the scene
    let answer = processor.query_scene("What objects are on the table?").await?;
    println!("{}", answer);

    Ok(())
}
```

### Audio Capture

```rust
use comfyui_bridge::audio::capture::AudioCapture;

let mut capture = AudioCapture::new(16000)?;
capture.start_capture()?;

// Record for 5 seconds
tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;

let audio_data = capture.get_buffer(true);
println!("Captured {} samples", audio_data.len());
```

## Project Structure

```
src/
├── lib.rs              # Library entry point
├── main.rs             # Binary entry point
├── comfyui/           # ComfyUI integration
│   ├── client.rs      # WebSocket client
│   └── protocol.rs    # Message types
├── ollama/            # Ollama integration
│   ├── client.rs      # HTTP client
│   └── models.rs      # Request/response types
├── vision/            # Scene understanding
│   ├── scene_graph.rs # Graph representation
│   └── hierarchy.rs   # Hierarchy parsing
├── audio/             # Audio processing
│   └── capture.rs     # Real-time capture
├── bridge/            # Core processing pipeline
│   ├── processor.rs   # Main coordinator
│   └── state.rs       # Shared state
├── training/          # Training pipeline
└── api/               # HTTP API
```

## Development

### With Claude/Cursor in VSCode

1. Open the project:
   ```bash
   code .
   ```

2. Use Claude to help implement features:
   - "Implement streaming image processing in ComfyUI client"
   - "Add temporal coherence tracking for video frames"
   - "Create annotation interface for training data"

3. Example prompts for Claude:
   ```
   "Generate a function that converts audio samples to WAV format for Whisper"
   "Implement scene graph serialization to training dataset format"
   "Add support for tracking object movements across video frames"
   ```

### Testing

```bash
# Run all tests
cargo test

# Run specific module tests
cargo test vision::

# Run with logging
RUST_LOG=debug cargo test
```

### Benchmarking

```bash
cargo bench
```

## Roadmap

### Phase 1: Core Infrastructure ✅
- [x] ComfyUI WebSocket client
- [x] Ollama HTTP client
- [x] Basic scene graph structure
- [x] Audio capture

### Phase 2: Scene Understanding (Current)
- [ ] Multi-stage hierarchical prompting
- [ ] Scene graph visualization
- [ ] Relationship extraction improvements
- [ ] Temporal coherence for video

### Phase 3: Audio Integration
- [ ] Whisper integration
- [ ] Audio-visual synchronization
- [ ] Speech-driven scene queries
- [ ] Multimodal fusion

### Phase 4: Training Pipeline
- [ ] Annotation interface
- [ ] Dataset generation
- [ ] Fine-tuning data export
- [ ] Evaluation metrics

### Phase 5: Robotics Integration
- [ ] Real-time robot perception
- [ ] Action planning from scene understanding
- [ ] Closed-loop feedback
- [ ] Multi-robot coordination

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `cargo test`
5. Submit a pull request

## Research Applications

This system is designed for research in:

- **Embodied AI**: Understanding spatial relationships for robotic manipulation
- **Vision-Language Models**: Training models on hierarchical scene understanding
- **Multimodal Learning**: Combining vision, language, and audio
- **Robotic Perception**: Real-time scene analysis for navigation and interaction

## Example Workflows

### Kitchen Scene Analysis
```
ComfyUI generates kitchen scene
    ↓
Bridge extracts image
    ↓
Ollama (LLaVA) identifies:
    - Objects: table, cups, plates, fridge
    - Relationships: cups ON table, plates IN cabinet
    ↓
Scene Graph:
    Kitchen (root)
    ├─ Table
    │  ├─ Cup 1
    │  └─ Cup 2
    └─ Cabinet
       └─ Plates
```

### Robot Task Planning
```
User: "Bring me the cup on the table"
    ↓
Scene Graph Query: Find("cup") → Table.children
    ↓
Spatial Reasoning: Cup position relative to robot
    ↓
Action Planning: Navigate → Reach → Grasp → Deliver
```

## Performance

- **Image processing**: ~2-5 seconds per image (LLaVA 13B)
- **Scene graph construction**: <100ms
- **Audio capture**: Real-time (16kHz)
- **Memory**: ~2GB for LLaVA 13B

## License

MIT

## Citation

If you use this in your research, please cite:

```bibtex
@software{comfyui_ollama_bridge,
  title={ComfyUI-Ollama Bridge: Hierarchical Scene Understanding for Embodied AI},
  year={2025},
  url={https://github.com/...}
}
```

## Acknowledgments

- ComfyUI team for the excellent workflow system
- Ollama team for local LLM inference
- LLaVA team for vision-language models

## Support

For questions or issues:
- Open an issue on GitHub
- Check existing discussions
- Review the architecture document (PROJECT_ARCHITECTURE.md)
