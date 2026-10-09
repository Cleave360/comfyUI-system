# ComfyUI-Ollama Bridge for Embodied AI
## Project Architecture & Implementation Guide

### Project Goals
- Teach LLMs/VLMs to understand visual hierarchies and contextual relationships
- Bridge ComfyUI workflows with Ollama models
- Add audio processing for multimodal understanding
- Foundation for embodied AI in robotics

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (Tauri + React)                 │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │  Workflow  │  │   Scene      │  │   Audio          │    │
│  │  Viewer    │  │   Hierarchy  │  │   Visualizer     │    │
│  └────────────┘  └──────────────┘  └──────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
                     Tauri IPC Bridge
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Rust Core Engine                          │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          ComfyUI Integration Layer                    │  │
│  │  - WebSocket client for ComfyUI API                   │  │
│  │  - Workflow parser & executor                         │  │
│  │  - Image/Video stream handler                         │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │       Vision-Language Processing Core                 │  │
│  │  - Scene graph builder                                │  │
│  │  - Hierarchical relationship parser                   │  │
│  │  - Spatial reasoning engine                           │  │
│  │  - Temporal coherence tracker (video)                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Ollama Integration Layer                   │  │
│  │  - Model management & selection                       │  │
│  │  - Streaming inference client                         │  │
│  │  - Vision model adapter (LLaVA, etc.)                 │  │
│  │  - Prompt engineering pipeline                        │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          Audio Processing Layer                       │  │
│  │  - Real-time audio capture (cpal)                     │  │
│  │  - Speech-to-text integration (Whisper via Ollama)   │  │
│  │  - Audio feature extraction                           │  │
│  │  - Audio-visual synchronization                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          Training & Feedback System                   │  │
│  │  - Annotation storage (SQLite/PostgreSQL)             │  │
│  │  - Fine-tuning data generator                         │  │
│  │  - Model evaluation metrics                           │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### Rust Crates
```toml
[dependencies]
# Core async runtime
tokio = { version = "1.35", features = ["full"] }
tokio-tungstenite = "0.21"  # WebSocket client

# Web framework for API
axum = "0.7"
tower-http = { version = "0.5", features = ["cors", "fs"] }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Image processing
image = "0.24"
imageproc = "0.23"

# Audio processing
cpal = "0.15"  # Cross-platform audio I/O
hound = "3.5"  # WAV encoding/decoding
rubato = "0.14"  # Resampling

# HTTP client for Ollama
reqwest = { version = "0.11", features = ["json", "stream"] }

# Database
sqlx = { version = "0.7", features = ["runtime-tokio-native-tls", "postgres", "sqlite"] }

# UI (Tauri)
tauri = { version = "1.5", features = ["api-all"] }

# Graph structures for scene representation
petgraph = "0.6"

# Error handling
anyhow = "1.0"
thiserror = "1.0"

# Logging
tracing = "0.1"
tracing-subscriber = "0.3"

# Configuration
config = "0.13"
```

---

## Project Structure

```
comfyui-bridge/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── main.rs
│   │
│   ├── comfyui/
│   │   ├── mod.rs
│   │   ├── client.rs          # WebSocket client
│   │   ├── workflow.rs        # Workflow parser/executor
│   │   ├── protocol.rs        # ComfyUI protocol messages
│   │   └── stream.rs          # Image/video streaming
│   │
│   ├── ollama/
│   │   ├── mod.rs
│   │   ├── client.rs          # HTTP client for Ollama API
│   │   ├── models.rs          # Model management
│   │   ├── vision.rs          # Vision model integration
│   │   └── streaming.rs       # Streaming response handler
│   │
│   ├── vision/
│   │   ├── mod.rs
│   │   ├── scene_graph.rs     # Scene graph representation
│   │   ├── hierarchy.rs       # Hierarchical relationship detection
│   │   ├── spatial.rs         # Spatial reasoning
│   │   ├── temporal.rs        # Video temporal coherence
│   │   └── features.rs        # Visual feature extraction
│   │
│   ├── audio/
│   │   ├── mod.rs
│   │   ├── capture.rs         # Real-time audio capture
│   │   ├── whisper.rs         # Whisper integration
│   │   ├── features.rs        # Audio feature extraction
│   │   └── sync.rs            # Audio-visual sync
│   │
│   ├── bridge/
│   │   ├── mod.rs
│   │   ├── processor.rs       # Main processing pipeline
│   │   ├── state.rs           # Shared state management
│   │   └── events.rs          # Event bus
│   │
│   ├── training/
│   │   ├── mod.rs
│   │   ├── annotation.rs      # Annotation storage/retrieval
│   │   ├── dataset.rs         # Dataset generation
│   │   └── metrics.rs         # Evaluation metrics
│   │
│   ├── api/
│   │   ├── mod.rs
│   │   ├── routes.rs          # HTTP API routes
│   │   └── handlers.rs        # Request handlers
│   │
│   └── ui/
│       └── tauri_commands.rs  # Tauri IPC commands
│
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── WorkflowViewer.tsx
│   │   │   ├── SceneHierarchy.tsx
│   │   │   ├── AudioVisualizer.tsx
│   │   │   └── ModelControl.tsx
│   │   └── hooks/
│   │       ├── useComfyUI.ts
│   │       └── useOllama.ts
│   └── tauri.conf.json
│
├── config/
│   └── default.toml           # Default configuration
│
└── README.md
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)
**Goal**: Get basic ComfyUI ↔ Ollama communication working

#### Tasks:
1. **ComfyUI WebSocket Client**
   - Connect to ComfyUI WebSocket API
   - Subscribe to workflow execution events
   - Capture output images/frames

2. **Ollama HTTP Client**
   - Basic chat completion API
   - Vision model support (LLaVA)
   - Streaming response handling

3. **Basic Bridge**
   - Forward ComfyUI outputs to Ollama
   - Simple prompt: "Describe this image"
   - Display results in CLI

**Code Starting Point** (`src/comfyui/client.rs`):
```rust
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::{StreamExt, SinkExt};
use serde::{Deserialize, Serialize};
use anyhow::Result;

#[derive(Debug, Serialize, Deserialize)]
pub struct ComfyUIMessage {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub data: serde_json::Value,
}

pub struct ComfyUIClient {
    url: String,
}

impl ComfyUIClient {
    pub fn new(url: String) -> Self {
        Self { url }
    }

    pub async fn connect(&self) -> Result<()> {
        let (ws_stream, _) = connect_async(&self.url).await?;
        let (mut write, mut read) = ws_stream.split();

        // Subscribe to execution events
        let subscribe_msg = serde_json::json!({
            "type": "subscribe",
            "data": {
                "events": ["executed", "execution_error"]
            }
        });

        write.send(Message::Text(subscribe_msg.to_string())).await?;

        while let Some(msg) = read.next().await {
            let msg = msg?;
            if let Message::Text(text) = msg {
                let comfy_msg: ComfyUIMessage = serde_json::from_str(&text)?;
                self.handle_message(comfy_msg).await?;
            }
        }

        Ok(())
    }

    async fn handle_message(&self, msg: ComfyUIMessage) -> Result<()> {
        match msg.msg_type.as_str() {
            "executed" => {
                tracing::info!("Node executed: {:?}", msg.data);
                // Extract output images here
            }
            "execution_error" => {
                tracing::error!("Execution error: {:?}", msg.data);
            }
            _ => {}
        }
        Ok(())
    }
}
```

**Code Starting Point** (`src/ollama/client.rs`):
```rust
use reqwest::Client;
use serde::{Deserialize, Serialize};
use anyhow::Result;
use futures_util::StreamExt;

#[derive(Debug, Serialize)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>, // Base64 encoded
}

#[derive(Debug, Deserialize)]
pub struct GenerateResponse {
    pub model: String,
    pub response: String,
    pub done: bool,
}

pub struct OllamaClient {
    client: Client,
    base_url: String,
}

impl OllamaClient {
    pub fn new(base_url: String) -> Self {
        Self {
            client: Client::new(),
            base_url,
        }
    }

    pub async fn generate_vision(
        &self,
        model: &str,
        prompt: &str,
        image_base64: String,
    ) -> Result<String> {
        let request = GenerateRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            stream: false,
            images: Some(vec![image_base64]),
        };

        let response = self.client
            .post(format!("{}/api/generate", self.base_url))
            .json(&request)
            .send()
            .await?
            .json::<GenerateResponse>()
            .await?;

        Ok(response.response)
    }

    pub async fn generate_stream(
        &self,
        model: &str,
        prompt: &str,
    ) -> Result<impl StreamExt<Item = Result<String>>> {
        let request = GenerateRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            stream: true,
            images: None,
        };

        let response = self.client
            .post(format!("{}/api/generate", self.base_url))
            .json(&request)
            .send()
            .await?;

        let stream = response.bytes_stream();
        // Parse streaming JSON responses

        todo!("Implement streaming parser")
    }
}
```

---

### Phase 2: Scene Understanding (Week 3-4)
**Goal**: Build hierarchical scene representation

#### Tasks:
1. **Scene Graph Structure**
   - Define graph nodes (objects, regions, attributes)
   - Define edges (spatial, semantic relationships)
   - Implement graph traversal algorithms

2. **Vision-Language Prompting**
   - Design prompts to extract hierarchies
   - "What objects do you see? What contains what?"
   - Parse LLM responses into structured data

3. **Visualization**
   - Display scene graph in UI
   - Interactive exploration
   - Highlight relationships

**Code Starting Point** (`src/vision/scene_graph.rs`):
```rust
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneObject {
    pub id: String,
    pub label: String,
    pub confidence: f32,
    pub bbox: Option<BoundingBox>,
    pub attributes: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Relationship {
    Contains,       // A contains B
    PartOf,         // A is part of B
    LeftOf,         // Spatial
    RightOf,
    Above,
    Below,
    Near,
    Interacting,    // A and B are interacting
    Supports,       // A supports B (physical)
}

pub struct SceneGraph {
    graph: DiGraph<SceneObject, Relationship>,
    index_map: HashMap<String, NodeIndex>,
}

impl SceneGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            index_map: HashMap::new(),
        }
    }

    pub fn add_object(&mut self, obj: SceneObject) -> NodeIndex {
        let idx = self.graph.add_node(obj.clone());
        self.index_map.insert(obj.id.clone(), idx);
        idx
    }

    pub fn add_relationship(
        &mut self,
        from_id: &str,
        to_id: &str,
        rel: Relationship,
    ) -> Option<()> {
        let from_idx = self.index_map.get(from_id)?;
        let to_idx = self.index_map.get(to_id)?;
        self.graph.add_edge(*from_idx, *to_idx, rel);
        Some(())
    }

    pub fn get_hierarchy(&self, root_id: &str) -> Option<Vec<String>> {
        // BFS traversal to get hierarchy
        let root_idx = self.index_map.get(root_id)?;
        let mut result = Vec::new();

        // Implement BFS here

        Some(result)
    }
}
```

---

### Phase 3: Audio Integration (Week 5-6)
**Goal**: Add audio processing and speech understanding

#### Tasks:
1. **Audio Capture**
   - Real-time microphone input
   - Buffer management
   - Format conversion

2. **Whisper Integration**
   - Call Whisper via Ollama
   - Transcription streaming
   - Timestamp alignment

3. **Multimodal Fusion**
   - Sync audio with visual events
   - "What is happening when the person says X?"
   - Audio cues for scene understanding

**Code Starting Point** (`src/audio/capture.rs`):
```rust
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Stream, StreamConfig,
};
use std::sync::{Arc, Mutex};
use anyhow::Result;

pub struct AudioCapture {
    stream: Option<Stream>,
    buffer: Arc<Mutex<Vec<f32>>>,
}

impl AudioCapture {
    pub fn new() -> Result<Self> {
        Ok(Self {
            stream: None,
            buffer: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub fn start_capture(&mut self) -> Result<()> {
        let host = cpal::default_host();
        let device = host.default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No input device"))?;

        let config = device.default_input_config()?;
        let buffer = Arc::clone(&self.buffer);

        let stream = device.build_input_stream(
            &config.into(),
            move |data: &[f32], _: &_| {
                let mut buf = buffer.lock().unwrap();
                buf.extend_from_slice(data);
            },
            |err| eprintln!("Audio error: {}", err),
            None,
        )?;

        stream.play()?;
        self.stream = Some(stream);

        Ok(())
    }

    pub fn get_buffer(&self, clear: bool) -> Vec<f32> {
        let mut buffer = self.buffer.lock().unwrap();
        let data = buffer.clone();
        if clear {
            buffer.clear();
        }
        data
    }
}
```

---

### Phase 4: Training Pipeline (Week 7-8)
**Goal**: Enable model fine-tuning and evaluation

#### Tasks:
1. **Annotation Interface**
   - Label hierarchies manually
   - Correct model mistakes
   - Export training data

2. **Dataset Generation**
   - Format for fine-tuning
   - Train/val/test splits
   - Data augmentation

3. **Evaluation Metrics**
   - Hierarchy accuracy
   - Relationship F1 score
   - Qualitative assessment

---

## Configuration Example

**`config/default.toml`**:
```toml
[comfyui]
websocket_url = "ws://127.0.0.1:8188/ws"
api_url = "http://127.0.0.1:8188"

[ollama]
base_url = "http://localhost:11434"
default_model = "llava:13b"
vision_models = ["llava:13b", "llava:7b", "bakllava"]

[audio]
sample_rate = 16000
channels = 1
whisper_model = "whisper:large"

[server]
host = "127.0.0.1"
port = 3030

[database]
url = "sqlite://comfyui_bridge.db"

[training]
annotation_dir = "./annotations"
dataset_dir = "./datasets"
```

---

## Development Workflow with Claude/Codex

### In VSCode/Cursor:

1. **Open project**: `code comfyui-bridge/`

2. **Use Claude to generate code**:
   - "Implement the ComfyUI WebSocket client in src/comfyui/client.rs"
   - "Add scene graph serialization to JSON"
   - "Create audio buffer management with ring buffer"

3. **Iterate with tests**:
   - Write tests first
   - Ask Claude to implement passing code
   - Refactor with Claude's suggestions

4. **Prompt examples**:
   ```
   "Generate a Rust function that converts ComfyUI image outputs
   to base64 for sending to Ollama vision models"

   "Implement a scene graph traversal that finds all objects
   contained within a parent object"

   "Create a streaming parser for Ollama's NDJSON response format"
   ```

---

## Next Steps

1. **Set up Rust project**: `cargo init --lib comfyui-bridge`
2. **Add dependencies**: Copy Cargo.toml dependencies above
3. **Implement Phase 1**: Get basic ComfyUI ↔ Ollama working
4. **Test with simple workflows**: Single image → LLaVA description
5. **Iterate**: Add complexity incrementally

Would you like me to generate more specific code for any component?
Or help with Tauri setup for the UI?
