# Getting Started with Claude/Cursor Development

This guide helps you use Claude (via Cursor or VSCode with Copilot) to develop the ComfyUI-Ollama Bridge.

## Setup

### 1. Install Dependencies

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install Cursor or VSCode
# Cursor: https://cursor.sh
# VSCode: https://code.visualstudio.com

# Open the project
cursor .  # or: code .
```

### 2. Build Initial Project

```bash
# Check that it compiles
cargo check

# Build the project
cargo build

# Run tests
cargo test
```

### 3. Start Required Services

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Pull models
ollama pull llava:13b
ollama pull whisper:large

# Terminal 3: Start ComfyUI (if you have it installed)
cd /path/to/ComfyUI
python main.py
```

## Development Workflow with Claude

### Phase 1: Implement ComfyUI Image Processing

#### Prompt for Claude:

```
I need to enhance the ComfyUI client to properly handle image outputs from workflows.

Current code in src/comfyui/client.rs has a TODO in the handle_execution method.

Please:
1. Properly extract image data from ComfyUI execution events
2. Fetch the actual image bytes from the ComfyUI API
3. Convert to base64 for sending to Ollama
4. Add error handling for missing images
5. Add logging for debugging

The execution event format is:
{
  "type": "executed",
  "data": {
    "node": "node_id",
    "output": {
      "images": [
        {"filename": "...", "type": "output", "subfolder": ""}
      ]
    }
  }
}
```

### Phase 2: Improve Scene Graph Parsing

#### Prompt for Claude:

```
The HierarchyParser in src/vision/hierarchy.rs needs to handle more robust parsing of LLM responses.

Current issues:
1. LLMs don't always return perfect JSON
2. Object names might not match exactly between responses
3. Need fuzzy matching for object labels

Please improve the parser to:
1. Handle JSON with extra text before/after
2. Use fuzzy string matching for object names (use strsim crate)
3. Add confidence scores for matches
4. Handle missing or malformed fields gracefully
5. Add detailed error messages

Add the strsim crate to Cargo.toml if needed.
```

### Phase 3: Add Streaming Response Handling

#### Prompt for Claude:

```
Implement streaming response handling for Ollama in src/ollama/client.rs.

The generate_stream method has a TODO. Ollama returns NDJSON (newline-delimited JSON).

Requirements:
1. Parse NDJSON stream correctly
2. Handle partial JSON objects at stream boundaries
3. Yield text chunks as they arrive
4. Handle stream errors gracefully
5. Add example usage in comments

Example Ollama streaming response:
{"model":"llava","response":"The ","done":false}
{"model":"llava","response":"image ","done":false}
{"model":"llava","response":"shows","done":true}
```

### Phase 4: Implement Audio-Visual Sync

#### Prompt for Claude:

```
Create a new module src/audio/sync.rs that synchronizes audio transcriptions with visual events.

Requirements:
1. Track timestamps for both audio and video frames
2. Align speech segments with scene changes
3. Support queries like "what was happening when the person said X?"
4. Use crossbeam-channel for event passing between audio and vision threads

Design the API first, then implement it.
Include usage examples and tests.
```

### Phase 5: Build Training Data Export

#### Prompt for Claude:

```
Implement the training module to export scene graphs as fine-tuning data.

In src/training/mod.rs, create:
1. A struct to represent training examples
2. Methods to convert scene graphs to different formats:
   - JSONL for Alpaca-style fine-tuning
   - Conversational format for chat models
   - Vision-language instruction format
3. Data augmentation (rotate, flip, crop scenes)
4. Train/val/test splitting
5. Statistics and validation

Format should be compatible with axolotl or similar fine-tuning frameworks.
```

## Prompting Best Practices

### ✅ Good Prompts

```
"Generate a function that validates scene graph consistency -
check for cycles, orphaned nodes, and contradictory relationships"

"Add integration tests for the full pipeline:
ComfyUI image → Ollama analysis → scene graph → JSON export"

"Implement caching for Ollama responses to avoid redundant API calls
during development. Use sled for the cache."
```

### ❌ Avoid

```
"Make it better"  // Too vague

"Add all the features"  // Too broad

"Fix the bug"  // Need to specify which bug
```

### Iterative Development

1. **Start Simple**: "Implement basic X with hardcoded values"
2. **Add Flexibility**: "Now make X configurable"
3. **Add Robustness**: "Add error handling and logging to X"
4. **Optimize**: "Profile X and optimize the bottleneck"
5. **Test**: "Write comprehensive tests for X"

## Debugging with Claude

### When You Get Errors

```
I'm getting this error:
[paste error]

In this code:
[paste relevant code]

The error occurs when:
[describe scenario]

Please help me:
1. Understand what's causing it
2. Fix it
3. Add checks to prevent it in the future
```

### When Performance is Slow

```
This function is slow:
[paste function]

Profiling shows it takes X ms and is called Y times per second.

Please help optimize it:
1. Identify bottlenecks
2. Suggest algorithmic improvements
3. Consider parallel processing if applicable
4. Maintain correctness
```

## Example Development Session

### Goal: Add Video Support

#### Step 1: Design

```
@claude I want to add video processing support to the bridge.

Video will come as a sequence of frames from ComfyUI.
Each frame needs to be analyzed and linked temporally.

Please design:
1. A VideoFrame struct with timestamp, frame data, and metadata
2. A TemporalSceneGraph that tracks objects across frames
3. Methods to detect:
   - New objects appearing
   - Objects leaving the scene
   - Object movement
   - Relationship changes over time

Just design the API and structs, don't implement yet.
```

#### Step 2: Implement Core

```
@claude Now implement the VideoFrame and TemporalSceneGraph
you designed in src/vision/temporal.rs

Focus on:
1. Efficient frame buffering (ring buffer)
2. Object tracking by ID across frames
3. Movement vector calculation
4. Memory efficiency (don't keep all frames)

Add TODO comments for features we'll add later.
```

#### Step 3: Test

```
@claude Write unit tests for the temporal scene graph.

Test cases:
1. Single frame → should work like normal scene graph
2. Object appearing then disappearing
3. Object moving from left to right
4. Two objects swapping positions
5. Rapid scene changes

Use mock data, don't need real images.
```

#### Step 4: Integrate

```
@claude Integrate temporal scene graph into BridgeProcessor.

Add a new method process_video that:
1. Takes a stream of frames
2. Builds temporal scene graph
3. Returns summary of what happened in the video

Update the API in src/bridge/processor.rs
```

## Tips for Working with Claude

1. **Be Specific**: Include file paths, function names, exact requirements
2. **Provide Context**: Show relevant code, error messages, or designs
3. **Ask for Tests**: Always request tests with implementations
4. **Iterate**: Start simple, then refine
5. **Review Code**: Claude makes mistakes - review and test everything
6. **Ask Why**: If something seems odd, ask Claude to explain the approach

## Common Tasks

### Add a New Crate

```
@claude I need to add the "uuid" crate for generating unique IDs.

Please:
1. Add it to Cargo.toml with the appropriate features
2. Show me how to use it in the SceneObject struct
3. Update the constructor to auto-generate IDs
```

### Refactor a Module

```
@claude The ollama/client.rs file is getting too large.

Please refactor it:
1. Split into multiple files by responsibility
2. Create a mod.rs that re-exports the public API
3. Keep the existing API unchanged
4. Add module-level documentation
```

### Add Documentation

```
@claude Add comprehensive documentation to src/vision/scene_graph.rs

Include:
1. Module-level docs explaining the scene graph concept
2. Examples showing how to build and query graphs
3. Docs for all public types and methods
4. Links to relevant research papers or concepts
```

## Troubleshooting

### Rust Compilation Errors

Most common issues:
- Missing use statements → Claude can add them
- Lifetime issues → Claude can help design around them
- Type mismatches → Claude can add conversions

### Ollama Connection Issues

```
@claude I'm getting "Connection refused" when connecting to Ollama.

Please add:
1. Better error messages that check if Ollama is running
2. Retry logic with exponential backoff
3. Health check on startup
4. Configuration validation
```

### ComfyUI WebSocket Dropping

```
@claude The ComfyUI WebSocket keeps disconnecting.

Please add:
1. Automatic reconnection logic
2. Exponential backoff
3. Event queue to buffer messages during reconnection
4. Connection state management
```

## Next Steps

Once you have the basics working:

1. **UI**: "Help me set up a Tauri frontend"
2. **Deployment**: "Create a Dockerfile for the bridge"
3. **Monitoring**: "Add Prometheus metrics"
4. **Testing**: "Set up CI with GitHub Actions"

## Resources

- Rust Book: https://doc.rust-lang.org/book/
- Tokio Docs: https://tokio.rs
- ComfyUI API: Check your ComfyUI `/docs` endpoint
- Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md
