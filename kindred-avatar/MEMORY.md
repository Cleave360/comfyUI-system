# Kindred Memory System

## Overview

Kindred uses **ChromaDB** as a vector database to maintain semantic memory across conversations. This allows Kindred to:
- Remember past conversations and context
- Retrieve relevant information when needed
- Build long-term understanding of user preferences
- Provide contextually aware responses

## Architecture

### Storage
- **Database**: ChromaDB (persistent, disk-based)
- **Location**: `backend/memory/` directory
- **Collection**: `kindred_memory`
- **Embeddings**: Sentence Transformers (default model)

### Data Structure
Each memory entry contains:
- **Document**: Conversation exchange ("User: ... \nKindred: ...")
- **Metadata**: Timestamp, client ID
- **ID**: Unique identifier based on timestamp
- **Embedding**: Auto-generated vector for semantic search

## Features

### Automatic Storage
Every conversation exchange is automatically stored:
```python
User: "Tell me about quantum computing"
Kindred: "Quantum computing uses quantum bits..."
```
Stored as single document with metadata.

### Semantic Retrieval
When processing new input, Kindred retrieves the 3 most relevant memories:
```python
Query: "What did we discuss earlier?"
Retrieved: Top 3 semantically similar past conversations
```

### Context Injection
Retrieved memories are injected into system prompt:
```
Relevant context from past conversations:
- User: "What's your favorite color?" Kindred: "I appreciate deep blues..."
- User: "Do you like art?" Kindred: "I find abstract art fascinating..."
```

## Usage

### Installation
ChromaDB is automatically installed with:
```bash
pip install -r requirements.txt
```

### Configuration
Memory system initializes automatically in `server_voice.py`:
```python
self.memory = client.get_or_create_collection(
    name="kindred_memory",
    metadata={"description": "Kindred's conversation and context memory"}
)
```

### Fallback Behavior
If ChromaDB is not installed or fails to initialize:
- Server continues to work normally
- Memory features are disabled
- Warning message is logged
- No impact on other functionality

## Management

### Viewing Memories
Access ChromaDB directly:
```python
import chromadb

client = chromadb.PersistentClient(path="backend/memory")
collection = client.get_collection("kindred_memory")

# Get all memories
memories = collection.get()
print(f"Total memories: {len(memories['ids'])}")

# Get specific memory
result = collection.get(ids=["msg_12345_1704672000000"])
```

### Searching Memories
```python
# Semantic search
results = collection.query(
    query_texts=["Tell me about AI"],
    n_results=5
)

# Filter by metadata
results = collection.get(
    where={"client_id": "12345"}
)
```

### Clearing Memories
```python
# Delete specific memory
collection.delete(ids=["msg_12345_1704672000000"])

# Delete all memories (reset)
client.delete_collection("kindred_memory")
collection = client.create_collection("kindred_memory")
```

### Exporting Memories
```python
# Export to JSON
import json

memories = collection.get()
export_data = {
    "ids": memories['ids'],
    "documents": memories['documents'],
    "metadatas": memories['metadatas']
}

with open("kindred_memories_backup.json", "w") as f:
    json.dump(export_data, f, indent=2)
```

### Importing Memories
```python
# Import from JSON
import json

with open("kindred_memories_backup.json", "r") as f:
    data = json.load(f)

collection.add(
    ids=data['ids'],
    documents=data['documents'],
    metadatas=data['metadatas']
)
```

## Advanced Features

### Custom Embedding Models
Change the default embedding model:
```python
from chromadb.utils import embedding_functions

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"  # Faster, smaller model
)

collection = client.get_or_create_collection(
    name="kindred_memory",
    embedding_function=sentence_transformer_ef
)
```

### Retrieval Tuning
Adjust number of retrieved memories in `server_voice.py`:
```python
results = self.memory.query(
    query_texts=[user_message],
    n_results=5  # Increase for more context (default: 3)
)
```

### Metadata Filtering
Add more metadata for filtering:
```python
self.memory.add(
    documents=[conversation],
    metadatas=[{
        "timestamp": time.time(),
        "client_id": str(client_id),
        "topic": "image_generation",  # Custom field
        "sentiment": "positive"        # Custom field
    }],
    ids=[memory_id]
)
```

## Performance

### Storage Efficiency
- **Average memory size**: ~200-500 bytes per conversation
- **1000 conversations**: ~500KB
- **10,000 conversations**: ~5MB
- **Embeddings**: ~1KB per document (384-dimensional vectors)

### Query Performance
- **Retrieval time**: 1-10ms for typical collections (<10k documents)
- **Embedding generation**: 50-200ms per query
- **Total overhead**: ~100-300ms per request

### Optimization Tips
1. **Limit collection size**: Keep most recent 10k memories
2. **Use faster embedding model**: all-MiniLM-L6-v2
3. **Batch operations**: Add memories in batches
4. **Index tuning**: Adjust HNSW parameters

## Troubleshooting

### Memory not persisting
Check database path exists:
```bash
ls -la backend/memory/
```

### Embedding errors
Ensure sentence-transformers is installed:
```bash
pip install sentence-transformers
```

### High memory usage
Clear old memories:
```python
# Keep only recent 5000 memories
all_memories = collection.get()
if len(all_memories['ids']) > 5000:
    old_ids = sorted(all_memories['ids'])[:len(all_memories['ids'])-5000]
    collection.delete(ids=old_ids)
```

### Permission errors
Fix directory permissions:
```bash
chmod -R 755 backend/memory/
```

## Database Schema

### ChromaDB Collection
```
kindred_memory/
├── chroma.sqlite3          # Metadata database
├── index/                  # HNSW index files
└── parquet files          # Document embeddings
```

### Document Format
```json
{
  "id": "msg_12345_1704672000000",
  "document": "User: How are you?\nKindred: I'm doing well!",
  "metadata": {
    "timestamp": 1704672000.0,
    "client_id": "12345"
  },
  "embedding": [0.123, -0.456, ...]  # 384-dim vector
}
```

## Future Enhancements

- [ ] Automatic memory summarization (compress old memories)
- [ ] Topic clustering (group related memories)
- [ ] Importance scoring (prioritize key memories)
- [ ] Multi-user isolation (separate collections per user)
- [ ] Memory pruning strategies (remove redundant/outdated info)
- [ ] Export/import UI
- [ ] Memory search API endpoint
- [ ] Integration with external knowledge bases

## Alternative Options

If ChromaDB doesn't meet your needs:

### SQLite (Structured)
```python
import sqlite3

conn = sqlite3.connect('kindred_memory.db')
c = conn.cursor()
c.execute('''CREATE TABLE conversations
             (id TEXT PRIMARY KEY, user_msg TEXT, kindred_msg TEXT,
              timestamp REAL, client_id TEXT)''')
```

### Redis (Fast, Ephemeral)
```python
import redis

r = redis.Redis(host='localhost', port=6379, db=0)
r.set(memory_id, json.dumps(conversation))
```

### PostgreSQL + pgvector (Production)
```python
from pgvector.sqlalchemy import Vector

# Requires PostgreSQL with pgvector extension
# More complex setup but more powerful
```

## References

- **ChromaDB Docs**: https://docs.trychroma.com/
- **Sentence Transformers**: https://www.sbert.net/
- **Vector Databases**: https://www.pinecone.io/learn/vector-database/
