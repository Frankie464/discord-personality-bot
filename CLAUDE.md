# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Discord bot that learns authentic personality from Discord server message history. Unlike typical chatbots, this bot captures and replicates the unfiltered communication style of a server community through deep fine-tuning, producing responses that are indistinguishable from real server members.

**Core Philosophy**: Personality = Authenticity. The bot preserves all natural communication patterns including single-word responses, repeated text, emojis, slang, and unfiltered language. No generic "AI assistant" behavior - just authentic Discord conversation.

## Technology Stack (November 2025 State-of-the-Art)

### Language Model
- **Base Model**: Qwen2.5-3B-Instruct (Alibaba, September 2024)
- **Why Qwen2.5-3B**:
  - Best-in-class for creative writing and personality capture (2025 research)
  - 128K context window (vs 32K for competitors)
  - Superior instruction following (77.4 IFEval score)
  - Optimal size for RTX 3070 8GB fine-tuning
  - 2.2GB quantized (Q4_K_M GGUF) for efficient laptop inference
  - Apache 2.0 license (commercial use approved)
  - Outperforms Llama 3.2-3B, Phi-3-Mini, Gemma 2-2B for personality tasks

### Alternative Models Evaluated
Based on November 2025 research:

**Llama 3.2-3B-Instruct**:
- Similar creative writing performance to Qwen2.5
- Meta backing, strong ecosystem
- 8K default context (expandable)
- Slightly higher memory (3.4GB vs 2.2GB quantized)
- Use if prefer Meta ecosystem

**Gemma 3-1B**:
- Ultra-lightweight (529MB base, 0.5GB quantized)
- 2,585 tokens/sec prefill speed
- 32K context only
- Best for extreme resource constraints
- Released March 2025, cutting-edge efficiency

**Phi-4 (14B)**:
- Excellent math/reasoning (80.4 MATH benchmark)
- Too large for RTX 3070 8GB comfortable fine-tuning
- Optimized for structured tasks, not personality

**DeepSeek-R1 distilled models**:
- 1.5B, 7B, 14B variants available
- Strong reasoning capabilities
- Less optimal for creative/personality tasks
- Consider for specialized reasoning needs

**Ministral 3B/8B**:
- Mistral AI's edge models
- Optimized for reasoning, not personality
- Sliding window attention efficient
- Use if need function-calling capabilities

### Fine-Tuning Approach
- **Method**: QLoRA (Quantized Low-Rank Adaptation)
- **Framework**: Unsloth (2x faster than standard implementations)
- **Why QLoRA**:
  - 70-80% VRAM reduction vs full fine-tuning
  - 98% quality retention
  - Enables 7B models on single RTX 3070 8GB
  - 3B models train comfortably in 4-5 hours
  - Industry standard for personality capture in 2025

**QLoRA Configuration**:
```python
lora_r=64  # High rank for personality capacity
lora_alpha=128  # 2x rank typical
target_modules=[  # Fine-tune all attention + FFN + embeddings
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "embed_tokens", "lm_head"
]
```

- **Post-Training**: DPO (Direct Preference Optimization) or ORPO
  - DPO: +20-30% style consistency
  - Uses message reactions as preference signal (reacted = preferred)
  - 2 additional epochs, 1-2 hours GPU time
  - ORPO alternative: combines SFT+DPO in single pass

### Inference Engine
- **Engine**: llama.cpp (via llama-cpp-python)
- **Why llama.cpp**:
  - Best CPU offloading for consumer hardware
  - Industry-standard GGUF format
  - Active development (Flash Attention 3, KV cache optimization)
  - Cross-platform (Windows/Linux/macOS)
  - Mature ecosystem

**Alternatives Considered**:
- **vLLM**: Overkill for single bot, requires multi-GPU
- **ExLlamaV2**: GPU-only, limited format support
- **Ollama**: Wrapper around llama.cpp, less control
- **MLX**: Mac-only optimization

### Vector Database
- **Database**: LanceDB (embedded mode)
- **Why LanceDB**:
  - Zero-copy reads (fastest queries)
  - Disk-based storage (scales beyond RAM)
  - 10MB binary vs ChromaDB's 50MB+
  - Built in Rust (memory-safe, no Python GIL)
  - Native versioning and SQL-like queries
  - Apache Arrow format (ML pipeline optimized)

**Performance (2025 benchmarks)**:
- Insert 10K vectors: 0.8s (vs ChromaDB 1.2s, FAISS 0.5s)
- Query p95 latency: 12ms (vs ChromaDB 18ms, FAISS 8ms)
- Memory footprint: 120MB (vs ChromaDB 280MB, FAISS 450MB)
- Disk storage: 450MB (vs ChromaDB 890MB, FAISS in-memory only)

**Why Not FAISS**: Current bot uses FAISS, but it's fully in-memory and doesn't persist efficiently. For production bot with continuous learning, LanceDB superior.

- **Embeddings**: BAAI/bge-small-en-v1.5 (384-dimensional)
- **RAG Strategy**: Minimal retrieval for context only, not style (personality is in model weights)

### Discord Framework
- **Library**: discord.py 2.4.x
- **Status**: Active development resumed in 2025
- **Why discord.py**:
  - Most mature ecosystem (millions of deployed bots)
  - Full Discord API v10 support
  - Best async performance
  - Largest community and documentation
  - Production-stable

**Alternatives**: py-cord, nextcord, hikari+arc (use if specific features needed)

### Management Interface
- **GUI Framework**: CustomTkinter 5.2+
- **Why CustomTkinter**:
  - Modern Windows 11-style appearance
  - Cross-platform (Windows/Linux/macOS)
  - Lightweight (~5MB)
  - Native performance (no browser)
  - System tray integration
  - Easy development

**Features**:
- One-click start/stop/restart
- Real-time logs and statistics
- Parameter adjustment without code changes
- System tray background operation
- Windows startup auto-launch
- Process monitoring and auto-restart

### State Management
- **Database**: SQLite
- **Why SQLite over JSON**:
  - ACID transactions (no corruption)
  - Efficient queries and indexing
  - Concurrent access safe
  - Bounded growth
  - Easy backup and migration

## Hardware Requirements

### Training Hardware
- **Minimum**: RTX 3070 8GB VRAM
- **Recommended**: RTX 4070 12GB or better
- **Cloud Alternative**: Google Colab (T4 free tier: 4-6 hours)

**RTX 3070 8GB Training Capacity**:
- Qwen2.5-3B: ✅ Comfortable (7.2GB peak VRAM)
- Qwen2.5-7B: ⚠️ Possible but tight (reduce batch size to 1)
- Llama 3.2-3B: ✅ Comfortable
- Phi-4-14B: ❌ Too large

**Training Time (RTX 3070)**:
- QLoRA SFT (5 epochs): 4-5 hours
- DPO (2 epochs): 1-2 hours
- Total: 5-7 hours

### Inference Hardware
- **Deployment**: Consumer laptop (CPU-only acceptable)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Processor**: 8+ core CPU for optimal performance

**Laptop Performance (Q4_K_M quantization)**:
- Model size: 2.2GB
- Memory usage: 3-4GB total (with overhead)
- Inference speed (8-core CPU): 8-12 tokens/sec
- Response time: 2-3 seconds for 20-30 token responses
- Excellent: Responses feel instant in Discord
- Background operation: Minimal impact on laptop usage

## Project Architecture

### Directory Structure

```
discord-personality-bot/
├── README.md                    # Project overview and quick start
├── CLAUDE.md                    # This file: comprehensive guide
├── TODO.md                      # Phase-by-phase implementation checklist
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
├── .gitignore                   # Git ignore patterns
│
├── launcher.py                  # GUI application launcher
├── bot_controller.py            # Bot process management
│
├── gui/                         # Management GUI
│   ├── __init__.py
│   ├── app.py                   # Main CustomTkinter application
│   ├── components/
│   │   ├── main_window.py       # Main control panel
│   │   ├── logs_window.py       # Real-time log viewer
│   │   ├── settings_window.py   # Configuration editor
│   │   └── system_tray.py       # System tray integration
│   └── assets/
│       ├── icon.ico             # Application icon
│       └── logo.png             # Bot logo
│
├── bot/                         # Discord bot core
│   ├── __init__.py
│   ├── main.py                  # Bot entry point, event loop
│   ├── commands.py              # Admin commands (!setrate, !settemp, etc.)
│   ├── handlers.py              # Message event handlers
│   └── config.py                # Bot configuration
│
├── data/                        # Data collection and processing
│   ├── __init__.py
│   ├── fetcher.py               # Discord message scraping (optimized)
│   ├── preprocessor.py          # Minimal data preprocessing
│   └── privacy.py               # Opt-out system
│
├── model/                       # ML model components
│   ├── __init__.py
│   ├── inference.py             # Model loading and generation (persistent)
│   ├── trainer.py               # QLoRA + DPO training scripts
│   └── prompts.py               # System prompts and templates
│
├── storage/                     # Data persistence
│   ├── __init__.py
│   ├── database.py              # SQLite state management
│   └── vectordb.py              # LanceDB vector storage
│
├── scripts/                     # Standalone utility scripts
│   ├── 1_fetch_all_history.py   # Step 1: Scrape Discord messages
│   ├── 2_prepare_training_data.py  # Step 2: Format for training
│   ├── 3_train_model.py         # Step 3: Fine-tune with QLoRA+DPO
│   ├── 4_evaluate_personality.py   # Step 4: Test personality match
│   └── 5_deploy_bot.py          # Step 5: Deploy to laptop
│
├── tests/                       # Testing suite
│   ├── test_personality.py      # Personality match scoring
│   └── test_performance.py      # Speed and resource benchmarks
│
├── data_storage/                # Runtime data (gitignored)
│   ├── messages/                # Scraped Discord messages (JSON)
│   ├── database/                # SQLite databases
│   └── embeddings/              # LanceDB vector indexes
│
└── models/                      # Model files (gitignored)
    ├── base/                    # Original Qwen2.5-3B Q4_K_M
    └── finetuned/               # Fine-tuned personality models
```

### Key Architectural Patterns

**1. Persistent Model Loading (CRITICAL for speed)**:
```python
# WRONG (current bot): Load model on every request
def generate_response(message):
    model = load_model()  # 20+ seconds!
    response = model.generate(message)
    return response

# CORRECT (new bot): Load once at startup
class Bot:
    def __init__(self):
        self.model = load_model()  # Once: 15-20 seconds

    def generate_response(self, message):
        response = self.model.generate(message)  # 2-3 seconds
        return response
```

**2. True Async Execution**:
```python
# Run inference in thread pool, never block Discord event loop
async def on_message(message):
    response = await asyncio.to_thread(self.model.generate, message)
    await message.channel.send(response)
```

**3. Minimal RAG (Context, Not Style)**:
```python
# Retrieve 3-5 relevant messages for context only
# Don't use for style injection - model has personality from fine-tuning
context = vectordb.search(message, limit=5)
prompt = f"{system_prompt}\n\nContext: {context}\n\nUser: {message}"
response = model.generate(prompt)
```

**4. Zero Post-Processing (Trust the Model)**:
```python
# No style enforcement, no uniqueness checks, no heavy filtering
# Only remove system artifacts and limit to Discord's 2000 char max
response = model.generate(prompt)
response = response[:2000]  # Discord limit
return response
```

## Data Collection Strategy

### Message Fetching Principles

**Quantity**:
- Fetch ENTIRE available history (no artificial limits)
- Discord API allows ~15-20 months back typically
- Target: 20,000 - 100,000+ messages
- More data = better personality capture

**Quality Filters (MINIMAL)**:
```python
REMOVE:
- Bot messages (from ANY bots)
- Opted-out users
- System notifications
- Pure link spam (no accompanying text)

KEEP EVERYTHING ELSE:
- ✅ Single-word responses ("lol", "bruh", "fr", "gg")
- ✅ Repeated/spammed text ("GGGGGG", "nooooo")
- ✅ Emojis, reactions, Unicode
- ✅ Typos and non-standard spelling
- ✅ Slang and community jargon
- ✅ Any length: 1 character to 2000 characters
- ✅ Conversational threads and standalone messages
- ✅ All caps, mixed case, lowercase
- ✅ Punctuation patterns (multiple !!!, ???, etc.)
```

**Server Blend Strategy**:
```python
# Personality represents entire server community
# Sample from all active users, weighted by activity
# Preserve communication diversity

Sampling Strategy:
- Include all users (except admin-excluded if any)
- Weight by message count (active users = more influence)
- Ensure diversity (avoid single-user dominance)
- Preserve temporal patterns (activity time, day of week)
- Maintain conversation threading (context important)
```

### Data Preprocessing (MINIMAL)

```python
Preserve Authenticity:
- Keep original punctuation and capitalization
- Keep emoji placement and Unicode
- Keep message length variations
- Keep typos (part of personality!)
- Keep conversation threading

Format for Training:
- Multi-turn conversations (5-10 message exchanges)
- Single message/response pairs
- Mix of interaction types
- ChatML format for Qwen2.5

Example Training Format:
<|im_start|>system
You're a regular on this Discord server. Chat naturally.
<|im_end|>
<|im_start|>user
[previous messages for context]
bruh did u see that game last night
<|im_end|>
<|im_start|>assistant
YOOO that was insane
<|im_end|>
```

## Fine-Tuning for Maximum Personality

### Training Philosophy

**Goal**: Model should be indistinguishable from real server members. Not "AI with personality flavor" but "authentic communication replication."

**Approach**: Deep embedding of personality patterns through extended training, minimal filtering, and preference optimization.

### Supervised Fine-Tuning (SFT) Configuration

```python
Training Hyperparameters:
{
    # Model Selection
    "base_model": "Qwen/Qwen2.5-3B-Instruct",
    "framework": "unsloth",

    # Quantization (for training efficiency)
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4",

    # LoRA Configuration
    "lora_r": 64,  # Rank: higher = more capacity for personality
    "lora_alpha": 128,  # Scaling: typically 2x rank
    "lora_dropout": 0.05,  # Regularization
    "target_modules": [
        # Attention layers (standard)
        "q_proj", "k_proj", "v_proj", "o_proj",
        # Feed-forward layers (standard)
        "gate_proj", "up_proj", "down_proj",
        # Embeddings and output (CRITICAL for style)
        "embed_tokens", "lm_head"
    ],
    "bias": "none",
    "task_type": "CAUSAL_LM",

    # Training Parameters
    "num_train_epochs": 5,  # More epochs = deeper personality
    "learning_rate": 1e-4,  # Lower = more careful learning
    "per_device_train_batch_size": 2,  # RTX 3070 8GB constraint
    "gradient_accumulation_steps": 16,  # Effective batch = 32
    "max_seq_length": 2048,  # Context window for training
    "warmup_ratio": 0.03,  # Gradual learning rate warmup

    # Optimization
    "optim": "paged_adamw_8bit",  # Memory-efficient optimizer
    "lr_scheduler_type": "cosine",  # Learning rate decay
    "weight_decay": 0.01,  # Regularization
    "max_grad_norm": 1.0,  # Gradient clipping
    "gradient_checkpointing": True,  # Trade compute for memory
    "fp16": True,  # Mixed precision training

    # Data Configuration
    "remove_unused_columns": False,
    "group_by_length": False,  # Keep natural length variation

    # NO CONTENT FILTERING
    # This is critical for authentic personality
    "filter_profanity": False,
    "normalize_text": False,
    "remove_special_tokens": False,

    # Logging
    "logging_steps": 10,
    "save_steps": 500,
    "eval_steps": 500,
    "evaluation_strategy": "steps",

    # Hardware
    "device_map": "auto",
    "ddp_find_unused_parameters": False,
}

Estimated Training Time (RTX 3070 8GB):
- Setup and data loading: 10-15 minutes
- Training (5 epochs): 4-5 hours
- Total: ~5 hours

Peak VRAM Usage: 7.2-7.5 GB
```

### Direct Preference Optimization (DPO)

**Purpose**: Reinforce preferred communication patterns using reaction data as preference signal.

**Strategy**:
```python
Preference Pair Creation:
- CHOSEN: Messages with positive reactions (👍, ❤️, 😂, etc.)
- REJECTED: Generic or formal alternatives

Example:
CHOSEN: "BRUH that's wild 😂😂"  [5 laugh reactions]
REJECTED: "That's quite interesting."  [0 reactions]

DPO Configuration:
{
    "method": "DPO",
    "beta": 0.1,  # Preference strength
    "num_train_epochs": 2,
    "learning_rate": 5e-5,  # Lower than SFT
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "max_seq_length": 2048,
}

Expected Improvement:
- +20-30% style consistency
- Better alignment with community preferences
- More natural reaction to different contexts

Training Time: 1-2 hours on RTX 3070
```

**ORPO Alternative** (Odds Ratio Preference Optimization):
```python
# Combines SFT + DPO in single training pass
# Faster than sequential SFT→DPO
# Similar final quality

{
    "method": "ORPO",
    "num_train_epochs": 5,
    "learning_rate": 8e-6,
    "lambda_orpo": 0.1,  # Preference weight
}

Training Time: 5-6 hours total (vs 6-7 for SFT+DPO)
```

### Evaluation Strategy

**Quantitative Metrics**:
```python
1. Perplexity on validation set
   - Target: <3.0 (lower = better prediction)

2. Style embedding similarity
   - Compare embeddings of generated vs real messages
   - Target: >0.85 cosine similarity

3. Length distribution match
   - Generated message lengths should match training data

4. Vocabulary overlap
   - Generated text should use community vocabulary
```

**Qualitative Evaluation** (Most Important):
```python
Human Evaluation Protocol:
1. Generate 50 sample responses
2. Mix with 50 real server messages
3. Ask server members to identify which are bot
4. Target: <60% detection rate (indistinguishable)

Personality Rubric (1-5 scale):
- Authenticity: Sounds like a real server member
- Naturalness: Flows like normal conversation
- Appropriateness: Matches context and tone
- Vocabulary: Uses community slang/jargon
- Length: Appropriate message length

Target Score: 4.0+ average across all dimensions
```

**Iterative Improvement**:
```python
If personality match < 85%:
1. Analyze failure cases (what feels "off"?)
2. Collect more training data (especially of lacking patterns)
3. Adjust hyperparameters (increase LoRA rank, more epochs)
4. Retrain and re-evaluate

Typical iterations needed: 1-3
```

## Response Generation System

### System Prompt Design

**Principles**:
- Natural and human-like (no robotic instructions)
- No hardcoded numbers or rules
- Emphasize authenticity over helpfulness
- Short and direct

**Recommended System Prompt**:
```
You're a regular on this Discord server. You've been here a while,
you know the vibe, you know the people.

Just chat like you normally would. Be yourself. If something's funny,
laugh. If something's dumb, call it out. If you want to spam a word
for emphasis, do it. Short responses are fine. Long rants are fine too.

Don't act like you're here to help or assist - you're just hanging out.
No formality, no AI speak, just natural conversation.

Match the energy of the room.
```

**Alternative: No System Prompt**:
```python
# Let the fine-tuned model BE the personality
# System prompt optional if model deeply trained
# Test both approaches and compare
```

### Generation Parameters

```python
Inference Configuration:
{
    "temperature": 0.75,  # Balance creativity and consistency
    "top_p": 0.9,  # Nucleus sampling
    "top_k": 40,  # Top-k sampling
    "repetition_penalty": 1.1,  # Mild penalty for repetition
    "max_tokens": 150,  # Typical Discord response length
    "stop_sequences": ["<|im_end|>", "\n\nUser:", "\n\nuser:"],
}

Adjustable via GUI:
- Temperature: 0.5-1.0 (lower = safer, higher = more creative)
- Response rate: 1-100% (default 5%)
- Max tokens: 50-500 (default 150)
```

### Response Flow

```
Incoming Discord Message
    ↓
Admin Exclusion Check (if any excluded users)
    - User admin-excluded? Skip
    ↓
Context Analysis
    - Is bot mentioned?
    - Is in active conversation thread?
    - Has bot responded recently in channel?
    ↓
Probability Check
    - Random roll vs configured rate (default 5%)
    - Always respond if directly mentioned
    ↓
[Optional] Context Retrieval
    - Search LanceDB for 3-5 relevant messages
    - Provide background context only
    - Don't inject style examples (model has personality)
    ↓
Prompt Construction
    - System prompt (if used)
    - Retrieved context (if any)
    - Recent conversation (last 5 messages)
    - Current user message
    ↓
Generation (llama.cpp)
    - Model already loaded in memory
    - Async execution (thread pool)
    - KV cache for multi-turn
    - 2-3 seconds on laptop CPU
    ↓
Post-Processing (MINIMAL)
    - Remove system artifacts (rare)
    - Trim to 2000 chars (Discord limit)
    - No style filtering
    - No content sanitization
    ↓
Send Response to Discord
    ↓
Update Conversation Context
    - Store in SQLite
    - Add to LanceDB (async)
```

### Performance Optimization

**Critical Optimizations**:
```python
1. Model Persistence (Saves 20+ seconds per response)
   - Load model once at bot startup
   - Keep in memory between requests
   - Graceful handling of memory pressure

2. Async Execution (No blocking)
   - Run inference in thread pool
   - Never block Discord event loop
   - Handle multiple requests concurrently

3. KV Cache (Saves 30% in multi-turn)
   - Cache key-value pairs from previous turns
   - Reuse for conversation continuation
   - Implement cache expiry (5-10 minutes)

4. Multi-Threading (Use all CPU cores)
   - llama.cpp automatic thread optimization
   - Typically uses 8-12 threads
   - Configure via n_threads parameter

5. Response Caching (Optional)
   - Cache identical or very similar queries
   - 5-minute TTL
   - Saves compute for common phrases
   - Implement conservatively (don't hurt personality)
```

**Performance Targets**:
```
Metric                  Target      Acceptable    Unacceptable
────────────────────────────────────────────────────────────────
Model Load Time         15-20s      30s           >45s
Response Time (p50)     2s          3s            >5s
Response Time (p95)     3s          5s            >8s
Response Time (p99)     5s          8s            >12s
Memory Usage (Stable)   3GB         4GB           >5GB
Memory Usage (Peak)     4GB         5GB           >6GB
Uptime                  99%         95%           <90%
Error Rate              <0.1%       <1%           >2%
```

## GUI Management Application

### Architecture

**Framework**: CustomTkinter 5.2+
**Pattern**: Single-process GUI controlling bot subprocess
**Communication**: File-based (SQLite) + process signals

### Main Window Components

**1. Status Display**:
```python
- Bot state: ● Running / ● Stopped / ⚠ Error
- Uptime counter
- Current memory usage
- Model loaded indicator
- Last response timestamp
```

**2. Control Buttons**:
```python
[Start Bot]    - Launch bot subprocess
[Stop Bot]     - Graceful shutdown (finish current response)
[Restart]      - Stop + Start sequence
[Force Stop]   - Immediate termination (emergency)
[View Logs]    - Open logs window
```

**3. Configuration Sliders**:
```python
Response Rate:   [=====>         ]  5%
Temperature:     [===========>   ]  0.75
Max Tokens:      [======>        ]  150

Changes apply immediately (no bot restart needed)
Persisted to SQLite configuration table
```

**4. Statistics Dashboard**:
```python
Last 24 Hours:
- Messages Seen:      1,247
- Responses Sent:     63 (5.1%)
- Avg Response Time:  2.3s
- Errors:             0

Last 7 Days:
- Total Responses:    421
- Unique Conversations: 87
- Fastest Response:   1.8s
- Slowest Response:   4.2s
```

**5. Quick Actions**:
```python
[Fetch New Messages]   - Run history scraper
[Retrain Model]        - Start training pipeline
[Open Data Folder]     - Open file explorer
[Settings]             - Open settings window
```

### System Tray Integration

**Features**:
```python
System Tray Icon:
- Right-click menu:
  - Show/Hide Window
  - Start/Stop Bot
  - Quick Stats (tooltip)
  - Exit Application

Notifications:
- Bot started successfully
- Bot stopped
- Error occurred
- Response rate milestone (100, 500, 1000 responses)

Background Operation:
- Minimize to tray instead of taskbar
- Bot continues running when window closed
- Low resource usage (~30MB GUI process)
```

### Logs Window

**Features**:
```python
- Real-time log streaming from bot process
- Color-coded by severity (INFO, WARNING, ERROR)
- Auto-scroll toggle
- Search/filter logs
- Export logs to file
- Clear logs button
- Timestamp for each entry

Example Log Format:
[2025-11-02 14:32:15] [INFO] Bot started
[2025-11-02 14:32:18] [INFO] Model loaded: Qwen2.5-3B (2.2GB)
[2025-11-02 14:35:42] [INFO] Response sent in 2.4s
[2025-11-02 14:38:11] [WARNING] High memory usage: 4.2GB
```

### Settings Window

**Configurable Options**:
```python
Startup:
- ☑ Start bot automatically when GUI launches
- ☑ Launch GUI on Windows startup
- ☐ Minimize to tray on startup

Notifications:
- ☑ Show desktop notifications
- ☐ Sound effects
- Notification Level: [Errors Only ▼]

Discord Configuration:
- Bot Token: [••••••••••••] [Show] [Edit]
- Server ID: 123456789012345678
- Channels: [#general, #chat, #memes]

Model Configuration:
- Model Path: [.../models/qwen2.5-3b-q4.gguf]
- Context Length: [2048 ▼]
- GPU Layers: [0 ▼] (CPU-only)

Admin Privacy Controls (hidden):
- Excluded Users: [View/Edit List]
- Data Retention: [Forever ▼]

Advanced:
- Thread Count: [Auto ▼]
- Log Level: [INFO ▼]
- Enable Debug Mode: ☐
```

### Bot Process Management

**Process Controller**:
```python
class BotController:
    """Manages bot subprocess lifecycle"""

    def start_bot(self):
        # Launch bot as subprocess
        # Monitor stdout/stderr
        # Update GUI state
        # Enable stop/restart buttons

    def stop_bot(self, timeout=10):
        # Send SIGTERM (graceful)
        # Wait up to timeout seconds
        # Force kill if necessary
        # Update GUI state

    def restart_bot(self):
        # Stop + Start sequence
        # Preserve configuration

    def check_health(self):
        # Verify process alive
        # Check memory usage
        # Detect hung state
        # Auto-restart if configured

    def read_logs(self):
        # Stream stdout/stderr
        # Parse and format
        # Send to GUI log window
```

### Auto-Start on Boot

**Windows Implementation**:
```python
import winreg

def add_to_startup():
    """Add GUI to Windows startup"""
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    key = winreg.OpenKey(
        winreg.HKEY_CURRENT_USER,
        key_path,
        0,
        winreg.KEY_SET_VALUE
    )
    exe_path = os.path.abspath("launcher.exe")  # Or launcher.py
    winreg.SetValueEx(
        key,
        "DiscordPersonalityBot",
        0,
        winreg.REG_SZ,
        f'"{exe_path}"'
    )
    winreg.CloseKey(key)

def remove_from_startup():
    """Remove from Windows startup"""
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    key = winreg.OpenKey(
        winreg.HKEY_CURRENT_USER,
        key_path,
        0,
        winreg.KEY_SET_VALUE
    )
    try:
        winreg.DeleteValue(key, "DiscordPersonalityBot")
    except FileNotFoundError:
        pass
    winreg.CloseKey(key)
```

## Privacy and Data Management

### Privacy Philosophy

**Data Handling**:
- All data stored locally only
- No external sharing or cloud uploads
- Server-wide personality training (blended from all users)
- Admin-only data management controls

### Admin-Only Privacy Controls

**Database Schema** (for admin use):
```sql
CREATE TABLE excluded_users (
    user_id TEXT PRIMARY KEY,
    username TEXT,
    excluded_at TIMESTAMP,
    reason TEXT,
    excluded_by_admin TEXT
);
```

**Admin Exclusion Flow** (hidden feature):
```python
1. Admin uses hidden command: !exclude <user_id> [reason]
2. Bot adds user to excluded_users table
3. Message fetcher excludes user from future scrapes
4. Training pipeline filters out user's messages
5. Existing trained model continues (can't remove from weights)
6. Next retraining cycle excludes user completely
```

**No Public Announcements**:
```python
# Message fetching is silent - no announcements
# No opt-out system advertised to users
# Admin-only controls for edge cases or legal requirements
```

### Data Retention

**Policies**:
```python
Raw Message Data:
- Keep for training and retraining
- Delete only if legally required or admin excluded
- No external sharing or API access
- Stored locally only

Training Data:
- Derivative of raw messages
- Regenerate on each training cycle
- Exclude admin-specified users only
- Delete after training complete (optional)

Model Weights:
- Cannot remove specific user influence
- Requires full retraining to exclude user
- Document training data provenance

Logs:
- Rotate after 30 days (configurable)
- Can anonymize user IDs if needed
- Purge old logs automatically
```

## Commands

### User Commands

**None** - Bot responds naturally, no commands needed for normal operation.

### Admin Commands

```python
!setrate <0.0-1.0>
    Set response probability (0.05 = 5%)
    Example: !setrate 0.1 (10% response rate)

!settemp <0.0-1.0>
    Set generation temperature (higher = more creative)
    Example: !settemp 0.8

!setmlength <50-500>
    Set maximum response length in tokens
    Example: !setmaxlen 200

!status
    Show bot status, statistics, configuration

!restart
    Restart bot process (reloads model if changed)

!fetch
    Manually trigger message history fetching

!train
    Manually trigger model retraining (requires admin)

!help
    Show available commands

!exclude <user_id> [reason] (admin only, hidden)
    Exclude specific user from training data
```

### Admin Privileges

**Configuration**:
```python
# In bot/config.py
ADMIN_USER_IDS = [
    123456789012345678,  # Your Discord user ID
]

# Or via .env
ADMIN_USERS=123456789012345678,987654321098765432
```

## Deployment

### Laptop Deployment Checklist

```
Prerequisites:
□ Python 3.9-3.13 installed
□ Discord bot token obtained
□ Server configured (channels, permissions)
□ Model files downloaded (Qwen2.5-3B Q4_K_M)
□ Training completed (fine-tuned model ready)

Installation:
□ Clone repository
□ Create virtual environment
□ Install requirements: pip install -r requirements.txt
□ Configure .env file (token, server ID, channels)
□ Run launcher.py (GUI opens)

First Run:
□ GUI opens successfully
□ Click "Start Bot"
□ Verify bot online in Discord
□ Test response (mention bot)
□ Check logs for errors
□ Verify memory usage acceptable

Optional Configuration:
□ Enable auto-start on boot
□ Adjust response rate as needed
□ Configure system tray notifications
□ Set up automatic updates (git pull)
```

### Maintenance

**Regular Tasks**:
```python
Weekly:
- Check bot uptime and error logs
- Verify memory usage stable
- Test response quality

Monthly:
- Fetch new messages (incremental)
- Review excluded users list (if any)
- Rotate/archive old logs
- Backup database and model

Quarterly:
- Retrain model with new messages
- Evaluate personality match
- Update dependencies (security patches)
- Review and optimize performance

Annually:
- Comprehensive evaluation
- Consider new base model upgrades
- Major refactoring if needed
```

**Backup Strategy**:
```bash
# Critical files to backup
- .env (Discord token, configuration)
- data_storage/database/ (SQLite databases)
- models/finetuned/ (fine-tuned model weights)

# Backup script (weekly automated)
#!/bin/bash
DATE=$(date +%Y%m%d)
tar -czf backup_$DATE.tar.gz \
    .env \
    data_storage/database/ \
    models/finetuned/
```

## Troubleshooting

### Common Issues

**1. Bot Not Responding**
```
Symptoms: Bot online but doesn't send messages
Diagnostics:
- Check logs for errors
- Verify response rate not too low
- Ensure bot has send message permissions
- Check if model loaded successfully
- Verify enough memory available

Solutions:
- Increase response rate temporarily
- Restart bot via GUI
- Check Discord permissions
- Restart laptop if memory pressure
```

**2. Slow Response Time (>5s)**
```
Symptoms: Bot responds but takes too long
Diagnostics:
- Check CPU usage during generation
- Verify model quantization (Q4_K_M)
- Check if other processes hogging CPU
- Review inference parameters

Solutions:
- Close other applications
- Reduce max_tokens (150 → 100)
- Drop to Q3_K_M quantization
- Increase thread count
- Consider GPU acceleration (if available)
```

**3. High Memory Usage (>5GB)**
```
Symptoms: Bot using excessive RAM
Diagnostics:
- Check model size (should be 2.2GB)
- Verify no memory leaks (increases over time?)
- Check if multiple models loaded
- Review conversation context cache

Solutions:
- Restart bot (clears caches)
- Reduce context window (2048 → 1024)
- Limit conversation history stored
- Use smaller model (Gemma 3-1B)
```

**4. "Model Not Found" Error**
```
Symptoms: Bot fails to start, model loading error
Diagnostics:
- Verify model file exists at configured path
- Check file permissions
- Ensure correct GGUF format
- Verify not corrupted (check file size)

Solutions:
- Re-download model file
- Check path in .env matches actual location
- Verify GGUF file not GIT-LFS placeholder
- Run integrity check (if hash available)
```

**5. Training Fails / Out of Memory**
```
Symptoms: Training crashes on RTX 3070
Diagnostics:
- Check VRAM usage (nvidia-smi)
- Verify batch size not too large
- Check dataset size reasonable
- Review training logs for OOM errors

Solutions:
- Reduce batch_size to 1
- Increase gradient_accumulation
- Enable gradient_checkpointing
- Reduce max_seq_length (2048 → 1536)
- Use Qwen2.5-1.5B instead
- Training on Google Colab instead
```

### Performance Degradation

**Symptoms**: Bot was fast, now slow over time

**Diagnostic Steps**:
```python
1. Check conversation context growth
   - SQLite database size increasing?
   - Clear old conversations (>30 days)

2. Check LanceDB index size
   - Vector database growing unbounded?
   - Rebuild index periodically

3. Check for memory leaks
   - Memory usage slowly increasing?
   - Restart bot to clear

4. Check model cache
   - KV cache growing too large?
   - Implement cache expiry

5. Check laptop health
   - CPU thermal throttling?
   - Dust buildup reducing performance?
   - Background updates running?
```

### Getting Help

**Resources**:
```
1. Project Documentation
   - README.md: Quick start guide
   - CLAUDE.md: This comprehensive guide
   - TODO.md: Implementation checklist

2. Logs
   - Enable debug logging
   - Check bot logs via GUI
   - Review training logs

3. Community
   - Unsloth Discord: Fine-tuning help
   - llama.cpp GitHub: Inference issues
   - discord.py Discord: Bot framework help

4. Model Providers
   - Qwen GitHub: Model-specific issues
   - HuggingFace Forums: General ML help
```

## Future Enhancements

### Planned Features

**Short-Term** (Next 3-6 months):
```
- Voice channel support (text-to-speech with personality)
- Image understanding (Qwen2.5-VL upgrade)
- Multi-server deployment (personality per server)
- Improved conversation threading
- Reaction learning (adjust based on user reactions)
- Advanced admin dashboard (web-based)
```

**Medium-Term** (6-12 months):
```
- Continual learning (incremental fine-tuning)
- Multi-agent conversations (bot talks to bots)
- Personality mixing (blend multiple servers)
- Voice cloning (match voice personality)
- Meme generation (image + text personality)
- Mobile app control (iOS/Android)
```

**Long-Term** (12+ months):
```
- Real-time learning (online learning)
- Personality evolution tracking
- A/B testing framework
- Multi-modal personality (text + voice + image)
- Federation (share personality across servers)
- Commercial deployment toolkit
```

### Upgrade Paths

**Model Upgrades**:
```
Future models to consider:
- Qwen 3.0 (when released)
- Llama 4.x series
- Phi-5 (if released)
- Gemma 4

Evaluation criteria:
- Creative writing benchmarks
- Personality retention after fine-tuning
- Inference speed on consumer hardware
- Context window size
- License terms
```

**Hardware Upgrades**:
```
Training:
- RTX 4070 12GB: More comfortable training
- RTX 4090 24GB: Train 7B models easily
- Multi-GPU: Parallel training

Inference:
- Laptop with discrete GPU: 5-10x faster responses
- Intel Core Ultra (NPU): Efficient on-device AI
- Apple M-series: MLX optimization
- Raspberry Pi 5 + Neural Compute: Edge deployment
```

## Research References

### Models Evaluated (November 2025)

**Qwen2.5 Series**:
- Technical Report: https://arxiv.org/abs/2412.15115
- HuggingFace: https://huggingface.co/Qwen
- Performance: Best creative writing, 128K context
- License: Apache 2.0

**Llama 3.2**:
- Meta AI Release: September 2024
- Sizes: 1B, 3B (text-only)
- Performance: Good creative writing, on-device optimized
- License: Llama 3 Community License

**Gemma 3**:
- Google Release: March 2025
- Sizes: 1B (text), 4B/12B/27B (multimodal)
- Performance: Ultra-efficient, quantization-aware trained
- License: Gemma Terms of Use

**Phi-4**:
- Microsoft Release: December 2024
- Size: 14B parameters
- Performance: Best math/reasoning, synthetic data trained
- License: MIT (open weights)

**DeepSeek-R1**:
- DeepSeek AI Release: 2025
- Distilled sizes: 1.5B, 7B, 14B
- Performance: Strong reasoning, less creative
- License: MIT

**Ministral**:
- Mistral AI Release: October 2024
- Sizes: 3B, 8B
- Performance: Edge-optimized, reasoning focus
- License: Apache 2.0

### Fine-Tuning Research

**QLoRA**:
- Paper: "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
- ArXiv: https://arxiv.org/abs/2305.14314
- Key Innovation: 4-bit quantization + LoRA for 7B on single GPU

**DPO**:
- Paper: "Direct Preference Optimization" (2023)
- ArXiv: https://arxiv.org/abs/2305.18290
- Key Innovation: RLHF without reward model

**ORPO**:
- Paper: "ORPO: Monolithic Preference Optimization" (2024)
- ArXiv: https://arxiv.org/abs/2403.07691
- Key Innovation: SFT + preference in single pass

**Unsloth**:
- GitHub: https://github.com/unslothai/unsloth
- Performance: 2x faster training, 70% less VRAM
- Support: Qwen2.5, Llama 3.2, Gemma, Phi-4

### Benchmarks Cited

**Creative Writing**:
- Study: "Evaluating LLM Performance: DeepSeek R1, Llama 3.2, Qwen 2.5, Gemma 2B" (January 2025)
- Finding: Llama and Qwen superior for creative writing

**Instruction Following**:
- Benchmark: IFEval (Instruction Following Evaluation)
- Qwen2.5-3B: 77.4 score
- Llama 3.2-3B: 73.2 score

**Model Efficiency**:
- Source: Artificial Analysis (artificialanalysis.ai)
- Metrics: Tokens/sec, cost/1M tokens, latency

## License and Legal

**This Project**:
- Code: MIT License (permissive)
- Models: Subject to individual model licenses
- Training Data: User-provided (Discord messages)

**Model Licenses**:
- Qwen2.5: Apache 2.0 (commercial use OK)
- Llama 3.2: Llama 3 Community License (restrictions apply)
- Gemma 3: Gemma Terms of Use (attribution required)
- Phi-4: MIT (fully open)

**Important Legal Considerations**:
```
1. Discord Terms of Service
   - Bots must follow Discord TOS
   - Don't scrape without permission
   - Respect rate limits

2. User Privacy
   - Admin-only exclusion controls (for legal compliance)
   - Comply with GDPR/CCPA if legally required
   - Store data securely locally only
   - Never share training data externally

3. Content Generation
   - Bot may generate offensive content (personality replication)
   - Consider content filters for public servers
   - Monitor for ToS violations
   - Admin liability for bot behavior

4. Commercial Use
   - Check model license if monetizing
   - Qwen2.5 and Phi-4 allow commercial use
   - Llama 3.2 has revenue restrictions
   - Consult lawyer for commercial deployment
```

## Contact and Support

**Project Maintainer**: [Your Name/Username]

**Issues and Questions**:
- GitHub Issues: [Repository URL]
- Discord Server: [Server invite if applicable]
- Email: [Contact email if public]

**Contributing**:
- Pull requests welcome
- Follow existing code style
- Add tests for new features
- Update documentation

---

Last Updated: November 2025
Document Version: 1.0
