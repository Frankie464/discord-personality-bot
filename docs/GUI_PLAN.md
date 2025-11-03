# Complete GUI Management Application Plan

**Status**: Approved for implementation (Phase 5)
**Date**: 2025-11-03
**Version**: 2.0 (Revised - Admin authentication removed)

---

## Overview

Build a comprehensive CustomTkinter GUI for managing the Discord personality bot with full 24/7 operational control, data management, model training, and admin features.

**Key Design Principles**:
- ✅ No admin login required - all features exposed (assumes admin-only access)
- ✅ Clear, user-friendly parameter descriptions
- ✅ Consolidated settings to avoid overlap
- ✅ Real-time monitoring and control
- ✅ Smart reminders for maintenance tasks

---

## Architecture

**Framework**: CustomTkinter 5.2+
**Pattern**: Single-process GUI controlling bot subprocess
**Communication**: SQLite database + subprocess stdout/stderr monitoring
**Dependencies**: Already installed (customtkinter, pillow, pystray, psutil)

---

## Tab Layout

```
┌────────────────────────────────────────────────────────────────┐
│  Discord Personality Bot                            [─][□][×]  │
├────────────────────────────────────────────────────────────────┤
│  [🏠 Dashboard] [💾 Data] [🤖 Training] [⚙️ Settings] [📋 Logs]│
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│                    [Active Tab Content]                         │
│                                                                 │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## 1. Dashboard Tab (Main Control)

### Top Status Panel
```
┌─────────────────────────────────────────────────────────────┐
│  BOT STATUS                                                  │
├─────────────────────────────────────────────────────────────┤
│  Status:     ● Running                   Uptime: 2d 14h 23m │
│  Memory:     3.2 GB / 16 GB              Model: Loaded ✅    │
│  Discord:    Connected ✅                Server: MyServer    │
│  Queue:      0 pending                   Last: 2m ago        │
└─────────────────────────────────────────────────────────────┘
```

### Control Buttons
```
┌─────────────────────────────────────────────────────────────┐
│  BOT CONTROL                                                 │
├─────────────────────────────────────────────────────────────┤
│  [▶️ Start Bot]  [⏹ Stop Bot]  [🔄 Restart]  [⚠️ Force Stop]│
└─────────────────────────────────────────────────────────────┘
```

### Quick Adjustments (Changes saved live)
```
┌─────────────────────────────────────────────────────────────┐
│  QUICK ADJUSTMENTS                                           │
├─────────────────────────────────────────────────────────────┤
│  Response Rate:     [=====>         ]  5%                   │
│  How often bot replies to random messages (not mentions)    │
│                                                              │
│  Temperature:       [=========>     ]  0.7                  │
│  Creativity level (lower = safer, higher = more creative)   │
│                                                              │
│  Max Tokens:        [=====>         ]  120                  │
│  Maximum response length (~1 token = 0.75 words)            │
│                                                              │
│  ☐ Respond only to mentions (ignores response rate)         │
│                                                              │
│  Changes saved automatically • See Settings for more options │
└─────────────────────────────────────────────────────────────┘
```

### Statistics Dashboard
```
┌─────────────────────────────────────────────────────────────┐
│  STATISTICS - LAST 24 HOURS                                  │
├─────────────────────────────────────────────────────────────┤
│  Messages Seen:          1,247        Responses Sent:    63 │
│  Response Rate:          5.1%         Avg Time:        2.3s │
│  Errors:                    0         Slowest:         4.2s │
│                                                              │
│  LAST 7 DAYS                                                 │
│  Total Responses:          421        Conversations:     87 │
│  Uptime:                99.2%         Fastest:         1.8s │
└─────────────────────────────────────────────────────────────┘
```

### Quick Actions
```
┌─────────────────────────────────────────────────────────────┐
│  QUICK ACTIONS                                               │
├─────────────────────────────────────────────────────────────┤
│  [📊 View Full Stats]  [📁 Open Data Folder]  [🔧 Diagnostics]│
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Data Collection Tab

### Top Status Panel
```
┌─────────────────────────────────────────────────────────────┐
│  📊 DATA COLLECTION STATUS                                   │
├─────────────────────────────────────────────────────────────┤
│  Database:           365,401 messages    │  12.3 MB         │
│  Vector DB:          365,401 embeddings  │  450 MB          │
│  Last Fetch:         2 hours ago         │  +3 new messages │
│  Next Recommended:   28 days             │  🔴 Monthly      │
└─────────────────────────────────────────────────────────────┘
```

### Channel Allowlist Management
```
┌─────────────────────────────────────────────────────────────┐
│  ALLOWLISTED CHANNELS                        [+ Add Channel] │
├─────────────────────────────────────────────────────────────┤
│  ☑ #general         │  Last: 2h ago   │  182,456 msgs      │
│  ☑ #chat            │  Last: 2h ago   │  127,893 msgs      │
│  ☑ #memes           │  Last: 2h ago   │   55,052 msgs      │
│                                                              │
│  [✏️ Manage Channels]  [🔄 Refresh List]                     │
└─────────────────────────────────────────────────────────────┘
```

### Fetch Control & Progress
```
┌─────────────────────────────────────────────────────────────┐
│  FETCH NEW MESSAGES                                          │
├─────────────────────────────────────────────────────────────┤
│  Status: ● Idle                                              │
│                                                              │
│  [▶️ Fetch Now]  [⏸️ Stop]  [📋 View Logs]                   │
│                                                              │
│  Progress: [████████░░░░░░░░░░░░░░░░░░]  40%               │
│  Current: Fetching #general... (1,234 messages)              │
│  Elapsed: 2m 15s  │  Estimated: 3m remaining                │
└─────────────────────────────────────────────────────────────┘
```

### Fetch History
```
┌─────────────────────────────────────────────────────────────┐
│  RECENT FETCH HISTORY                        [📊 View All]   │
├─────────────────────────────────────────────────────────────┤
│  2025-11-03 14:36  │  +3 msgs     │  +3 embeddings  │  ✅  │
│  2025-10-05 03:00  │  +1,234 msgs │  +1,234 embed   │  ✅  │
│  2025-09-08 03:00  │  +2,156 msgs │  +2,156 embed   │  ✅  │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Model Training Tab

### Top Status Panel
```
┌─────────────────────────────────────────────────────────────┐
│  🤖 MODEL TRAINING STATUS                                    │
├─────────────────────────────────────────────────────────────┤
│  Current Model:      Qwen2.5-3B-finetuned   │  2.2 GB       │
│  Base Model:         Qwen/Qwen2.5-3B-Instruct               │
│  Last Training:      87 days ago (2025-08-07)               │
│  Training Data:      365,401 messages                        │
│  Next Recommended:   3 days                  │  🔴 Quarterly│
└─────────────────────────────────────────────────────────────┘
```

### Training History
```
┌─────────────────────────────────────────────────────────────┐
│  TRAINING HISTORY                            [📊 View All]   │
├─────────────────────────────────────────────────────────────┤
│  2025-08-07  │  SFT+DPO  │  365K msgs  │  6h 23m  │  ✅     │
│  2025-05-12  │  SFT      │  298K msgs  │  4h 51m  │  ✅     │
│  2025-02-03  │  SFT      │  156K msgs  │  3h 12m  │  ✅     │
└─────────────────────────────────────────────────────────────┘
```

### Training Control
```
┌─────────────────────────────────────────────────────────────┐
│  START NEW TRAINING                                          │
├─────────────────────────────────────────────────────────────┤
│  Mode: ● SFT only    ○ SFT + DPO                            │
│  Estimated Time: 4-5 hours (RTX 3070)                        │
│                                                              │
│  ⚠️  Training requires RTX 3070 machine, not this laptop    │
│                                                              │
│  [📋 View Training Guide]  [✅ Mark as Trained]              │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Settings Tab

### General Settings
```
┌─────────────────────────────────────────────────────────────┐
│  GENERAL                                                     │
├─────────────────────────────────────────────────────────────┤
│  Startup:                                                    │
│  ☑ Start bot automatically when GUI launches                │
│  ☑ Launch GUI on Windows startup                            │
│  ☐ Minimize to tray on startup                              │
│                                                              │
│  Notifications:                                              │
│  ☑ Show desktop notifications                               │
│  ☐ Play sound effects                                       │
│  Notification Level: [All ▼] (All/Warnings/Errors Only)     │
└─────────────────────────────────────────────────────────────┘
```

### Discord Configuration
```
┌─────────────────────────────────────────────────────────────┐
│  DISCORD                                                     │
├─────────────────────────────────────────────────────────────┤
│  Bot Token:       [••••••••••••••••••]  [👁 Show] [✏️ Edit] │
│  Server ID:       1234567890123456789                        │
│  Channels:        #general, #chat, #memes (3 channels)      │
│                   [✏️ Edit Channel Allowlist]                │
└─────────────────────────────────────────────────────────────┘
```

### Model Configuration
```
┌─────────────────────────────────────────────────────────────┐
│  MODEL                                                       │
├─────────────────────────────────────────────────────────────┤
│  Model Path:      models/finetuned/qwen-3b-q4.gguf         │
│                   [📁 Browse] [🔄 Reload Model]              │
│                                                              │
│  Chat Template:   [chatml ▼]                                │
│  The format for structuring conversation history            │
│                                                              │
│  Context Length:  [2048 ▼] (512/1024/2048/4096/8192)        │
│  How many tokens of conversation history to remember        │
│  (Longer = more memory, slower response)                    │
│                                                              │
│  GPU Layers:      [0 ▼] (0=CPU-only, 10-35 if GPU)          │
│  Number of model layers to run on GPU (0 = CPU only)        │
│  Set to 0 for laptop deployment                             │
│                                                              │
│  Thread Count:    [Auto ▼] (Auto/4/8/12/16)                 │
│  CPU cores to use for inference (Auto = optimal)            │
│                                                              │
│  Embedding Model: BAAI/bge-small-en-v1.5 (384-dim)          │
│  Model for generating message embeddings (RAG context)      │
│                                                              │
│  Vector DB Path:  data_storage/embeddings                   │
└─────────────────────────────────────────────────────────────┘
```

### Behavior Configuration
```
┌─────────────────────────────────────────────────────────────┐
│  BEHAVIOR                                                    │
├─────────────────────────────────────────────────────────────┤
│  Response Mode:                                              │
│  ● Random probability (uses response rate below)            │
│  ○ Only when mentioned                                      │
│  ○ Respond to everything (testing only)                     │
│                                                              │
│  Response Rate:        [5%]                                 │
│  Chance to reply to random messages (when in Random mode)   │
│  5% = replies to ~1 in 20 messages                          │
│  Always responds to @mentions regardless of this setting    │
│                                                              │
│  ────────────────────────────────────────────────           │
│  GENERATION PARAMETERS                                       │
│  ────────────────────────────────────────────────           │
│                                                              │
│  Temperature:          [0.7]   (0.0 - 1.0)                  │
│  Controls randomness and creativity                         │
│  • 0.0-0.3: Very focused, deterministic, safe               │
│  • 0.4-0.7: Balanced creativity (recommended)               │
│  • 0.8-1.0: Very creative, unpredictable                    │
│                                                              │
│  Top P (Nucleus):      [0.9]   (0.0 - 1.0)                  │
│  Limits word choice to most likely options                  │
│  • 0.5: Very focused vocabulary                             │
│  • 0.9: Balanced (recommended)                              │
│  • 1.0: Uses all vocabulary                                 │
│                                                              │
│  Top K:                [40]    (1 - 100)                    │
│  Maximum number of words to consider at each step           │
│  • 10-20: More predictable                                  │
│  • 40: Balanced (recommended)                               │
│  • 80+: More varied                                         │
│                                                              │
│  Max Tokens:           [120]   (50 - 500)                   │
│  Maximum response length in tokens                          │
│  • ~1 token = 0.75 words                                    │
│  • 120 tokens ≈ 90 words (short paragraph)                  │
│  • Discord limit: 2000 characters                           │
│                                                              │
│  Repetition Penalty:   [1.1]   (1.0 - 1.5)                  │
│  Discourages repeating the same words/phrases               │
│  • 1.0: No penalty (may repeat)                             │
│  • 1.1: Mild penalty (recommended)                          │
│  • 1.3+: Strong penalty (more varied)                       │
│                                                              │
│  [Reset to Defaults]                                        │
└─────────────────────────────────────────────────────────────┘
```

### Reminder Settings
```
┌─────────────────────────────────────────────────────────────┐
│  REMINDERS                                                   │
├─────────────────────────────────────────────────────────────┤
│  Data Fetch Reminders:                                       │
│  ☑ Enable data fetch reminders                              │
│  Interval: [Monthly ▼] (Weekly/Monthly/Custom)              │
│  Reminds you to fetch new Discord messages for training     │
│                                                              │
│  Model Training Reminders:                                   │
│  ☑ Enable training reminders                                │
│  Interval: [Quarterly ▼] (Monthly/Quarterly/Biannual)       │
│  Reminds you to retrain the model with new data             │
│                                                              │
│  Notification Options:                                       │
│  ☑ Desktop notifications                                    │
│  ☑ In-app badges on tabs                                    │
│  Snooze duration: [7 days ▼]                                │
│  How long to wait before reminding again after snooze       │
└─────────────────────────────────────────────────────────────┘
```

### Privacy & Data Management
```
┌─────────────────────────────────────────────────────────────┐
│  PRIVACY & DATA                                              │
├─────────────────────────────────────────────────────────────┤
│  User Exclusion:                                             │
│  Exclude specific users from future training data           │
│  [👥 View Excluded Users] (0 users excluded)                │
│  [➕ Exclude User by ID]                                     │
│                                                              │
│  ⚠️  Note: Existing model already learned from all messages │
│  Exclusion only affects future training runs                │
│                                                              │
│  ────────────────────────────────────────────────           │
│                                                              │
│  Data Maintenance:                                           │
│  [🗑️ Clear Old Messages] (Messages older than X days)       │
│  [📦 Backup Database]                                        │
│  [♻️ Compact Vector DB] (Optimize disk usage)                │
│  [🔄 Rebuild Vector Index] (Fix search performance)          │
└─────────────────────────────────────────────────────────────┘
```

### Advanced Settings
```
┌─────────────────────────────────────────────────────────────┐
│  ADVANCED                                                    │
├─────────────────────────────────────────────────────────────┤
│  Logging:                                                    │
│  Log Level:        [INFO ▼] (DEBUG/INFO/WARNING/ERROR)      │
│  • DEBUG: Very detailed (troubleshooting only)              │
│  • INFO: Normal operation (recommended)                     │
│  • WARNING: Only warnings and errors                        │
│  • ERROR: Only errors                                       │
│                                                              │
│  Log File:         logs/bot.log                             │
│  Max Log Size:     [10 MB]                                  │
│  Backup Count:     [5]  (keeps 5 old log files)             │
│                                                              │
│  ────────────────────────────────────────────────           │
│                                                              │
│  Database:                                                   │
│  Path:             data_storage/database/bot.db             │
│  Size:             12.3 MB                                   │
│  [🔧 Optimize Database]  [📋 View Schema]                    │
│                                                              │
│  ────────────────────────────────────────────────           │
│                                                              │
│  Performance:                                                │
│  ☐ Enable response caching                                  │
│  Caches similar queries for 5 minutes (faster responses)    │
│                                                              │
│  ☑ Enable KV cache                                          │
│  Remembers conversation context (faster multi-turn)         │
│  Cache Expiry:     [10 minutes]                             │
│                                                              │
│  ────────────────────────────────────────────────           │
│                                                              │
│  Debug Mode:                                                 │
│  ☐ Enable debug logging                                     │
│  ☐ Enable trace logging (very verbose - log everything)     │
│  [🔍 Run Diagnostics]                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Logs Tab

### Log Viewer
```
┌─────────────────────────────────────────────────────────────┐
│  LOGS                                     [Auto-scroll ☑]    │
├─────────────────────────────────────────────────────────────┤
│  [2025-11-03 14:32:15] [INFO] Bot started                   │
│  [2025-11-03 14:32:18] [INFO] Model loaded: Qwen2.5-3B      │
│  [2025-11-03 14:35:42] [INFO] Response sent in 2.4s         │
│  [2025-11-03 14:38:11] [WARN] High memory usage: 4.2GB      │
│  ...                                                         │
│                                                              │
│  Filter: [All ▼]  Search: [________]  [🔍]                  │
│                                                              │
│  [💾 Export Logs]  [🗑️ Clear Logs]  [🔄 Refresh]            │
└─────────────────────────────────────────────────────────────┘
```

**Features**:
- Real-time log streaming from bot subprocess
- Color-coded by severity (DEBUG/INFO/WARNING/ERROR)
- Auto-scroll toggle
- Filter by level
- Search functionality
- Export to file
- Clear logs button

---

## 6. System Tray Integration

### System Tray Icon States
- 🟢 Green: Bot running normally
- 🟡 Yellow: Bot running with warnings
- 🔴 Red: Bot stopped or error
- ⚪ Gray: GUI running, bot not started

### Right-Click Menu
```
┌─────────────────────────────┐
│  Discord Personality Bot    │
├─────────────────────────────┤
│  ● Running (2d 14h)         │
│  📊 Quick Stats             │
│  ────────────────────       │
│  🪟 Show Window             │
│  ▶️ Start Bot               │
│  ⏹ Stop Bot                 │
│  🔄 Restart Bot              │
│  ────────────────────       │
│  📊 Data Fetch Due (3d)     │
│  🤖 Training Due (3d)       │
│  ────────────────────       │
│  ⚙️ Settings                │
│  ❌ Exit                     │
└─────────────────────────────┘
```

### Desktop Notifications
- Bot events (started, stopped, crashed)
- Milestones (100, 1000 responses)
- Reminders (data fetch, training)
- Operations complete (fetch, training)

---

## 7. Database Schema Extensions

### New Tables for GUI State
```sql
-- Fetch history tracking
CREATE TABLE fetch_history (
    id INTEGER PRIMARY KEY,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    status TEXT,  -- 'success', 'error', 'cancelled'
    messages_fetched INTEGER,
    embeddings_added INTEGER,
    channels_processed INTEGER,
    error_message TEXT,
    duration_seconds INTEGER
);

-- Training history tracking
CREATE TABLE training_history (
    id INTEGER PRIMARY KEY,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    status TEXT,
    mode TEXT,  -- 'sft', 'sft+dpo', 'dpo'
    data_size INTEGER,
    base_model TEXT,
    output_model_path TEXT,
    duration_seconds INTEGER,
    notes TEXT
);

-- GUI state persistence
CREATE TABLE gui_state (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP
);

-- GUI settings
CREATE TABLE gui_settings (
    key TEXT PRIMARY KEY,
    value TEXT,
    category TEXT,
    updated_at TIMESTAMP
);
```

---

## 8. File Structure

```
discord-personality-bot/
├── launcher.py                      # Entry point
├── bot_controller.py                # Process management
│
├── gui/
│   ├── __init__.py
│   ├── app.py                       # Main CustomTkinter app
│   │
│   ├── components/
│   │   ├── dashboard_tab.py
│   │   ├── data_collection_tab.py
│   │   ├── model_training_tab.py
│   │   ├── settings_tab.py
│   │   ├── logs_tab.py
│   │   ├── system_tray.py
│   │   ├── channel_manager.py
│   │   ├── training_guide.py
│   │   └── reminder_manager.py
│   │
│   ├── utils/
│   │   ├── subprocess_monitor.py
│   │   ├── progress_parser.py
│   │   └── windows_integration.py
│   │
│   └── assets/
│       ├── icon.ico
│       ├── logo.png
│       └── tray_icons/
```

---

## 9. Implementation Phases

### Phase 1: Core Infrastructure (4-5 hours)
1. bot_controller.py - Subprocess management
2. GUI base application (main window, tabs structure)
3. Dashboard tab (bot control, status, statistics)
4. Settings tab (basic configuration)
5. Logs tab (real-time log viewer)

### Phase 2: Data Management (3-4 hours)
6. Data Collection tab
7. Channel allowlist manager dialog
8. Fetch subprocess control
9. Progress monitoring and parsing

### Phase 3: Training & Reminders (3-4 hours)
10. Model Training tab
11. Training history tracking
12. Reminder system logic
13. Desktop notifications

### Phase 4: Polish & Advanced (2-3 hours)
14. System tray integration
15. Windows startup integration
16. Diagnostics & debugging tools
17. Bug fixes and polish

**Total Estimated Time**: 12-16 hours

---

## 10. Key Improvements from Original Plan

✅ **No admin authentication** - All features exposed (assumes admin-only access)
✅ **Consolidated behavior settings** - Dashboard for quick adjustments, Settings for detailed params
✅ **Clear parameter descriptions** - Every slider/dropdown has user-friendly explanations
✅ **No feature overlap** - Settings organized logically without duplication
✅ **Smart reminders built-in** - Monthly data fetch, quarterly retraining
✅ **Historical tracking** - Fetch and training history preserved in database

---

## 11. Benefits

### User-Friendly
- One-click operations
- Visual feedback (progress bars, status indicators)
- Smart reminders
- System tray integration

### Professional
- Comprehensive control
- Historical tracking
- Performance monitoring
- 24/7 operational design

### Developer-Friendly
- Separation of concerns
- Database-backed state
- Subprocess-based (crash-safe)
- Extensible architecture

---

**Last Updated**: 2025-11-03
**Status**: Ready for implementation after Phase 4 (Training) is complete
