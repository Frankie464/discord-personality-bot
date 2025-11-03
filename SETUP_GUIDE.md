# Setup Guide - Phase 1: Data Collection

This guide will walk you through setting up your Discord bot and collecting message history.

## Prerequisites

- Python 3.9-3.13 installed
- Discord account with a server to test on
- RTX 3070 8GB for training (later phases)

---

## Step 1: Create Discord Bot

### 1.1 Go to Discord Developer Portal

Visit: https://discord.com/developers/applications

### 1.2 Create New Application

1. Click **"New Application"**
2. Name it: `Personality Bot` (or your choice)
3. Click **"Create"**

### 1.3 Create Bot User

1. Click **"Bot"** in left sidebar
2. Click **"Add Bot"**
3. Click **"Yes, do it!"**
4. **IMPORTANT**: Copy your bot token and save it securely
   - Click **"Reset Token"** if you need to regenerate
   - You'll use this in your .env file

### 1.4 Enable Privileged Gateway Intents

**CRITICAL**: These must be enabled for the bot to read messages!

1. Scroll down to **"Privileged Gateway Intents"**
2. Enable these three intents:
   - ✅ **Presence Intent**
   - ✅ **Server Members Intent**
   - ✅ **Message Content Intent** (REQUIRED!)
3. Click **"Save Changes"**

---

## Step 2: Invite Bot to Your Server

### 2.1 Generate Invite URL

1. Click **"OAuth2"** → **"URL Generator"** in left sidebar
2. Select **Scopes**:
   - ✅ `bot`
3. Select **Bot Permissions**:
   - ✅ Read Messages/View Channels
   - ✅ Send Messages
   - ✅ Read Message History
   - ✅ Add Reactions
4. Copy the generated URL at the bottom

### 2.2 Invite Bot

1. Open the copied URL in your browser
2. Select your server from dropdown
3. Click **"Continue"**
4. Click **"Authorize"**
5. Complete CAPTCHA if prompted

✅ Your bot should now appear in your server (offline)

---

## Step 3: Get Required Discord IDs

### 3.1 Enable Developer Mode

1. Open Discord
2. Click Settings (⚙️) → Advanced
3. Enable **"Developer Mode"**
4. Close settings

### 3.2 Get Server ID

1. Right-click your server name
2. Click **"Copy Server ID"**
3. Save this ID (you'll use it in .env)

### 3.3 Get Channel IDs

1. Right-click each channel you want to scrape
2. Click **"Copy Channel ID"**
3. Save all channel IDs (comma-separated)

**Recommended channels**:
- #general
- #chat
- #memes
- Any active conversation channels

**Avoid**:
- Bot command channels
- Admin-only channels
- Low-activity channels

### 3.4 Get Your User ID (Admin)

1. Right-click your username in member list
2. Click **"Copy User ID"**
3. Save this ID (you're the admin)

---

## Step 4: Configure Environment Variables

### 4.1 Create .env File

Copy the template:
```bash
cp .env.example .env
```

### 4.2 Edit .env File

Open `.env` in a text editor and fill in your values:

```env
# Discord Bot Configuration
DISCORD_BOT_TOKEN=YOUR_BOT_TOKEN_HERE
DISCORD_SERVER_ID=YOUR_SERVER_ID_HERE
DISCORD_CHANNEL_IDS=CHANNEL_ID_1,CHANNEL_ID_2,CHANNEL_ID_3

# Admin User IDs (comma-separated)
ADMIN_USER_IDS=YOUR_USER_ID_HERE

# Model Configuration (for later phases)
MODEL_PATH=models/finetuned/qwen2.5-3b-personality-q4.gguf
MODEL_CONTEXT_LENGTH=2048
MODEL_THREADS=0  # 0 = auto-detect

# Generation Parameters
GENERATION_TEMPERATURE=0.75
GENERATION_TOP_P=0.9
GENERATION_TOP_K=40
GENERATION_MAX_TOKENS=150
GENERATION_REPETITION_PENALTY=1.1

# Bot Behavior
RESPONSE_RATE=0.05  # 5% probability to respond
ALWAYS_RESPOND_TO_MENTIONS=true

# Vector Database
VECTOR_DB_PATH=data_storage/embeddings
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Database
DATABASE_PATH=data_storage/database/bot.db

# Logging
LOG_LEVEL=INFO
LOG_FILE=data_storage/bot.log
LOG_MAX_SIZE_MB=50
LOG_BACKUP_COUNT=5

# GUI Settings
GUI_START_BOT_ON_LAUNCH=false
GUI_START_ON_WINDOWS_BOOT=false
GUI_MINIMIZE_TO_TRAY=true
GUI_SHOW_NOTIFICATIONS=true
```

**Example with real values**:
```env
DISCORD_BOT_TOKEN=your_actual_bot_token_goes_here
DISCORD_SERVER_ID=987654321098765432
DISCORD_CHANNEL_IDS=111222333444555666,777888999000111222,333444555666777888
ADMIN_USER_IDS=123456789012345678
```

---

## Step 5: Install Dependencies

### 5.1 Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 5.2 Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 5.3 Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- discord.py (Discord API)
- python-dotenv (environment variables)
- tqdm (progress bars)
- And other dependencies

---

## Step 6: Initialize Database

Run the database initialization script:

```bash
python -c "from storage.database import init_database; db = init_database(); print('✅ Database initialized!')"
```

You should see:
```
✅ Database initialized!
```

Check that the database was created:
```bash
dir data_storage\database\bot.db    # Windows
ls data_storage/database/bot.db     # Linux/Mac
```

---

## Step 7: Test Configuration

Test that your configuration is valid:

```bash
python bot/config.py
```

Expected output:
```
Testing configuration loading...
✅ Configuration loaded successfully!
BotConfig(
  bot_token=********XXXX,
  server_id=987654321098765432,
  channels=3,
  admins=1,
  model_path=models/finetuned/qwen2.5-3b-personality-q4.gguf,
  response_rate=5.0%
)
```

---

## Step 8: Run Message Collection

**Important**: Bot can stay OFFLINE during this step. The bot token provides API access to read messages without the bot being deployed.

### 8.1 Test with Limited Messages (Recommended First)

Test with 100 messages:

```bash
python scripts/fetch_and_embed.py
# Then in database, manually limit or use test database
```

### 8.2 Full Collection

Once you've verified the test works, run full collection:

```bash
python scripts/fetch_and_embed.py
```

**Expected behavior**:
- Loads configuration from .env
- Checks channel allowlist (database)
- Fetches new messages incrementally (since last_message_id)
- Stores in SQLite database
- Creates embeddings in LanceDB
- Shows progress and statistics

**Target**: 20,000-100,000+ messages (entire available history)

**Time estimate**:
- Small server (~20K messages): 10-20 minutes
- Medium server (~50K messages): 30-45 minutes
- Large server (~100K+ messages): 1-2 hours

### 8.3 Verify Data Quality

Check the saved messages:

```bash
# View a sample (Windows)
type data_storage\messages\channel_*.json | more

# View a sample (Linux/Mac)
head -n 50 data_storage/messages/channel_*.json
```

**Quality checklist**:
- ✅ Single-word responses preserved ("lol", "bruh")
- ✅ Emojis preserved (😂, 👍, etc.)
- ✅ Typos preserved (part of personality!)
- ✅ Repeated text preserved ("GGGGGG")
- ✅ No bot messages
- ✅ Message timestamps correct

---

## Step 9: Admin Commands (Testing)

You can test admin commands later when the bot is running, but here's how they work:

### In Discord (when bot is running):

```
!status
```
Shows bot status and statistics

```
!setrate 0.1
```
Set response rate to 10%

```
!settemp 0.8
```
Set temperature to 0.8

```
!exclude 123456789012345678 Testing exclusion
```
**HIDDEN command** - Exclude user from training (SILENT operation)

```
!excluded
```
**HIDDEN command** - List all excluded users

---

## Troubleshooting

### Error: "DISCORD_BOT_TOKEN not set"

- Check that .env file exists
- Check that token is copied correctly (no spaces)
- Token should start with something like "MTIzNDU2..."

### Error: "discord.errors.Forbidden"

- Check that bot has been invited to server
- Check that bot has Read Message History permission
- Check that channels are accessible to bot role

### Error: "Message Content Intent not enabled"

- Go to Discord Developer Portal
- Enable "Message Content Intent" under Bot settings
- Save changes
- Wait 5 minutes for changes to propagate

### Error: "No messages collected"

- Check that channel IDs are correct
- Check that bot has access to channels
- Check that channels have message history
- Try with --limit 10 first to test

### Low message count (<10,000)

- Add more channels to DISCORD_CHANNEL_IDS
- Check that you're scraping active channels
- Check date range in statistics (should be ~15-20 months)

---

## Next Steps

Once you have 20,000+ messages collected:

1. ✅ **Phase 1 Complete!**
2. ➡️ **Phase 2**: Data preprocessing
   - Run `scripts/2_prepare_training_data.py`
   - Format messages for training
   - Create preference pairs (DPO)
3. ➡️ **Phase 3**: Model training (RTX 3070)
4. ➡️ **Phase 4**: Bot development
5. ➡️ **Phase 5**: GUI development

---

## Privacy Notes

**Admin-Only Controls**:
- No public opt-out feature
- No announcements to users
- Silent message collection
- Server-wide personality blend

**Admin Exclusion**:
- Use `!exclude <user_id> [reason]` if needed
- Only for edge cases or legal requirements
- User's messages filtered from future training
- Requires retraining to remove from model

**Data Storage**:
- All data stored locally only
- No external sharing
- Backup .env and data_storage/ regularly

---

## Questions?

Refer to:
- README.md - Project overview
- CLAUDE.md - Technical details
- TODO.md - Full implementation checklist

**Good luck with your personality bot!** 🤖
