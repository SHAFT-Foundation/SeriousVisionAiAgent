# Vision Agent Development Guide

## Quick Start

### Prerequisites
- Python 3.11 or higher
- PostgreSQL database
- Redis server
- OpenAI API key (optional, for full functionality)

### Installation

1. **Clone and setup the environment:**
```bash
git clone <repository-url>
cd SeriousVisionAiAgent
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
```

2. **Setup environment variables:**
```bash
cp config/.env.example .env
# Edit .env file with your API keys and database settings
```

3. **Setup database:**
```bash
# Create PostgreSQL database
createdb vision_agent

# Run migrations (when implemented)
# alembic upgrade head
```

4. **Start Redis server:**
```bash
redis-server
```

### Running the Application

#### Start the Server
```bash
python start_server.py
```
The server will start on http://localhost:8000

#### Start the Desktop Client
```bash
python start_client.py
```

### Default Hotkeys

- **Ctrl+Alt+C** - Capture and analyze current screen
- **Ctrl+Alt+R** - Capture selected region (not yet implemented)
- **Ctrl+Alt+L** - Repeat last result
- **Ctrl+Alt+M** - Toggle automatic monitoring
- **Ctrl+Alt+Space** - Stop current speech
- **Ctrl+Alt+P** - Pause/resume speech
- **Ctrl+Alt+↑** - Increase verbosity
- **Ctrl+Alt+↓** - Decrease verbosity
- **Ctrl+Alt+T** - Toggle TTS
- **Ctrl+Alt+F1** - Show help
- **Ctrl+Alt+Q** - Quit application

## Architecture Overview

### Server Components
- **FastAPI Backend** (`server/`) - REST API with async support
- **Database Models** (`server/models/`) - SQLAlchemy async models
- **LLM Providers** (`server/services/llm_providers.py`) - OpenAI, Claude, Gemini integration
- **Vision Service** (`server/services/vision_service.py`) - Main processing coordinator
- **API Routes** (`server/api/`) - HTTP endpoints for processing, users, health

### Desktop Client Components
- **Screen Monitor** (`desktop_agent/core/screen_monitor.py`) - Multi-monitor capture with mss
- **Image Processor** (`desktop_agent/core/image_processor.py`) - OpenCV preprocessing
- **TTS Engine** (`desktop_agent/core/tts_engine.py`) - Text-to-speech with pyttsx3
- **Hotkey Manager** (`desktop_agent/core/hotkey_manager.py`) - Global hotkeys with pynput
- **API Client** (`desktop_agent/core/api_client.py`) - Async HTTP client for server communication
- **Main App** (`desktop_agent/main.py`) - Application coordinator

## Development Workflow

### Code Style
```bash
# Format code
black .
isort .

# Lint code
flake8 .

# Type checking
mypy .
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=desktop_agent --cov=server --cov-report=html

# Run specific test file
pytest tests/test_screen_monitor.py
```

### Database Migrations
```bash
# Create new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Downgrade
alembic downgrade -1
```

## API Documentation

### Health Endpoints
- `GET /api/v1/health` - Basic health check
- `GET /api/v1/health/detailed` - Detailed health with system metrics
- `GET /api/v1/health/ready` - Kubernetes readiness probe
- `GET /api/v1/health/live` - Kubernetes liveness probe

### Processing Endpoints
- `POST /api/v1/process` - Process base64 encoded image
- `POST /api/v1/process/upload` - Upload and process image file
- `GET /api/v1/process/job/{job_id}` - Get job status and results

### User Endpoints
- `GET /api/v1/users/{user_id}` - Get user profile
- `GET /api/v1/users/{user_id}/preferences` - Get accessibility preferences
- `POST /api/v1/users/{user_id}/preferences` - Update preferences
- `GET /api/v1/users/{user_id}/stats` - Get usage statistics
- `POST /api/v1/users/{user_id}/feedback` - Submit feedback

## Configuration

### Environment Variables
```bash
# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
DEBUG=false

# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/vision_agent

# Redis
REDIS_URL=redis://localhost:6379/0

# LLM API Keys
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here

# Security
MASTER_PASSWORD=your_encryption_password
ENCRYPTION_SALT=your_salt_here
```

### YAML Configuration
Edit `config/default.yaml` for application settings:
- LLM provider settings
- Processing parameters
- Accessibility defaults
- Security options

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure you're in the virtual environment
   - Check PYTHONPATH includes project root

2. **Screen Capture Issues**
   - On macOS: Grant screen recording permissions
   - On Linux: Install xorg-dev packages

3. **TTS Not Working**
   - Install system TTS engines
   - Check audio output devices

4. **API Connection Failed**
   - Verify server is running on correct port
   - Check firewall settings
   - Ensure database and Redis are accessible

### Debug Mode
```bash
# Start server in debug mode
DEBUG=true python start_server.py

# Start client in debug mode  
python start_client.py --debug
```

### Logging
- Server logs: `logs/vision_agent.log`
- Client logs: `vision_agent.log`
- Set `LOG_LEVEL=DEBUG` for verbose logging

## Contributing

### Adding New LLM Providers
1. Create new provider class in `server/services/llm_providers.py`
2. Inherit from `BaseLLMProvider`
3. Implement required methods
4. Add to `LLMProviderManager`

### Adding New Hotkeys
1. Add action to `HotkeyAction` enum
2. Define key combination in `HotkeyManager._setup_default_hotkeys()`
3. Add callback in `VisionAgentApp._setup_hotkey_callbacks()`

### Adding New API Endpoints
1. Create route function in appropriate `server/api/` module
2. Add to router with proper decorators
3. Include router in `server/main.py`
4. Update API documentation

## Security Considerations

- API keys are stored in environment variables
- Sensitive data can be processed locally
- User data is encrypted before storage
- Rate limiting prevents abuse
- Input validation on all endpoints

## Performance Optimization

- Image preprocessing reduces LLM processing time
- Intelligent caching avoids redundant API calls
- Change detection prevents unnecessary captures
- Async processing maintains responsiveness
- Connection pooling optimizes network usage