# Google Gemini API Migration Guide

## Overview
The project has been successfully migrated from using local Ollama inference to Google's Gemini API. This migration allows the application to run without requiring significant computational resources on the local machine.

## What Changed

### 1. New Dependencies Added
- `google-generativeai` (v0.8.6) - Google's official Generative AI client
- `langchain-google` (v0.1.1) - LangChain integration for Google APIs
- `google.genai` (v1.73.1) - Newer Google Generative AI package

### 2. Configuration Files

#### `.env` (New)
Created in project root to store sensitive API credentials:
```env
GOOGLE_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2 .0-flash
```

**IMPORTANT**: Never commit `.env` to Git! It's in `.gitignore`.

#### `config.py` (Updated)
- Added `GOOGLE_API_KEY` and `GEMINI_MODEL` configuration
- Kept Ollama config for backward compatibility
- Default model changed to `gemini-2.0-flash`

### 3. New LLM Builder Module
**File**: `chains/llm_builder.py` (New)

Created a custom `GeminiLLM` wrapper class that:
- Extends LangChain's `LLM` base class
- Provides unified interface for both Gemini and Ollama
- Handles API calls to Google's Generative AI API
- Includes error handling and fallbacks

**Usage**:
```python
from chains.llm_builder import build_llm

# Create a Gemini LLM instance
llm = build_llm(model_name="gemini-2.0-flash", provider="gemini", temperature=0)

# Or use Ollama as fallback
llm = build_llm(model_name="qwen3.5:0.8b", provider="ollama", temperature=0)
```

### 4. Updated Chain Modules

All chain files have been updated to use the new `build_llm()` function instead of directly instantiating ChatOllama:

- **`chains/qa_chain.py`**: Updated both `ask_question_phase2()` and `ask_question()` functions
- **`chains/summarize_chain.py`**: Updated `_build_llm()` helper function
- **`chains/insight_chain.py`**: Updated LLM instantiation in `extract_insights()`
- **`chains/hallucination.py`**: Updated LLM instantiation in `check_faithfulness()`

### 5. Updated UI

**File**: `app.py` (Updated)
- Changed sidebar label from "Ollama model" to "Gemini Model"
- Updated caption to reflect cloud-based Gemini instead of local-first Ollama
- Default model changed to `GEMINI_MODEL` from `DEFAULT_OLLAMA_MODEL`

### 6. Updated Requirements

**File**: `requirements.txt` (Updated)
```txt
google-generativeai>=0.8.0
langchain-google>=0.0.15
```

## Setup Instructions

### 1. Get a Google Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the generated key

### 2. Configure Environment

1. Create `.env` file in project root:
   ```env
   GOOGLE_API_KEY=your_actual_api_key_here
   GEMINI_MODEL=gemini-2.0-flash
   ```

2. Or copy and modify `.env.example`:
   ```bash
   cp .env.example .env
   # Edit .env and add your actual API key
   ```

### 3. Install New Dependencies

```bash
cd multi_doc_intelligence
pip install -r requirements.txt
```

## Running the Application

```bash
cd multi_doc_intelligence
streamlit run app.py
```

The application will now use Google Gemini API for:
- Document summarization
- Insight extraction
- Question answering
- Faithfulness checking

## API Pricing & Quotas

### Free Tier Limits
- **Requests per day**: Limited (check Google's rate limit docs)
- **Requests per minute**: Limited
- **Input tokens per minute**: Limited
- **No credit card required** (but limited usage)

### Paid Tier
- More generous quotas
- Better support
- Enable billing in Google AI Studio to upgrade

### Rate Limiting
If you hit rate limits:
- The app will return an error message
- Wait ~10 seconds and retry
- Consider upgrading to paid tier for production use

## Troubleshooting

### Issue: "GOOGLE_API_KEY not set"
**Solution**: Ensure `.env` file exists in project root with valid `GOOGLE_API_KEY`.

### Issue: "429 Quota exceeded"
**Solution**: 
- You've hit the free tier limits
- Wait and retry later, or
- Enable billing for your Google Cloud project

### Issue: Import errors for `google.generativeai`
**Solution**: Reinstall dependencies:
```bash
pip install --upgrade google-generativeai langchain-google
```

### Issue: "Pydantic v1 deprecation warning"
**Solution**: This is a non-critical warning. The app still works fine. It's due to LangChain's compatibility layer.

## Backward Compatibility

The project still supports Ollama if you want to switch back:

```python
from chains.llm_builder import build_llm

# Use Ollama instead of Gemini
llm = build_llm(
    model_name="qwen3.5:0.8b",
    provider="ollama",
    temperature=0
)
```

You can also manually override in the Streamlit UI by changing the "Gemini Model" field to an Ollama model name.

## Next Steps

1. **Test the integration**: Run `python ../test_gemini_integration.py` from `multi_doc_intelligence/` directory
2. **Upload test documents**: Use the Upload page to test the pipeline
3. **Try analysis**: Generate summaries and insights
4. **Ask questions**: Use the Chat page to test QA functionality

## Files Modified

```
multi_doc_intelligence/
├── chains/
│   ├── llm_builder.py (NEW)
│   ├── qa_chain.py (UPDATED)
│   ├── summarize_chain.py (UPDATED)
│   ├── insight_chain.py (UPDATED)
│   └── hallucination.py (UPDATED)
├── config.py (UPDATED)
├── app.py (UPDATED)
├── requirements.txt (UPDATED)
└── .env.example (NEW)

/
├── .env (NEW - NOT IN GIT)
└── test_gemini_integration.py (NEW - for testing)
```

## Migration Success Indicators

✅ Test script runs without errors  
✅ Gemini model instance creates successfully  
✅ API returns responses (even if rate limited)  
✅ All chain modules import correctly  
✅ Streamlit app starts without errors

## Support References

- [Google Gemini API Docs](https://ai.google.dev/api/rest)
- [LangChain Documentation](https://python.langchain.com/)
- [Rate Limits & Quotas](https://ai.google.dev/gemini-api/docs/rate-limits)
