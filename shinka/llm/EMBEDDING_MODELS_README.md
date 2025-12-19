# Embedding Models Configuration Guide

This guide explains how to configure and extend the embedding models used in Shinka.

## Overview

Embedding models are configured in `shinka/llm/embedding_models.yaml`. This centralized configuration file allows you to:
- Add new embedding providers without code changes
- Update model pricing easily
- Switch between models using environment variables
- Define model metadata (dimensions, context limits, etc.)

## Quick Start

### Using a Different Model

**Option 1: Environment Variable (Recommended)**
```bash
export EMBEDDING_MODEL=gemini-embedding-001
python run_evo.py
```

**Option 2: In Your Script**
```python
from shinka.core import EvolutionConfig

config = EvolutionConfig(
    embedding_model="gemini-embedding-001",
    # ... other config
)
```

### Available Models

Current models defined in `embedding_models.yaml`:

| Model | Provider | Cost/1M tokens | Dimensions | Description |
|-------|----------|----------------|------------|-------------|
| `text-embedding-3-small` | OpenAI | $0.02 | 1536 | Efficient, recommended for 95% of use cases |
| `text-embedding-3-large` | OpenAI | $0.13 | 3072 | High quality when semantic precision matters |
| `gemini-embedding-001` | Google | $0.15 | 768 | Production model with 100+ language support |
| `gemini-embedding-exp-03-07` | Google | Free | 768 | Experimental (deprecated Aug 14, 2025) |
| `azure-text-embedding-3-small` | Azure | $0.02* | 1536 | Azure-hosted OpenAI model |
| `azure-text-embedding-3-large` | Azure | $0.13* | 3072 | Azure-hosted OpenAI model |

*Azure pricing may vary by region

## Adding a New Embedding Model

### Example 1: Adding Cohere Embeddings

1. **Edit `embedding_models.yaml`:**

```yaml
models:
  # ... existing models ...

  # Add new Cohere model
  cohere-embed-english-v3.0:
    provider: cohere
    cost_per_million_tokens: 0.10
    dimensions: 1024
    max_input_tokens: 512
    description: "Cohere's English embedding model v3.0"
    env_vars:
      api_key: COHERE_API_KEY

providers:
  # ... existing providers ...

  # Add Cohere provider configuration
  cohere:
    client_library: cohere
    client_class: Client
    requires_env_vars:
      - COHERE_API_KEY
```

2. **Update `get_client_model()` in `embedding.py`:**

```python
def get_client_model(model_name: str) -> tuple[Union[openai.OpenAI, str], str]:
    # ... existing code ...

    # Add Cohere handler
    elif provider == "cohere":
        import cohere
        env_vars = model_info.get("env_vars", {})
        api_key = os.getenv(env_vars.get("api_key", "COHERE_API_KEY"))
        if not api_key:
            raise ValueError(f"COHERE_API_KEY not set for {model_name}")
        client = cohere.Client(api_key)
        model_to_use = model_name

    # ... rest of code ...
```

3. **Update `get_embedding()` method to handle Cohere:**

```python
def get_embedding(self, code: Union[str, List[str]]) -> ...:
    # ... existing code ...

    # Add Cohere handling
    if self.model_config.get("provider") == "cohere":
        try:
            response = self.client.embed(
                texts=code if isinstance(code, list) else [code],
                model=self.model
            )
            cost = len(code) * cost_per_token  # Approximate
            if single_code:
                return response.embeddings[0], cost
            else:
                return response.embeddings, cost
        except Exception as e:
            logger.error(f"Error getting Cohere embedding: {e}")
            # ... error handling ...
```

4. **Set environment variable and use:**

```bash
export COHERE_API_KEY=your-api-key
export EMBEDDING_MODEL=cohere-embed-english-v3.0
python run_evo.py
```

### Example 2: Adding OpenAI's New Model

If OpenAI releases a new embedding model, you only need to edit the YAML file:

```yaml
models:
  text-embedding-4-small:  # Hypothetical new model
    provider: openai
    cost_per_million_tokens: 0.015  # New pricing
    dimensions: 2048
    max_input_tokens: 16384
    description: "OpenAI's next-gen embedding model"
```

No code changes needed! The existing OpenAI handler will work automatically.

## Configuration Reference

### Model Definition Schema

```yaml
model-name:
  provider: string              # Required: openai, azure, gemini, etc.
  cost_per_million_tokens: float # Required: Cost per 1M tokens
  dimensions: int               # Optional: Embedding dimension size
  max_input_tokens: int         # Optional: Max context length
  description: string           # Optional: Human-readable description
  experimental: bool            # Optional: Mark as experimental
  deprecated: bool              # Optional: Mark as deprecated
  deprecation_date: string      # Optional: Deprecation date (YYYY-MM-DD)
  base_model: string           # Optional: For provider variants (e.g., Azure)
  env_vars:                     # Optional: Model-specific env vars
    api_key: string
    api_version: string
    endpoint: string
```

### Provider Definition Schema

```yaml
provider-name:
  client_library: string        # Required: Python package name
  client_class: string          # Required: Client class name
  strip_prefix: string          # Optional: Prefix to strip from model name
  model_prefix: string          # Optional: Prefix to add to model name
  task_type: string            # Optional: Default task type (for Gemini)
  requires_env_vars: list       # Required: List of required env vars
```

## Environment Variables

### Required by Provider

| Provider | Environment Variables | Purpose |
|----------|---------------------|---------|
| OpenAI | `OPENAI_API_KEY` | API authentication |
| Azure | `AZURE_OPENAI_API_KEY`<br>`AZURE_API_VERSION`<br>`AZURE_API_ENDPOINT` | Azure-specific config |
| Gemini | `GEMINI_API_KEY` | Google API key |

### Global Settings

- `EMBEDDING_MODEL`: Override default model (e.g., `export EMBEDDING_MODEL=gemini-embedding-001`)

## Updating Pricing

When providers update their pricing, simply edit `embedding_models.yaml`:

```yaml
gemini-embedding-001:
  cost_per_million_tokens: 0.20  # Updated from $0.15 to $0.20
```

No code changes or redeployment needed!

## Advanced Usage

### Accessing Model Metadata

```python
from shinka.llm.embedding import load_embedding_config

config = load_embedding_config()
model_info = config["models"]["text-embedding-3-small"]

print(f"Dimensions: {model_info['dimensions']}")
print(f"Cost: ${model_info['cost_per_million_tokens']}/M tokens")
```

### Dynamic Model Selection

```python
import os
from shinka.llm import EmbeddingClient

# Use environment variable or fall back to default
embedding = EmbeddingClient()  # Uses EMBEDDING_MODEL env var or default

# Or specify directly
embedding = EmbeddingClient(model_name="gemini-embedding-001")
```

### Batch Processing Cost Optimization

OpenAI offers 50% savings with batch processing:

```yaml
text-embedding-3-small:
  cost_per_million_tokens: 0.02        # Standard tier
  cost_per_million_tokens_batch: 0.01  # Batch tier (50% off)
```

## Troubleshooting

### Error: "Model not found in config"

Check that the model is defined in `embedding_models.yaml`:

```bash
# List available models
python -c "from shinka.llm.embedding import load_embedding_config; print(list(load_embedding_config()['models'].keys()))"
```

### Error: "Environment variable not set"

Ensure required API keys are set:

```bash
# For OpenAI
export OPENAI_API_KEY=sk-...

# For Gemini
export GEMINI_API_KEY=...

# For Azure
export AZURE_OPENAI_API_KEY=...
export AZURE_API_VERSION=2024-02-15-preview
export AZURE_API_ENDPOINT=https://your-resource.openai.azure.com/
```

### Deprecation Warnings

Models marked as deprecated will show warnings:

```
WARNING: Model gemini-embedding-exp-03-07 is deprecated and will be removed on 2025-08-14.
Please migrate to an alternative model.
```

## Best Practices

1. **Use environment variables** for model selection in production to avoid hardcoding
2. **Monitor costs** by checking the `cost_per_million_tokens` values
3. **Test new models** on a small dataset before switching in production
4. **Keep pricing updated** by regularly checking provider documentation
5. **Document custom models** by adding clear descriptions in the YAML

## References

- [OpenAI Pricing](https://openai.com/api/pricing/)
- [Google Gemini Pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Azure OpenAI Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/)
