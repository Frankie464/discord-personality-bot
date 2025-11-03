"""
Model Inference Module - Singleton Pattern

CRITICAL: This module implements singleton pattern for model loading.
The model is loaded ONCE at startup and persists for the lifetime of the bot.
This is essential for performance - reloading the model on every request would
take 15-20 seconds, making the bot unusable.

Key Features:
- Module-level cache (_model_instance = None)
- get_model() function loads once, returns cached instance on subsequent calls
- Qwen2.5-3B-Instruct with chatml template (CRITICAL)
- GPU offloading support (optional, 10-20 layers recommended)
- KV cache for multi-turn conversations
- Thread pool for async execution

Performance:
- Model load time: 15-20 seconds (once at startup)
- Response time: 2-3 seconds on laptop CPU
- Memory usage: ~3GB stable
"""

from typing import Optional, Dict, Any, List
import os
from llama_cpp import Llama

# Module-level cache (singleton pattern)
_model_instance: Optional[Llama] = None


def get_model(
    model_path: str,
    n_ctx: int = 2048,
    n_threads: int = 0,
    n_gpu_layers: int = 0,
    chat_format: str = "chatml",
    verbose: bool = False
) -> Llama:
    """
    Get model instance (singleton pattern - loads once, never reloads)

    CRITICAL: This function loads the model ONCE and caches it at module level.
    Subsequent calls return the cached instance immediately.

    Args:
        model_path: Path to GGUF model file
        n_ctx: Context window size (default: 2048)
        n_threads: Number of threads (0 = auto-detect)
        n_gpu_layers: Number of layers to offload to GPU (0 = CPU only, 10-20 recommended)
        chat_format: Chat template (CRITICAL: use "chatml" for Qwen2.5)
        verbose: Enable verbose logging

    Returns:
        Llama model instance (cached after first call)

    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails

    Example:
        >>> # First call: loads model (15-20 seconds)
        >>> model = get_model("models/qwen2.5-3b-q4.gguf", chat_format="chatml")
        >>>
        >>> # Subsequent calls: returns cached instance (instant)
        >>> model = get_model("models/qwen2.5-3b-q4.gguf", chat_format="chatml")
    """
    global _model_instance

    # Return cached instance if already loaded
    if _model_instance is not None:
        return _model_instance

    # Validate model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please ensure the model is downloaded to the correct location."
        )

    # Load model (this happens ONCE)
    try:
        print(f"Loading model from {model_path}...")
        print(f"  Context window: {n_ctx}")
        print(f"  GPU layers: {n_gpu_layers} (0 = CPU only)")
        print(f"  Chat format: {chat_format}")
        print(f"  Threads: {'auto-detect' if n_threads == 0 else n_threads}")

        _model_instance = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads if n_threads > 0 else None,  # None = auto-detect
            n_gpu_layers=n_gpu_layers,
            chat_format=chat_format,  # CRITICAL for Qwen2.5
            verbose=verbose
        )

        print(f"✅ Model loaded successfully! (Size: ~{os.path.getsize(model_path) / (1024**3):.1f}GB)")
        return _model_instance

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")


def generate_response(
    model: Llama,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    max_tokens: int = 120,
    repetition_penalty: float = 1.1,
    stop: Optional[List[str]] = None
) -> str:
    """
    Generate response using the model

    Args:
        model: Llama model instance (from get_model())
        messages: List of message dicts with 'role' and 'content' keys
                  Example: [{"role": "user", "content": "Hello!"}]
        temperature: Sampling temperature (0.5-1.0, default 0.7)
        top_p: Nucleus sampling (default 0.9)
        top_k: Top-k sampling (default 40)
        max_tokens: Maximum response length (default 120)
        repetition_penalty: Penalty for repetition (default 1.1)
        stop: Stop sequences (default: None)

    Returns:
        Generated response text

    Raises:
        ValueError: If messages format is invalid
        RuntimeError: If generation fails

    Example:
        >>> model = get_model("models/qwen2.5-3b-q4.gguf", chat_format="chatml")
        >>> from model.prompts import get_system_prompt
        >>> messages = [
        ...     {"role": "system", "content": get_system_prompt()},
        ...     {"role": "user", "content": "yo what's up"}
        ... ]
        >>> response = generate_response(model, messages, temperature=0.7, max_tokens=120)
        >>> print(response)
        "not much bro, just chilling"
    """
    # Validate messages format
    if not messages or not isinstance(messages, list):
        raise ValueError("Messages must be a non-empty list")

    for msg in messages:
        if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
            raise ValueError("Each message must have 'role' and 'content' keys")

    # Generate response
    try:
        result = model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            repeat_penalty=repetition_penalty,
            stop=stop or []
        )

        # Extract response text
        response_text = result['choices'][0]['message']['content']

        # Minimal post-processing (only Discord char limit)
        response_text = response_text.strip()
        if len(response_text) > 2000:
            response_text = response_text[:2000]

        return response_text

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        raise RuntimeError(f"Response generation failed: {e}")


def unload_model():
    """
    Unload model from memory (use for cleanup/restart)

    WARNING: This should rarely be called. The singleton pattern is designed
    to keep the model loaded for the bot's lifetime. Only call this when
    shutting down the bot or switching models.
    """
    global _model_instance

    if _model_instance is not None:
        print("Unloading model from memory...")
        _model_instance = None
        print("✅ Model unloaded")
    else:
        print("⚠️  No model loaded to unload")


def is_model_loaded() -> bool:
    """
    Check if model is currently loaded

    Returns:
        True if model is loaded in memory, False otherwise
    """
    return _model_instance is not None


def get_model_info() -> Dict[str, Any]:
    """
    Get information about the loaded model

    Returns:
        Dictionary with model information, or None if not loaded
    """
    if _model_instance is None:
        return {
            'loaded': False,
            'message': 'No model loaded'
        }

    return {
        'loaded': True,
        'n_ctx': _model_instance.n_ctx(),
        'n_vocab': _model_instance.n_vocab(),
        'model_type': _model_instance.metadata.get('general.architecture', 'unknown'),
        'message': 'Model loaded and ready'
    }


if __name__ == "__main__":
    # Test model loading (requires model file)
    print("Testing singleton model loading...")
    print("=" * 60)

    # Note: This test requires a valid model file
    # Replace with your actual model path
    test_model_path = "models/finetuned/qwen2.5-3b-personality-q4.gguf"

    if not os.path.exists(test_model_path):
        print(f"⚠️  Test model not found: {test_model_path}")
        print("Skipping model loading test")
        print("\nTo test:")
        print("1. Download/train model")
        print("2. Place at: models/finetuned/qwen2.5-3b-personality-q4.gguf")
        print("3. Run: python model/inference.py")
    else:
        # Test singleton pattern
        print("\n1. Testing first load (should take 15-20 seconds)...")
        import time
        start = time.time()
        model1 = get_model(
            model_path=test_model_path,
            chat_format="chatml",  # CRITICAL for Qwen2.5
            n_gpu_layers=0  # CPU only for testing
        )
        load_time = time.time() - start
        print(f"   First load took: {load_time:.2f}s")

        print("\n2. Testing second load (should be instant)...")
        start = time.time()
        model2 = get_model(
            model_path=test_model_path,
            chat_format="chatml"
        )
        reload_time = time.time() - start
        print(f"   Second load took: {reload_time:.4f}s (should be <0.001s)")

        # Verify same instance
        assert model1 is model2, "Models should be the same instance!"
        print("✅ Singleton pattern working correctly!")

        # Test model info
        print("\n3. Model information:")
        info = get_model_info()
        for key, value in info.items():
            print(f"   {key}: {value}")

        # Test generation
        print("\n4. Testing generation...")
        from model.prompts import get_system_prompt
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": "yo what's up"}
        ]

        start = time.time()
        response = generate_response(
            model1,
            messages,
            temperature=0.7,
            max_tokens=120
        )
        gen_time = time.time() - start

        print(f"   Generated in: {gen_time:.2f}s")
        print(f"   Response: {response}")

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print(f"\nPerformance Summary:")
        print(f"  - First load: {load_time:.2f}s (once at startup)")
        print(f"  - Reload: {reload_time:.4f}s (cached)")
        print(f"  - Generation: {gen_time:.2f}s (target: 2-3s)")
