
# Global configuration
import os
from dotenv import load_dotenv

load_dotenv()

# OCR configuration
OCR_MODE = os.getenv("OCR_MODE", "api")  # "local" | "api"

# Local OCR service configuration
LOCAL_OCR_HOST = os.getenv("LOCAL_OCR_HOST", "http://localhost:8010")

# API-based OCR configuration
OCR_API_URL = os.getenv("OCR_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
OCR_API_KEY = os.getenv("OCR_API_KEY", "YOURAPIKEY")
OCR_API_MODEL = os.getenv("OCR_API_MODEL", "qwen-vl-ocr-2025-11-20")

# VLM configuration
VLM_PROVIDER = os.getenv("VLM_PROVIDER", "openai")  # "openai" | "anthropic" | "local"
VLM_API_KEY = os.getenv("VLM_API_KEY", "YOURAPIKEY")
VLM_MODEL = os.getenv("VLM_MODEL", "qwen3-vl-flash")
VLM_BASE_URL = os.getenv("VLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")