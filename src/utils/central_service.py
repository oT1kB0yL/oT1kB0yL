import base64
import io
import httpx
import numpy as np

from typing import Any, List, Tuple, Optional
from PIL import Image
try:
    from openai import OpenAI
    _OPENAI_IMPORT_ERROR: Optional[Exception] = None
except Exception as e:
    OpenAI = None
    _OPENAI_IMPORT_ERROR = e
from tenacity import retry, stop_after_attempt, wait_fixed


class VLM_Inferencer:
    def __init__(
        self,
        vlm_endpoint: str,
        api_key: str,
        model_name: str,
        max_files_allowed: int,
        max_response_tokens: int,
        logger: Any = None,
    ):
        self.vlm_endpoint = vlm_endpoint
        self.api_key = api_key
        self.model_name = model_name
        self.max_files_allowed = max_files_allowed
        self.max_response_tokens = max_response_tokens
        self.logger = logger
        # model parameter settings need to be aligned with the dedicated VLM model,
        # the following is for miniCPM specifically
        self.model_paras = {
            "use_beam_search": True,
            "n": 3,
            "repetition_penalty": 1.2,
            "max_slice_nums": 6,
        }

    def call_vlm(self, b64_img_list: List[str], txt_prompt: str) -> Tuple[int, dict]:
        """Call centralized vlm service with base64 encoded image list and text prompt.

        Args:
            b64_img_list (List[str]): list of base64 encoded images
            txt_prompt (str): text prompt for VLM

        Returns:
            Tuple[int, dict]: HTTP response status code, with restructured VLM contents
        """
        if OpenAI is None:
            return 500, {
                "error": f"openai import failed: {_OPENAI_IMPORT_ERROR}. "
                         f"Please fix your Python SSL/openssl installation and reinstall openai.",
            }
        payload = self.build_openai_message(image_list=b64_img_list, query=txt_prompt)

        last_vlm_error: Optional[str] = None
        @retry(stop=stop_after_attempt(3), wait=wait_fixed(60))
        def _call_vlm_with_retry():
            nonlocal last_vlm_error
            try:
                client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.vlm_endpoint,
                    http_client=httpx.Client(verify=False),
                )
                response_content = client.chat.completions.create(
                    model=self.model_name,
                    max_tokens=self.max_response_tokens,
                    messages=payload,
                    stream=False,
                    extra_body=self.model_paras,
                )
                return 200, self.construct_vlm_response(response_content)
            except Exception as e:
                print(f"[VLM] retrying call, error: {e}")
                last_vlm_error = str(e)
                raise

        try:
            return _call_vlm_with_retry()
        except Exception:
            print(f"[VLM] failed after 3 attempts, last error: {last_vlm_error}")
            return 500, {"error": last_vlm_error or "unknown error"}

    def build_openai_message(self, image_list: List[str], query: str) -> List[dict]:
        """Builds an OpenAI message payload with images and a text query.

        Args:
            image_list (list): A list of base64 encoded images.
            query (str): The text prompt passing into VLM.

        Returns:
            dict: The OpenAI message payload.
        """
        openai_message = {"role": "user", "content": [{"type": "text", "text": query}]}
        for image_data in image_list[: self.max_files_allowed]:
            openai_message["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                }
            )
        return [openai_message]

    def construct_vlm_response(self, vlm_response_content: dict) -> dict:
        """Re-construct VLM response according to the actual requirements.

        Args:
            vlm_response_content (dict): responses from centralized VLM service.

        Returns:
            dict: compact responses selected from raw vlm response.
        """
        return {
            "generated_text": vlm_response_content.choices[0].message.content,
            "response_timestamp": vlm_response_content.created,
            "chat_response_id": vlm_response_content.id,
        }


class OCR_Inferencer:
    def __init__(
        self,
        ocr_endpoint: str,
        ocr_auth_endpoint: str,
        client_app_name: str,
        server_app_name: str,
        api_key: str,
        logger: Any = None,
    ):
        self.ocr_endpoint = ocr_endpoint
        self.ocr_auth_endpoint = ocr_auth_endpoint
        self.client_app_name = client_app_name
        self.server_app_name = server_app_name
        self.api_key = api_key
        self.logger = logger
        self.model_name = server_app_name or "gpt-4o-mini"
        self.authorize()

    def authorize(self) -> None:
        return None

    @staticmethod
    def _process_image(image: Image.Image) -> str:
        width, height = image.size

        if width < 10 or height < 10:
            image = image.resize((max(width, 10), max(height, 10)))
            width, height = image.size

        ratio = max(width, height) / min(width, height)
        if ratio > 200:
            if width > height:
                new_height = max(10, int(width / 199))
                image = image.resize((width, new_height))
            else:
                new_width = max(10, int(height / 199))
                image = image.resize((new_width, height))
            width, height = image.size

        max_pixels = 7680 * 4320
        if width * height > max_pixels:
            scale = (max_pixels / (width * height)) ** 0.5
            image = image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)

        image = image.convert("RGB")
        quality = 90
        while quality > 10:
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=quality)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            if len(img_str) < 10 * 1024 * 1024:
                return img_str
            quality -= 10

        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def call_ocr(self, pil_img: Image.Image, fmt: str = "png") -> List[dict]:
        """Call centralized OCR service with one image.

        Args:
            pil_img (Image.Image): image to be OCRed.
            fmt (str, optional): the format of image. Defaults to 'png'.

        Returns:
            (List[dict]): List of OCR results in dictionary format consisting of
            `angle`, `rec_texts`, `rec_boxes` and `rec_scores`.
        """
        if OpenAI is None:
            return [{
                "angle": 0,
                "rec_texts": [],
                "rec_boxes": [],
                "rec_scores": [],
                "error": f"openai import failed: {_OPENAI_IMPORT_ERROR}. "
                         f"Please fix your Python SSL/openssl installation and reinstall openai.",
            }]
        base64_data = self._process_image(pil_img)
        content = [
            {"type": "text", "text": "Please identify all text in the image and output the content directly."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"}},
        ]

        last_ocr_error: Optional[str] = None

        @retry(stop=stop_after_attempt(3), wait=wait_fixed(60))
        def _call_ocr_with_retry() -> str:
            nonlocal last_ocr_error
            try:
                client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.ocr_endpoint,
                    http_client=httpx.Client(verify=False),
                )
                completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a professional OCR text recognition assistant."},
                        {"role": "user", "content": content},
                    ],
                    stream=True,
                )

                answer_content = ""
                for chunk in completion:
                    if not getattr(chunk, "choices", None):
                        continue
                    delta = chunk.choices[0].delta
                    if getattr(delta, "content", None):
                        answer_content += delta.content
                return answer_content.strip()
            except Exception as e:
                last_ocr_error = str(e)
                print(f"[OCR] retrying call, error: {e}")
                raise

        try:
            text = _call_ocr_with_retry()
            rec_texts = [line.strip() for line in text.splitlines() if line.strip()]
            return [{
                "angle": 0,
                "rec_texts": rec_texts,
                "rec_boxes": [],
                "rec_scores": [],
            }]
        except Exception:
            print(f"[OCR] failed after 3 attempts, last error: {last_ocr_error}")
            return [{
                "angle": 0,
                "rec_texts": [],
                "rec_boxes": [],
                "rec_scores": [],
                "error": last_ocr_error or "unknown error",
            }]


class LocalEasyOCR:
    """
    A wrapper class for EasyOCR that provides a standardized interface
    for OCR operations with consistent output format.
    """

    def __init__(
        self,
        languages: List[str] = ['en'],
        gpu: bool = False,
        model_storage_directory: str = None,
        download_enabled: bool = True
    ):
        """
        Initialize the EasyOCR reader.

        Args:
            languages: List of language codes to use for OCR (default: ['en'])
            gpu: Whether to use GPU acceleration (default: False)
            model_storage_directory: Custom path for model files (default: ~/.EasyOCR/model)
            download_enabled: Whether to download models if not found (default: True)
        """
        try:
            import easyocr as _easyocr
        except Exception as e:
            raise RuntimeError(
                f"easyocr import failed: {e}. "
                f"Install with: pip install easyocr"
            ) from e

        self.reader = _easyocr.Reader(
            languages,
            gpu=gpu,
            model_storage_directory=model_storage_directory,
            download_enabled=download_enabled
        )

    def call_ocr(
        self,
        pil_img: Image.Image,
        fmt: str = 'png'
    ) -> List[dict]:
        """
        Perform OCR on a PIL image and return results in standardized format.

        Args:
            pil_img: PIL Image object
            fmt: Image format (default: 'png') - currently not used but kept for compatibility

        Returns:
            List containing a single payload dictionary with keys:
            - angle: Rotation angle (always 0, EasyOCR doesn't provide this)
            - rec_texts: List of recognized text strings
            - rec_boxes: List of bounding box coordinates [x_min, y_min, x_max, y_max]
            - rec_scores: List of confidence scores for each detection
        """
        # Convert PIL Image to numpy array
        image_array = np.array(pil_img)

        # Perform OCR using EasyOCR
        results = self.reader.readtext(image_array, detail=1)

        # Extract components from results
        rec_boxes = []
        rec_texts = []
        rec_scores = []

        for result in results:
            # Format: (bbox, text, confidence)
            bbox, text, confidence = result

            # Convert bbox from 4 corner points to [x_min, y_min, x_max, y_max]
            # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

            rec_boxes.append(box)
            rec_texts.append(text)
            rec_scores.append(confidence)

        # Create payload
        payload = {
            "angle": 0,  # EasyOCR doesn't provide rotation angle
            "rec_texts": rec_texts,
            "rec_boxes": rec_boxes,
            "rec_scores": rec_scores
        }

        return [payload]
