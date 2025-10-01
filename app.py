import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fal_client
import numpy as np
import requests
import streamlit as st
from openai import OpenAI
from PIL import Image
from streamlit_drawable_canvas import st_canvas


st.set_page_config(
    page_title="FashionAI",
    page_icon="👗",
    layout="wide"
)


DEFAULT_PROMPT = (
    "Based on these images, create a 2D technical flat sketch of the garment, showing both the front "
    "and back views. Exclude textures and the human body — use clean linework only."
)

DEFAULT_PROMPT_KR = (
    "이 이미지들을 기반으로 의상의 앞면과 뒷면을 모두 보여주는 2D 테크니컬 플랫 스케치를 만들어주세요. "
    "텍스처와 인체는 제외하고, 깔끔한 선 작업만 사용하세요."
)

PROMPT_PATH = Path(__file__).parent / "prompts" / "edit_prompt.txt"
PROMPT_KR_PATH = Path(__file__).parent / "prompts" / "edit_prompt_kr.txt"
PROMPTS_DIR = PROMPT_PATH.parent
INPAINT_PROMPT_KR_PATH = PROMPTS_DIR / "refine_prompt_kr.txt"
INPAINT_PROMPT_EN_PATH = PROMPTS_DIR / "refine_prompt_en.txt"
REGENERATE_PROMPT_KR_PATH = PROMPTS_DIR / "regenerate_prompt_kr.txt"
REGENERATE_PROMPT_EN_PATH = PROMPTS_DIR / "regenerate_prompt_en.txt"

MODEL_ENDPOINTS = [
    {
        "label": "Seedream v4 Edit",
        "endpoint": "fal-ai/bytedance/seedream/v4/edit",
    },
    {
        "label": "Flux Pro Kontext Max Multi",
        "endpoint": "fal-ai/flux-pro/kontext/max/multi",
    },
    {
        "label": "Qwen Image Edit Plus",
        "endpoint": "fal-ai/qwen-image-edit-plus",
    },
    {
        "label": "Gemini 2.5 Flash Image Edit",
        "endpoint": "fal-ai/gemini-25-flash-image/edit",
    },
]


def ensure_prompt_file() -> None:
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    if not PROMPT_PATH.exists():
        PROMPT_PATH.write_text(DEFAULT_PROMPT, encoding="utf-8")
    if not PROMPT_KR_PATH.exists():
        PROMPT_KR_PATH.write_text(DEFAULT_PROMPT_KR, encoding="utf-8")


def load_prompt_text() -> str:
    ensure_prompt_file()
    try:
        return PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return DEFAULT_PROMPT


def save_prompt_text(text: str) -> None:
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    PROMPT_PATH.write_text(text.strip(), encoding="utf-8")


def load_prompt_kr_text() -> str:
    ensure_prompt_file()
    try:
        return PROMPT_KR_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def save_prompt_kr_text(text: str) -> None:
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    PROMPT_KR_PATH.write_text(text.strip(), encoding="utf-8")


def load_text_file(path: Path, default: str = "") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return default


def save_text_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip(), encoding="utf-8")


def extract_url(upload_result: Any) -> str:
    if isinstance(upload_result, str):
        return upload_result
    if isinstance(upload_result, dict):
        return upload_result.get("url", "")
    return ""


def upload_images_to_fal(files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[str]:
    urls: List[str] = []
    for file in files:
        suffix = Path(file.name).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.getbuffer())
            tmp_path = tmp.name
        try:
            upload_result = fal_client.upload_file(tmp_path)
            url = extract_url(upload_result)
            if url:
                urls.append(url)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return urls


def upload_bytes_to_fal(data: bytes, suffix: str = ".png") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        upload_result = fal_client.upload_file(tmp_path)
        url = extract_url(upload_result)
        if not url:
            raise RuntimeError("Failed to obtain upload URL from fal.ai response.")
        return url
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def normalize_images(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    images = data.get("images") or []
    if images and isinstance(images[0], list):
        images = images[0]
    return images


def set_selected_image(label: str, url: str) -> None:
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    image_bytes = response.content
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_array = np.array(image)
    st.session_state.selected_image = {
        "label": label,
        "url": url,
        "image_bytes": image_bytes,
        "image_width": image.width,
        "image_height": image.height,
        "image_array": image_array,
    }

    st.session_state.selected_image_cache = st.session_state.selected_image


def get_selected_image() -> Optional[Dict[str, Any]]:
    selected = st.session_state.get("selected_image")
    if selected:
        st.session_state.selected_image_cache = selected
        return selected

    cached = st.session_state.get("selected_image_cache")
    if cached:
        st.session_state.selected_image = cached
        return cached
    return None


MAX_CANVAS_DIM = 720


def _ensure_uint8(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if np.issubdtype(arr.dtype, np.floating):
        max_val = float(np.nanmax(arr)) if arr.size else 0.0
        if max_val <= 1.0 + 1e-3:
            arr = np.clip(arr * 255.0, 0, 255)
        else:
            arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)

def prepare_canvas_image(image: Image.Image, max_dim: int = MAX_CANVAS_DIM) -> Tuple[Image.Image, float]:
    width, height = image.size
    if width <= max_dim and height <= max_dim:
        return image.copy(), 1.0

    scale = min(max_dim / width, max_dim / height)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.BILINEAR), scale


def build_mask_from_canvas(
    canvas_data: np.ndarray,
    display_reference: np.ndarray,
    target_size: Tuple[int, int],
    alpha_threshold: int = 24,
    red_margin: int = 35,
) -> Image.Image:
    canvas_uint8 = _ensure_uint8(canvas_data)
    reference_uint8 = _ensure_uint8(display_reference)

    if canvas_uint8.shape[:2] != reference_uint8.shape[:2]:
        raise ValueError("Mask canvas size does not match the reference image size.")

    red_channel = canvas_uint8[..., 0].astype(np.int16)
    other_channels = np.maximum(canvas_uint8[..., 1], canvas_uint8[..., 2]).astype(np.int16)
    red_dominance = (red_channel - other_channels) > red_margin

    if canvas_uint8.shape[-1] == 4:
        alpha_mask = canvas_uint8[..., 3] > alpha_threshold
        drawn_mask = red_dominance & alpha_mask
    else:
        drawn_mask = red_dominance

    mask = drawn_mask.astype(np.uint8) * 255
    mask_image = Image.fromarray(mask, mode="L")
    if mask_image.size != target_size:
        mask_image = mask_image.resize(target_size, Image.NEAREST)
    return mask_image


def create_overlay_image(
    base_image: Image.Image,
    mask_image: Image.Image,
    overlay_color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.6,
) -> Image.Image:
    base_rgb = np.asarray(base_image.convert("RGB"), dtype=np.float32)
    mask_array = np.asarray(mask_image.convert("L")) > 0
    if not mask_array.any():
        return base_image.convert("RGB")

    alpha_clamped = max(0.0, min(1.0, alpha))
    overlay = np.array(overlay_color, dtype=np.float32)
    base_rgb[mask_array] = base_rgb[mask_array] * (1.0 - alpha_clamped) + overlay * alpha_clamped
    return Image.fromarray(base_rgb.astype(np.uint8), mode="RGB")


def call_model(
    model: Dict[str, str],
    prompt: str,
    image_urls: List[str],
    mask_url: Optional[str] = None,
) -> Dict[str, Any]:
    return fal_client.run(
        model["endpoint"],
        arguments={
            "prompt": prompt,
            "image_urls": image_urls,
            "num_images": 1,
        },
    )


def call_model_inpaint(
    endpoint: str,
    prompt: str,
    image_url: str,
    mask_url: str,
) -> Dict[str, Any]:
    return fal_client.run(
        endpoint,
        arguments={
            "prompt": prompt,
            "image_url": image_url,
            "mask_url": mask_url,
            "num_images": 1,
        },
    )


def run_all_models(
    prompt: str,
    image_urls: List[str],
    mask_url: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    import logging
    logger = logging.getLogger(__name__)

    results: Dict[str, Dict[str, Any]] = {model["label"]: {} for model in MODEL_ENDPOINTS}

    # 로깅을 리스트에 수집 (thread-safe)
    logs = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_model = {}
        for model in MODEL_ENDPOINTS:
            label = model['label']
            endpoint = model['endpoint']
            logs.append(f"🔄 Submitting {label} to {endpoint}")
            logger.info(f"Submitting {label} to {endpoint}")
            future = executor.submit(call_model, model, prompt, image_urls, mask_url)
            future_to_model[future] = model

        for future in as_completed(future_to_model):
            model = future_to_model[future]
            label = model["label"]
            try:
                result = future.result(timeout=180)  # 3분 타임아웃
                logs.append(f"✅ {label} completed")
                logger.info(f"{label} completed")
                results[label] = result
            except Exception as exc:
                error_msg = f"❌ {label} failed: {exc}"
                logs.append(error_msg)
                logger.error(error_msg)
                results[label] = {"error": str(exc)}

    # with 블록 밖에서 안전하게 출력
    for log in logs:
        st.write(log)

    return results


def render_model_outputs(results: Dict[str, Dict[str, Any]]) -> None:
    selected = get_selected_image()
    for label, data in results.items():
        with st.container(border=True):
            st.subheader(label)
            if "error" in data:
                st.error(f"Failed to generate image: {data['error']}")
                continue

            images = normalize_images(data)
            if not images:
                st.warning("No images returned.")
            else:
                cols = st.columns(min(len(images), 3))
                for idx, (image, col) in enumerate(zip(images, cols)):
                    url = image.get("url") if isinstance(image, dict) else None
                    if url:
                        with col:
                            st.image(url, caption=f"Result {idx + 1}", use_column_width=True)
                            st.caption(f"[Download image {idx + 1}]({url})")
                            if selected and selected.get("url") == url:
                                st.success("Selected for refinement")
                            select_key = f"select_{label}_{idx}"
                            if st.button("Select for refinement", key=select_key):
                                try:
                                    set_selected_image(label, url)
                                except Exception as exc:
                                    st.error(f"Failed to prepare image: {exc}")
                                else:
                                    st.success("Image stored. Go to the Image Refinement tab.")

            description = data.get("description")
            if description:
                st.info(description)


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable `OPENAI_API_KEY` is not set.")
    return OpenAI(api_key=api_key)


def translate_korean_prompt(korean_prompt: str) -> str:
    if not korean_prompt.strip():
        raise ValueError("Korean prompt is empty.")

    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": (
                    "Translate the following Korean garment sketch prompt into English, keeping it concise and "
                    "technical for AI image editing. Provide only the translated prompt.\n\n" + korean_prompt
                ),
            }
        ],
    )
    translated = (response.choices[0].message.content or "").strip()
    if not translated:
        raise RuntimeError("Received empty translation from OpenAI.")
    return translated


def render_image_edit_page() -> None:
    st.title("👗 FashionAI Image Edit")
    st.write(
        "Upload reference images and generate flat sketch variations using multiple fal.ai models."
    )

    fal_key = os.getenv("FAL_KEY")
    if not fal_key:
        st.error("Environment variable `FAL_KEY` is not set. Please configure your fal.ai API key.")
        return

    if "prompt_text_en" not in st.session_state:
        st.session_state.prompt_text_en = load_prompt_text()
    if "prompt_text_kr" not in st.session_state:
        st.session_state.prompt_text_kr = load_prompt_kr_text()
    if "selected_image" not in st.session_state:
        st.session_state.selected_image = None
    if "selected_image_cache" not in st.session_state:
        st.session_state.selected_image_cache = None
    if "latest_results" not in st.session_state:
        st.session_state.latest_results = None

    with st.expander("Prompt settings", expanded=True):
        st.caption("`prompts/edit_prompt.txt`")
        st.text_area(
            "Current English prompt",
            value=st.session_state.prompt_text_en,
            height=150,
            disabled=True,
            key="prompt_text_en_viewer",
        )

        st.caption("`prompts/edit_prompt_kr.txt`")
        st.session_state.prompt_text_kr = st.text_area(
            "Korean prompt (editable)",
            value=st.session_state.prompt_text_kr,
            key="prompt_text_kr_editor",
            height=180,
        )

    col_save_kr, col_translate = st.columns([1, 1])
    with col_save_kr:
        if st.button("Save Korean prompt", type="secondary"):
            save_prompt_kr_text(st.session_state.prompt_text_kr)
            st.success("Saved Korean prompt to `edit_prompt_kr.txt`.")

    with col_translate:
        if st.button("Translate to English with GPT-5", type="primary"):
            try:
                save_prompt_kr_text(st.session_state.prompt_text_kr)
                with st.spinner("Translating via OpenAI..."):
                    translated = translate_korean_prompt(st.session_state.prompt_text_kr)
                save_prompt_text(translated)
                st.session_state.prompt_text_en = translated
                st.rerun()
            except Exception as exc:
                st.error(f"Translation failed: {exc}")

    prompt_text = st.session_state.prompt_text_en

    uploaded_files = st.file_uploader(
        "Upload garment reference images",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        help="You can upload multiple images for multi-reference editing.",
        key="image_edit_uploader",
    )

    files_to_use = uploaded_files

    if files_to_use:
        st.write("### Uploaded previews")
        preview_cols = st.columns(min(len(files_to_use), 3))
        for file, col in zip(files_to_use, preview_cols):
            with col:
                st.image(file, caption=file.name, use_column_width=True)

    generate_disabled = not files_to_use

    results_to_render = st.session_state.get("latest_results")

    if st.button("Generate designs", type="primary", disabled=generate_disabled):
        if not files_to_use:
            st.warning("Please upload at least one reference image.")
            return

        st.write("### 🎨 Generating with this prompt:")
        st.info(prompt_text)

        with st.spinner("Submitting to fal.ai models..."):
            image_urls = upload_images_to_fal(files_to_use)
            if not image_urls:
                st.error("Failed to upload images to fal.ai.")
                return

            results = run_all_models(prompt_text, image_urls)
            st.session_state.latest_results = results
            results_to_render = results

        st.success("Generation complete.")

    if results_to_render:
        render_model_outputs(results_to_render)



def render_image_refine_page() -> None:
    st.title("🛠️ Image Refinement")
    selected = get_selected_image()
    if not selected:
        st.info("Select an image from the Image Edit tab first.")
        return

    base_image = Image.open(BytesIO(selected["image_bytes"])).convert("RGB")
    st.image(base_image, caption=f"Selected from {selected['label']}", use_column_width=True)

    display_image, _ = prepare_canvas_image(base_image)
    display_reference_array = np.asarray(display_image)

    if "inpaint_prompt_kr" not in st.session_state:
        st.session_state.inpaint_prompt_kr = load_text_file(INPAINT_PROMPT_KR_PATH)
    if "inpaint_prompt_en" not in st.session_state:
        st.session_state.inpaint_prompt_en = load_text_file(INPAINT_PROMPT_EN_PATH)
    if "mask_canvas_seed" not in st.session_state:
        st.session_state.mask_canvas_seed = 0
    if "last_overlay_bytes" not in st.session_state:
        st.session_state.last_overlay_bytes = None
    if "latest_refine_results" not in st.session_state:
        st.session_state.latest_refine_results = None

    st.session_state.inpaint_prompt_kr = st.text_area(
        "Refinement prompt (Korean)",
        value=st.session_state.inpaint_prompt_kr,
        key="inpaint_prompt_kr_editor",
        height=120,
        help="Describe the desired edits in Korean. The prompt will be translated to English automatically.",
    )

    save_col, _ = st.columns([1, 5])
    with save_col:
        if st.button("한국어 프롬프트 저장", key="save_inpaint_prompt_button"):
            save_text_file(INPAINT_PROMPT_KR_PATH, st.session_state.inpaint_prompt_kr)
            st.success("인페인트 한국어 프롬프트를 저장했어요.")

    english_placeholder = st.empty()

    st.write("### Draw mask over areas to modify")
    canvas_key = f"mask_canvas_{st.session_state.mask_canvas_seed}"

    canvas_col, info_col = st.columns([4, 1])
    with canvas_col:
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=15,
            stroke_color="#FF0000",
            background_image=display_image,
            update_streamlit=True,
            height=display_image.height,
            width=display_image.width,
            drawing_mode="freedraw",
            key=canvas_key,
            display_toolbar=True,
        )

    with info_col:
        st.markdown(
            """
            - **Draw red** on regions to edit.
            - Use the eraser in the toolbar to correct mistakes.
            - The canvas scales the image to fit, and the mask will be resized back automatically.
            """
        )

        if st.button("Clear mask", type="secondary"):
            st.session_state.mask_canvas_seed += 1
            st.session_state.last_overlay_bytes = None
            st.rerun()

    overlay_bytes = st.session_state.get("last_overlay_bytes")
    if overlay_bytes:
        st.write("### Latest mask overlay preview")
        overlay_preview = Image.open(BytesIO(overlay_bytes)).convert("RGB")
        st.image(overlay_preview, caption="Mask overlay preview", use_column_width=True)

    results_to_render = st.session_state.get("latest_refine_results")

    if st.button("Translate & Refine", type="primary"):
        korean_prompt = st.session_state.inpaint_prompt_kr.strip()
        if not korean_prompt:
            st.warning("Please enter a Korean prompt before running the refinement.")
            return

        if canvas_result.image_data is None:
            st.warning("Draw a mask on the image before running the refinement.")
            return

        try:
            mask_image = build_mask_from_canvas(
                canvas_result.image_data,
                display_reference_array,
                base_image.size,
            )
        except Exception as exc:
            st.error(f"Failed to build mask: {exc}")
            return

        if not mask_image.getbbox():
            st.warning("The mask is empty. Please draw over the areas you want to change.")
            return

        try:
            with st.spinner("Translating prompt with GPT-5..."):
                english_prompt = translate_korean_prompt(korean_prompt)
        except Exception as exc:
            st.error(f"Translation failed: {exc}")
            return

        save_text_file(INPAINT_PROMPT_KR_PATH, korean_prompt)
        save_text_file(INPAINT_PROMPT_EN_PATH, english_prompt)
        st.session_state.inpaint_prompt_en = english_prompt

        overlay_image = create_overlay_image(base_image, mask_image)

        st.write("### 입력 이미지 확인")
        preview_col1, preview_col2, preview_col3 = st.columns(3)
        with preview_col1:
            st.image(base_image, caption="원본 이미지", use_column_width=True)
        with preview_col2:
            st.image(mask_image, caption="B&W 마스크 이미지", use_column_width=True)
        with preview_col3:
            st.image(overlay_image, caption="붉은 마스킹 오버레이", use_column_width=True)

        overlay_buffer = BytesIO()
        overlay_image.save(overlay_buffer, format="PNG")
        overlay_bytes = overlay_buffer.getvalue()

        mask_buffer = BytesIO()
        mask_image.save(mask_buffer, format="PNG")
        mask_bytes = mask_buffer.getvalue()

        base_buffer = BytesIO()
        base_image.save(base_buffer, format="PNG")
        base_bytes = base_buffer.getvalue()

        st.write("### 🎨 Refining with this prompt:")
        st.info(english_prompt)

        try:
            with st.spinner("Uploading images to fal.ai..."):
                overlay_url = upload_bytes_to_fal(overlay_bytes, suffix=".png")
                mask_url = upload_bytes_to_fal(mask_bytes, suffix=".png")
                base_url = upload_bytes_to_fal(base_bytes, suffix=".png")
        except Exception as exc:
            st.error(f"Failed to upload images: {exc}")
            return

        try:
            with st.spinner("Running fal.ai refinement models..."):
                import logging
                logger = logging.getLogger(__name__)

                # Image Refinement 전용: Flux와 Qwen은 inpaint API로 원본+마스크 사용
                results: Dict[str, Dict[str, Any]] = {}
                logs = []  # thread-safe 로그 수집

                with ThreadPoolExecutor(max_workers=4) as executor:
                    future_to_label = {}

                    for model in MODEL_ENDPOINTS:
                        label = model["label"]
                        if label == "Flux Pro Kontext Max Multi":
                            # Flux LoRA Inpainting API 사용
                            logs.append(f"🔄 Submitting {label} with inpaint API (base_url + mask_url)")
                            logger.info(f"Submitting {label} with inpaint API")
                            future = executor.submit(
                                call_model_inpaint,
                                "fal-ai/flux-lora/inpainting",
                                english_prompt,
                                base_url,
                                mask_url
                            )
                            future_to_label[future] = label
                        elif label == "Qwen Image Edit Plus":
                            # Qwen Inpaint API 사용
                            logs.append(f"🔄 Submitting {label} with inpaint API (base_url + mask_url)")
                            logger.info(f"Submitting {label} with inpaint API")
                            future = executor.submit(
                                call_model_inpaint,
                                "fal-ai/qwen-image-edit/inpaint",
                                english_prompt,
                                base_url,
                                mask_url
                            )
                            future_to_label[future] = label
                        else:
                            # 나머지는 오버레이 이미지 사용
                            logs.append(f"🔄 Submitting {label} with overlay image")
                            logger.info(f"Submitting {label} with overlay image")
                            future = executor.submit(call_model, model, english_prompt, [overlay_url])
                            future_to_label[future] = label

                    for future in as_completed(future_to_label):
                        label = future_to_label[future]
                        try:
                            result = future.result(timeout=180)  # 3분 타임아웃
                            logs.append(f"✅ {label} completed")
                            logger.info(f"{label} completed")
                            results[label] = result
                        except Exception as exc:
                            error_msg = f"❌ {label} failed: {exc}"
                            logs.append(error_msg)
                            logger.error(error_msg)
                            results[label] = {"error": str(exc)}

                # with 블록 밖에서 안전하게 출력
                for log in logs:
                    st.write(log)

        except Exception as exc:
            st.error(f"Generation failed: {exc}")
            return

        st.session_state.latest_refine_results = results
        st.session_state.last_overlay_bytes = overlay_bytes
        results_to_render = results
        st.success("Generation complete.")

    english_placeholder.text_area(
        "번역된 프롬프트 (영문)",
        value=st.session_state.get("inpaint_prompt_en", ""),
        height=120,
        disabled=True,
    )

    if results_to_render:
        render_model_outputs(results_to_render)


def render_prompt_regeneration_page() -> None:
    st.title("🔁 Prompt Regeneration (Korean)")
    selected = get_selected_image()
    if not selected:
        st.info("Select an image from the Image Edit tab first.")
        return

    image = Image.open(BytesIO(selected["image_bytes"])).convert("RGB")
    st.image(image, caption=f"Selected from {selected['label']}", use_column_width=True)

    if "regenerate_prompt_kr" not in st.session_state:
        st.session_state.regenerate_prompt_kr = load_text_file(REGENERATE_PROMPT_KR_PATH)
    if "regenerate_prompt_en" not in st.session_state:
        st.session_state.regenerate_prompt_en = load_text_file(REGENERATE_PROMPT_EN_PATH)
    if "latest_regenerate_results" not in st.session_state:
        st.session_state.latest_regenerate_results = None

    prompt_value = st.text_area(
        "Regeneration prompt (Korean)",
        value=st.session_state.regenerate_prompt_kr,
        key="regenerate_prompt_kr_editor",
        height=120,
        help="Describe how you want to modify the selected image in Korean. It will be translated to English automatically.",
    )
    st.session_state.regenerate_prompt_kr = prompt_value

    save_col, _ = st.columns([1, 5])
    with save_col:
        if st.button("한국어 프롬프트 저장", key="save_regenerate_prompt_button"):
            save_text_file(REGENERATE_PROMPT_KR_PATH, st.session_state.regenerate_prompt_kr)
            st.success("리제너레이션 한국어 프롬프트를 저장했어요.")

    english_placeholder = st.empty()

    results_to_render = st.session_state.get("latest_regenerate_results")

    if st.button("Translate & Regenerate", type="primary"):
        korean_prompt = st.session_state.regenerate_prompt_kr.strip()
        if not korean_prompt:
            st.warning("Please enter a Korean prompt before regenerating.")
            return

        try:
            with st.spinner("Translating prompt with GPT-5..."):
                english_prompt = translate_korean_prompt(korean_prompt)
        except Exception as exc:
            st.error(f"Translation failed: {exc}")
            return

        save_text_file(REGENERATE_PROMPT_KR_PATH, korean_prompt)
        save_text_file(REGENERATE_PROMPT_EN_PATH, english_prompt)
        st.session_state.regenerate_prompt_en = english_prompt

        st.write("### 🎨 Regenerating with this prompt:")
        st.info(english_prompt)

        try:
            with st.spinner("Uploading reference image to fal.ai..."):
                reference_url = upload_bytes_to_fal(selected["image_bytes"], suffix=".png")
        except Exception as exc:
            st.error(f"Failed to upload reference image: {exc}")
            return

        try:
            with st.spinner("Submitting to fal.ai models..."):
                results = run_all_models(english_prompt, [reference_url])
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
            return

        st.session_state.latest_regenerate_results = results
        results_to_render = results
        st.success("Generation complete.")

    english_placeholder.text_area(
        "번역된 프롬프트 (영문)",
        value=st.session_state.get("regenerate_prompt_en", ""),
        height=120,
        disabled=True,
    )

    if results_to_render:
        render_model_outputs(results_to_render)



def render_about_page() -> None:
    st.title("About FashionAI")
    st.markdown(
        """
        FashionAI generates technical flat sketches from reference images using
        multiple AI models powered by fal.ai.

        ## Features
        - **Image Edit**: Generate flat sketches from reference images
        - **Prompt Regeneration**: Modify selected images with new prompts
        - **Image Refinement**: Draw masks to edit specific regions

        ## How to use
        1. Configure your `FAL_KEY` and `OPENAI_API_KEY` environment variables
        2. Upload reference images in the Image Edit tab
        3. Generate designs and select your favorite result
        4. Refine or regenerate as needed

        ---
        **Created by Jedi Kim**
        """
    )


def main() -> None:
    menu = [
        "Image Edit",
        "Prompt Regeneration",
        "Image Refinement",
        "About",
    ]
    choice = st.sidebar.radio("Navigation", options=menu)

    if choice == "Image Edit":
        render_image_edit_page()
    elif choice == "Prompt Regeneration":
        render_prompt_regeneration_page()
    elif choice == "Image Refinement":
        render_image_refine_page()
    else:
        render_about_page()


if __name__ == "__main__":
    main()
