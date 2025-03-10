import anthropic
import base64
import httpx
import textwrap
import utils.config_log as config_log
import time  # 用來做延遲

config, logger, CONFIG_PATH = config_log.setup_config_and_logging()
config.read(CONFIG_PATH)

def template(prompt, image_bytes=None, image_media_type="image/png", max_retries=5):
    """
    Sends a text prompt and optionally an image to Claude 3.7 Sonnet API with exponential backoff.

    :param prompt: The text prompt to send.
    :param image_bytes: Optional image data in bytes.
    :param image_media_type: The media type of the image (default: "image/png").
    :param max_retries: Maximum number of retry attempts (default: 5).
    :return: Claude's response or "ERROR" on failure.
    """
    api_key = config.get('Claude', 'api_key')
    userprompt = textwrap.dedent(prompt).strip()

    client = anthropic.Anthropic(api_key=api_key)

    messages = [{"role": "user", "content": []}]

    if image_bytes:
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        messages[0]["content"].append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_media_type,
                "data": encoded_image
            }
        })

    messages[0]["content"].append({"type": "text", "text": userprompt})

    retry_count = 0
    backoff = 1  # 初始回退時間為 1 秒

    while retry_count < max_retries:
        try:
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=8192,
                messages=messages
            )
            return response.content[0].text  # 成功則回傳回應
        except (httpx.TimeoutException, anthropic.APIError) as e:
            logger.error(f"Claude API error (attempt {retry_count + 1}): {e}")
            retry_count += 1
            if retry_count >= max_retries:
                return "ERROR"
            time.sleep(backoff)
            backoff *= 2  # 指數回退 (1s, 2s, 4s, 8s, ...)

        except Exception as e:
            logger.error(f"Unexpected Claude API error: {e}")
            return "ERROR"

    return "ERROR"
