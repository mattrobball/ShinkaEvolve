import backoff
import openai
import re
from .pricing import GEMINI_MODELS
from .result import QueryResult
import logging

logger = logging.getLogger(__name__)


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.info(
            f"Gemini - Retry {details['tries']} due to error: {exc}. Waiting {details['wait']:0.1f}s..."
        )


@backoff.on_exception(
    backoff.expo,
    (
        openai.APIConnectionError,
        openai.APIStatusError,
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
    max_tries=20,
    max_value=20,
    on_backoff=backoff_handler,
)
def query_gemini(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query Gemini model."""
    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    if output_model is None:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                *new_msg_history,
            ],
            **kwargs,
        )
        try:
            text = response.choices[0].message.content
            # Handle None content (can happen with some Gemini models)
            if text is None:
                logger.warning(f"Gemini response content is None for model {model}")
                # Try to get from refusal field if available
                if hasattr(response.choices[0].message, 'refusal') and response.choices[0].message.refusal:
                    text = f"[REFUSAL] {response.choices[0].message.refusal}"
                else:
                    # Log the full response for debugging
                    logger.error(f"Full response object: {response}")
                    raise ValueError(f"Gemini returned None content. Model: {model}, Response: {response}")
        except Exception as e:
            logger.warning(f"Error accessing standard content field: {e}")
            # Reasoning models - ResponseOutputMessage
            try:
                text = response.output[1].content[0].text
            except Exception as e2:
                logger.error(f"Failed to access alternate response format: {e2}")
                logger.error(f"Full response: {response}")
                raise ValueError(f"Unable to extract content from Gemini response: {e2}")
        new_msg_history.append({"role": "assistant", "content": text})
    else:
        raise ValueError("Gemini does not support structured output.")

    # Safely handle potential None content for thought extraction
    message_content = response.choices[0].message.content or text or ""

    thought_match = re.search(
        r"<thought>(.*?)</thought>", message_content, re.DOTALL
    )

    thought = thought_match.group(1) if thought_match else ""

    content_match = re.search(
        r"<thought>(.*?)</thought>", message_content, re.DOTALL
    )
    if content_match:
        # Extract everything before and after the <thought> tag as content
        content = (
            message_content[: content_match.start()]
            + message_content[content_match.end() :]
        ).strip()
    else:
        content = message_content

    input_cost = GEMINI_MODELS[model]["input_price"] * response.usage.prompt_tokens
    output_tokens = response.usage.total_tokens - response.usage.prompt_tokens
    output_cost = GEMINI_MODELS[model]["output_price"] * output_tokens
    result = QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=output_tokens,
        cost=input_cost + output_cost,
        input_cost=input_cost,
        output_cost=output_cost,
        thought=thought,
        model_posteriors=model_posteriors,
    )
    return result
