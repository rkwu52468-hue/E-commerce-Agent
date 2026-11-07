"""Conservative context-overflow classification; error bodies are never logged."""
import re

_OVERFLOW_CODES = frozenset({'prompt_too_long', 'context_length_exceeded'})
_GENERIC_REQUEST_CODES = frozenset({'invalid_request_error', 'invalid_request', 'request_too_large', 'invalid_parameters'})
_OVERFLOW_TEXT = re.compile(
    r'\b(?:prompt_too_long|context_length_exceeded)\b|'
    r'\bprompt (?:is )?too long\b|'
    r'\bmaximum context length\b|\bcontext (?:length|window)(?: is)? exceeded\b',
    re.IGNORECASE,
)


_NON_CONTEXT_TEXT = re.compile(
    r'\b(?:rate[ _-]?limit(?:ed)?|tpm|tps|rpm|quota|billing|insufficient balance|'
    r'tokens?[ _-]+per[ _-]+(?:minute|second|day)|invalid[ _-]+api[ _-]+key|authentication)\b',
    re.IGNORECASE,
)


def is_context_overflow(error):
    """Only classify the original model exception, not customer/tool text.

    Prefer SDK status/body fields. The short-message fallback also supports the
    original course's synthetic errors, without treating every HTTP 400 as size.
    """
    try:
        if (isinstance(error, (TimeoutError, ConnectionError))
                or type(error).__name__ in ('APIConnectionError', 'APITimeoutError')):
            return False
        status = getattr(error, 'status_code', None)
        if status is not None and status not in (400, 413, 422):
            return False
        body = getattr(error, 'body', None)
        detail = body.get('error', body) if isinstance(body, dict) else None
        codes = []
        if isinstance(detail, dict):
            codes = [detail.get(name) for name in ('type', 'code') if isinstance(detail.get(name), str)]
            codes = [value.lower() for value in codes]
            if any(value not in _OVERFLOW_CODES | _GENERIC_REQUEST_CODES for value in codes):
                return False
            if any(value in _OVERFLOW_CODES for value in codes):
                return True
            message = detail.get('message')
        else:
            message = detail if isinstance(detail, str) else None
        if not isinstance(message, str):
            message = str(error)
        message = message[:8192]
        # A generic token complaint can mean rate/quota rather than context.
        # Keep the exact original teaching signal, but reject ambiguous prose.
        if _NON_CONTEXT_TEXT.search(message):
            return False
        if message.strip().lower().rstrip('.!') == 'too many tokens':
            return True
        return bool(_OVERFLOW_TEXT.search(message))
    except Exception:
        return False
