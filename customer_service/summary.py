"""Observed SDK boundary for opt-in history summaries; no retry or business tools."""


class SummaryError(RuntimeError):
    def __init__(self, reason, error_type=None):
        self.reason = reason
        self.error_type = error_type
        super().__init__(reason)


def _usage(response):
    try:
        value = getattr(response, 'usage', None)
        if hasattr(value, 'model_dump'):
            value = value.model_dump(mode='json', exclude_none=True)
        return dict(value) if isinstance(value, dict) and value else None
    except Exception:
        return None


def _usage_missing(usage):
    if usage is None:
        return True
    required = ('input_tokens', 'output_tokens')
    optional = ('cache_read_input_tokens', 'cache_creation_input_tokens')
    for name in (*required, *(name for name in optional if name in usage)):
        value = usage.get(name)
        if type(value) is not int or value < 0:
            return True
    return False


class SummaryClient:
    """Client shape expected by the unmodified original summarize_history method.

    Raw client credentials and request bodies are never copied into event metadata.
    The live raw client is already configured with max_retries=0 by AnthropicModel.
    """
    def __init__(self, raw_client, model_name, mode='scripted-offline'):
        self.raw_client = raw_client
        self.model = model_name
        self.mode = mode
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        if kwargs.get('max_tokens') != 2000 or 'tools' in kwargs:
            raise SummaryError('invalid_summary_request')
        record = {'status': 'attempted', 'usage': None, 'usage_missing': True,
                  'max_tokens': 2000, 'summary_mode': self.mode}
        self.calls.append(record)
        try:
            response = self.raw_client.messages.create(**kwargs)
        except Exception as error:
            record.update(status='failed', reason='summary_request_failed',
                          error_type=type(error).__name__)
            raise SummaryError('summary_request_failed', type(error).__name__) from None
        record['usage'] = _usage(response)
        record['usage_missing'] = _usage_missing(record['usage'])
        stop_reason = getattr(response, 'stop_reason', None)
        record['stop_reason'] = stop_reason
        content = getattr(response, 'content', None)
        if (stop_reason != 'end_turn' or not isinstance(content, (list, tuple))
                or any(getattr(block, 'type', None) == 'tool_use' for block in content)):
            record.update(status='failed', reason='summary_incomplete')
            raise SummaryError('summary_incomplete')
        text = '\n'.join(getattr(block, 'text', '') for block in content
                         if getattr(block, 'type', None) == 'text').strip()
        if not text:
            record.update(status='failed', reason='summary_empty')
            raise SummaryError('summary_empty')
        record['status'] = 'completed'
        return response
