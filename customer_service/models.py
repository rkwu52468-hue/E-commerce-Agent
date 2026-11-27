"""Live adapter is loaded only by explicit --live; offline replies are scripted."""
from .safeguards import call_with_transient_retry


class ScriptedModel:
    mode = 'scripted-offline'

    def __init__(self, replies):
        self.replies = iter(replies)
        self.steps = 0

    def complete(self, messages, tools, system):
        self.steps += 1
        reply = next(self.replies)
        if callable(reply):
            reply = reply(messages)
        return {'content': reply, 'stop_reason': 'tool_use' if any(b.get('type') == 'tool_use' for b in reply) else 'end_turn', 'usage': {}}


class AnthropicModel:
    mode = 'live-deepseek'

    def __init__(self, api_key, config, max_retries=0, retry_base_delay=0.5,
                 retry_max_delay=4.0, sleep_fn=None, client=None):
        if client is not None:
            self.client = client
        else:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key, base_url=config['base_url'],
                                    max_retries=0, timeout=60)
        self.model = config['model']
        self.max_retries = max(0, int(max_retries))
        self.retry_base_delay = float(retry_base_delay)
        self.retry_max_delay = float(retry_max_delay)
        self._sleep_fn = sleep_fn
        self.last_retry_count = 0

    def complete(self, messages, tools, system):
        retries = {'count': 0}

        def record_retry(attempt, delay, error):
            retries['count'] = attempt

        def invoke():
            return self.client.messages.create(model=self.model, messages=messages,
                                               tools=tools, system=system, max_tokens=2048)

        response = call_with_transient_retry(
            invoke,
            max_retries=self.max_retries,
            base_delay=self.retry_base_delay,
            max_delay=self.retry_max_delay,
            sleep_fn=self._sleep_fn,
            on_retry=record_retry,
        )
        self.last_retry_count = retries['count']
        # Keep the complete assistant array, including any thinking blocks.
        return {'content': [block.model_dump(mode='json', exclude_none=True) for block in response.content],
                'stop_reason': response.stop_reason,
                'usage': response.usage.model_dump(mode='json', exclude_none=True)}


    def complete_scheduled(self, messages, tools, system):
        """One bounded explanation of already-read facts; no autonomous tools."""
        if tools:
            raise ValueError('Scheduled explanations cannot expose tools')
        response = self.client.messages.create(model=self.model, messages=messages,
                    system=system, max_tokens=1024,
                    extra_body={'thinking': {'type': 'disabled'}})
        return {'content': [block.model_dump(mode='json', exclude_none=True) for block in response.content],
                'stop_reason': response.stop_reason,
                'usage': response.usage.model_dump(mode='json', exclude_none=True)}


    def complete_teammate(self, messages, tools, system):
        """Bounded s13 provider call; same hidden process credential, no retries."""
        response = self.client.messages.create(model=self.model, messages=messages,
                    tools=tools, system=system, max_tokens=1024, timeout=20,
                    extra_body={'thinking': {'type': 'disabled'}})
        return {'content': [block.model_dump(mode='json', exclude_none=True) for block in response.content],
                'stop_reason': response.stop_reason,
                'usage': response.usage.model_dump(mode='json', exclude_none=True)}
