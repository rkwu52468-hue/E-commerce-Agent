"""Live adapter is loaded only by explicit --live; offline replies are scripted."""

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

    def __init__(self, api_key, config):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key, base_url=config['base_url'],
                                max_retries=0, timeout=60)
        self.model = config['model']

    def complete(self, messages, tools, system):
        response = self.client.messages.create(model=self.model, messages=messages,
                    tools=tools, system=system, max_tokens=2048)
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
