#!/usr/bin/env python3
"""Wrap the model publisher's native template with our fast default.

No model files are modified. Vision and tool syntax remain publisher-owned.
"""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--model', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
config = json.loads((a.model / 'tokenizer_config.json').read_text())
template_path = a.model / 'chat_template.jinja'
tool_template = a.model / 'chat_templates/tool_use.jinja'
default_template = a.model / 'chat_templates/default.jinja'
if tool_template.is_file():
    template = tool_template.read_text()
elif template_path.is_file():
    template = template_path.read_text()
elif default_template.is_file():
    template = default_template.read_text()
else:
    template = config.get('chat_template')
if isinstance(template, list):
    templates = {t['name']: t['template'] for t in template}
    template = templates.get('tool_use', templates.get('default'))
if not isinstance(template, str) or not template.strip():
    raise SystemExit('Publisher chat_template unavailable; refusing to invent vision/tool syntax')
prefix = "{# ArciTech fast default; native Hcompany/Qwen template follows. #}\n"
prefix += "{%- set enable_thinking = enable_thinking | default(false) -%}\n"
prefix += "{%- set reasoning_effort = reasoning_effort | default('low') -%}\n"
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(prefix + template)
print(json.dumps({'template': str(a.output), 'publisher_sha256': hashlib.sha256(template.encode()).hexdigest(),
                  'default_thinking': False, 'default_effort': 'low'}))
