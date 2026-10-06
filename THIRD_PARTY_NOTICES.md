# Third-party notices

PersonTTS original code is licensed under Apache-2.0. The third-party code
identified below retains its original license and copyright notices.
Complete upstream license texts are included in `third_party/licenses/`.

## Included source files

| Local path | Upstream source at the reviewed commit | License and local changes |
| --- | --- | --- |
| `eval/sal/utils/math.py` | [Search and Learn](https://github.com/huggingface/search-and-learn/blob/04f78ed7902bb5844a9fd7d265a3f58254434a11/src/sal/utils/math.py) | Apache-2.0; unchanged from the reviewed upstream file. |
| `eval/sal/utils/qwen_math_parser.py` | [Search and Learn](https://github.com/huggingface/search-and-learn/blob/04f78ed7902bb5844a9fd7d265a3f58254434a11/src/sal/utils/qwen_math_parser.py) | Apache-2.0; the Chinese answer marker is represented using Unicode code points, and its extraction uses a single split. Upstream Qwen and grader provenance comments are retained. |
| `eval/evaluation/grader.py` | [Qwen2.5-Math evaluation](https://github.com/QwenLM/Qwen2.5-Math/blob/a45202bd16f1ec06f433442dc1152d0074773465/evaluation/grader.py) | MIT; unchanged from the reviewed upstream file. |

The two Search and Learn files retain the copyright notice for the 2024
HuggingFace Inc. team. Qwen2.5-Math evaluation is copyright 2024 Zhibin Gou.

The parser and grader also identify Hendrycks MATH, ProphetNet / CRITIC,
PRM800K, ToRA, and DeepSeek-Math as sources of their evaluation logic.
Their license texts and copyright notices are preserved below. These links
record the upstream revisions reviewed for this release; they do not claim
to identify the historical commit at which each inherited fragment was copied.

## Included license texts

| Upstream project | License | Complete local text | Reviewed upstream text |
| --- | --- | --- | --- |
| search-and-learn | Apache-2.0 | [`search-and-learn-APACHE-2.0.txt`](third_party/licenses/search-and-learn-APACHE-2.0.txt) | [License at `04f78ed7902b`](https://github.com/huggingface/search-and-learn/blob/04f78ed7902bb5844a9fd7d265a3f58254434a11/LICENSE) |
| Qwen2.5-Math evaluation | MIT | [`qwen2.5-math-evaluation-MIT.txt`](third_party/licenses/qwen2.5-math-evaluation-MIT.txt) | [License at `a45202bd16f1`](https://github.com/QwenLM/Qwen2.5-Math/blob/a45202bd16f1ec06f433442dc1152d0074773465/evaluation/LICENSE) |
| Hendrycks MATH | MIT | [`hendrycks-math-MIT.txt`](third_party/licenses/hendrycks-math-MIT.txt) | [License at `985bdc1696e8`](https://github.com/hendrycks/math/blob/985bdc1696e88e8643f081a0ff4719da39f2ae2a/LICENSE) |
| PRM800K | MIT | [`prm800k-MIT.txt`](third_party/licenses/prm800k-MIT.txt) | [License at `7ecc794703b2`](https://github.com/openai/prm800k/blob/7ecc794703b2877f63226f2477a49b34f9b25163/LICENSE) |
| ProphetNet / CRITIC | MIT | [`prophetnet-critic-MIT.txt`](third_party/licenses/prophetnet-critic-MIT.txt) | [License at `5cf70eb41cda`](https://github.com/microsoft/ProphetNet/blob/5cf70eb41cdaa1d8faa3e1265d95ee5792d49a53/LICENSE) |
| ToRA | MIT | [`tora-MIT.txt`](third_party/licenses/tora-MIT.txt) | [License at `213c1c995038`](https://github.com/microsoft/ToRA/blob/213c1c995038c73fab10343814df7a42f990f026/LICENSE) |
| DeepSeek-Math code | MIT | [`deepseek-math-MIT.txt`](third_party/licenses/deepseek-math-MIT.txt) | [License at `b8b0f8ce093d`](https://github.com/deepseek-ai/DeepSeek-Math/blob/b8b0f8ce093d80bf8e9a641e44142f06d092c305/LICENSE-CODE) |

## Installed dependencies

`requirements.txt` references separately distributed packages. They retain
their own licenses: SymPy and NumPy (BSD-3-Clause with their included component
notices), latex2sympy2 and word2number (MIT), ANTLR Python runtime (BSD-3-Clause),
and regex 2025.9.18 (Apache-2.0 AND CNRI-Python).

Claude Agent SDK Python source is MIT-licensed. Its bundled Claude Code CLI
and service use are subject to the applicable Anthropic terms described in
[the SDK 0.1.58 README](https://github.com/anthropics/claude-agent-sdk-python/blob/v0.1.58/README.md#license-and-terms).
These dependency packages and the CLI are not included in this source repository.

## Data

The project code license does not relicense third-party benchmark questions
or other third-party data. Their existing rights and applicable terms remain
with their respective owners.
