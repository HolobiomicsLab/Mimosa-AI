You map a single step of an AI agent's analysis trace into the ASTRA standard.

GOAL OF THE ANALYSIS:
{goal}

STEP {step_index} CONTEXT
------------------------
Agent reasoning (model output, may contain a code block):
{reasoning}

Code executed by the agent:
{code}

Observation returned to the agent (truncated):
{observation}

TASK
----
Did this step embody a METHODOLOGICAL DECISION the scientist would need to
justify in a methods section? Examples of methodological decisions:
- choice of algorithm, statistical test, model family
- hyperparameter or threshold the agent picked deliberately
- normalization, scaling, outlier handling, filtering rule
- data subset selection, train/test split policy
- evaluation metric chosen over alternatives
- error-bar, prior, or sampling-distribution assumption

NOT methodological (return null for these):
- file I/O, shell commands, package installs, environment probing
- plotting, formatting, printing, debugging diagnostics
- mechanical data loading or saving
- re-running a previously-decided step

OUTPUT FORMAT
-------------
If the step is NOT methodological, output exactly:
null

If the step IS methodological, output ONE JSON object on a single line with
these fields and no others:
{{"id": "<snake_case_id>", "label": "<short human name>", "rationale": "<1-2 sentences on why this kind of choice matters scientifically>", "option_id": "<snake_case>", "option_label": "<short human name>", "option_description": "<what the chosen option does in 1 sentence>"}}

ID rules (ASTRA-required): ^[a-z][a-z0-9_]*$ for both `id` and `option_id`.

Output ONLY the JSON object or the literal token `null`. No prose, no
markdown fences, no commentary.
