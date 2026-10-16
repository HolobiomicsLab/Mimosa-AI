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
{{"id": "<snake_case_id>", "label": "<short human name>", "rationale": "<1-2 sentences on why this kind of choice matters scientifically>", "chosen_option_id": "<snake_case>", "options": [{{"id": "<snake_case>", "label": "<short human name>", "description": "<what this option does in 1 sentence>"}}]}}

`options` MUST contain the option the agent actually used, and SHOULD also
list any alternative options the agent explicitly considered and rejected in
this step — each with its own `id`, `label`, and `description`. This is the
point of the export: reviewers need to see what was chosen *over what*. Only
include alternatives that appear in the step's reasoning or code; never invent
plausible-sounding options that were not actually weighed. `chosen_option_id`
MUST equal the `id` of exactly one entry in `options`.

ID rules (ASTRA-required): ^[a-z][a-z0-9_]*$ for `id`, `chosen_option_id`, and
every `options[*].id`.

Output ONLY the JSON object or the literal token `null`. No prose, no
markdown fences, no commentary.
