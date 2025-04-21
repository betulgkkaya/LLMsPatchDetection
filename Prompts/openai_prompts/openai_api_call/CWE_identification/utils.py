# utils.py - PatchDB-Specific Prompt Templates

# **System Instruction**
SYS_INST = "You are a security expert specializing in static program analysis and CWE (Common Weakness Enumeration) identification. Your task is to analyze code patches and determine the most relevant CWE ID based on the provided security patch."




# **Zero-Shot Learning Example**
ZERO_SHOT_PROMPT = """Analyze the following security patch and identify the most relevant CWE ID:


```
{diff_code}
```

Respond with only the CWE ID in the following format:

<CWE-ID>


Do not provide explanations, additional details, or any other text beyond the CWE ID.
"""

# **Controlled Choice CWE Identification Prompt**
CONTROLLED_CHOICE_PROMPT = """Analyze the following security patch and identify the most relevant CWE ID from the predefined list:

**Potential CWE IDs:**
416, 190, 476, 189, 264, 125, 200, 399, 362, 79, 787, 89, 352, 22, 78, 862, 434

**Security Patch:**
```
{diff_code}
```

Respond with only the CWE ID in the following format:

<CWE-ID>


Do not provide explanations, additional details, or any other text beyond the CWE ID.
"""




# **One-Shot Learning Example**
ONESHOT_USER = """
"""

ONESHOT_ASSISTANT = ""

# **Two-Shot Learning Example**
TWOSHOT_USER = """
"""

TWOSHOT_ASSISTANT = ""

# **Chain of Thought Prompt**
CoT_PROMPT_TEMPLATE = """

"""

# **Error Analysis Prompt**
ERROR_ANALYSIS_PROMPT = """GPT-3.5 classified the following patch as `{incorrect_prediction}`, but the ground truth indicates `{correct_answer}`.

**Patch:**
```
{diff_code}
```

Your task is to analyze why GPT-3.5 might have made this mistake. Identify possible misinterpretations, missing context, or reasons why the model's classification was incorrect.

Please provide a clear and concise justification and explain why the correct answer is more accurate.
"""

# **Semantic Code Prompt**
SEMANTIC_CODE_PROMPT = """
"""

