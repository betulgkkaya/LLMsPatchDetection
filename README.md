# Leveraging Large Language Models for Security Patch Detection and CWE Classification

This repository contains the implementation and prompt templates described in the paper:

**"Leveraging Large Language Models for Security Patch Detection and CWE Classification"**  
*Betul Gokkaya*  
Presented at the *2025 7th International Congress on Human-Computer Interaction, Optimization and Robotic Applications (ICHORA)*


---

## 🔍 Overview

This work evaluates the ability of modern Large Language Models (LLMs) to:
1. **Detect security patches** in code changes using code-only context (i.e., without commit messages),
2. **Classify** those security patches into **Common Weakness Enumeration (CWE)** categories.

### LLMs Evaluated
- GPT-4o
- GPT-4o-mini
- GPT-3.5
- Claude 3 Haiku
- Claude 3.5 Haiku
- DeepSeek V3

---

## 📁 Directory Structure

```bash
Prompts/
├── claude_prompts/          # Claude-specific prompt and API setup
│   └── claude_api_call/     # Claude API interaction code
├── deepseek_prompts/        # DeepSeek prompt templates and scripts
├── openai_prompts/          # OpenAI (GPT-3.5, GPT-4o) prompt templates and code

