# Overview

### Evaluators

- In application development with LLMs, one of the most critical components is ensuring that 
the outcomes produced by your models are **reliable and useful across a broad array of inputs.**
  - Evaluation is critical when deploying LLM applications, since production environments require 
  **repeatable and useful outcomes that can be evaluated based on a given criteria.**

# Evaluation Lab

- In this lab, we will explore some **Built-In Evaluators**, which come with langchain. They include:
  - Relevance
  - Conciseness
  - Depth
  - Helpfulness
  - Harmfulness
  - Insensitivity
  - Plenty more that can be found in the documentation linked below.
- We will also create a **Custom Evaluator**, which will allow us to create our own criteria for response evaluation.

### Files to Modify:

- You will be directly modifying ```src/main/lab.py.```
  - Look for the "TODO" comments, which will specify the requirements for completing the lab. 
- You may modify ```src/app.py```, which contains sample code that should provide a valid output upon lab completion. Note that there is likely no reason modify app.py in this particular lab.
- DO NOT modify ```src/main/labtest.py```, as it contains the tests that you must pass to complete the lab.
  - You can consider your lab complete when every test passes.


### Notes & Resources

- This lab utilizes an LLM over an external connection, and can become inaccessible for various reasons, including an invalid API key. 
- Environment variables should be automatically configured for you upon opening the lab.

- [Langchain Evaluation](https://python.langchain.com/docs/guides/evaluation/)
- [Built-In and Custom Critera](https://python.langchain.com/docs/guides/evaluation/string/criteria_eval_chain)