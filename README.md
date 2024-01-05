# Overview

### Evaluators

- In application development with LLMs, it is critical to ensure that the outcomes produced by your models are **predictable and useful for a broad array of inputs.**
  - Thus, Evaluation can be beneficial when deploying LLM applications, since they can assess responses across a wide array of criteria.

- In this lab, we will explore Langchain's **String Evaluators**, which assess a predicted string for a given input, assessing it based on specific criteria or comparing it against a reference string.

# Evaluation Lab

### Objectives:

- In this lab, we will use an Evaluator with **Built-In Criteria**. They include:
  - Depth
  - Relevance
  - Conciseness
  - Helpfulness
  - Plenty more that can be found in the documentation linked below.
  
- We will also use an Evaluator with **Custom Criteria**, which allows us to create our own evaluation parameters.

### Files to Modify:

- You will be directly modifying ```src/main/lab.py.```
  - Look for the 3 "TODO" comments, which will specify the requirements for completing the lab. 
- You may modify ```src/app.py```, which contains sample code that should provide a valid output upon lab completion. Note that there is likely no reason modify app.py in this particular lab.
- DO NOT modify ```src/test/test_lab.py```, as it contains the tests that you must pass to complete the lab.
  - You can consider your lab complete when every test passes.


### Notes & Resources

- This lab utilizes an LLM over an external connection, and can become inaccessible for various reasons, including an invalid API key. 
- Environment variables should be automatically configured for you upon opening the lab.

- [Langchain Evaluation](https://python.langchain.com/docs/guides/evaluation/)
- [Built-In and Custom Critera](https://python.langchain.com/docs/guides/evaluation/string/criteria_eval_chain)
