import unittest

from langchain.chat_models import AzureChatOpenAI

"""
This file will contain test cases for the automatic evaluation of your
solution in main/lab.py. You should not modify the code in this file. You should
also manually test your solution by running app.py.
"""

class TestLLMResponse(unittest.TestCase):
    """
    This test will verify that the connection to an external LLM is made. If it does not
    work, this may be because the API key is invalid, or the service may be down.
    If that is the case, this lab may not be completable.
    """
    def test_llm_sanity_check(self):
        llm = AzureChatOpenAI(model_name="gpt-35-turbo")

    """
    This test will ensure that the built-in evaluator is working properly.
    """
    def test_built_in_evaluator(self):
        "todo"

        """
    This test will ensure that the custom evaluator is working properly.
    """
    def test_built_in_evaluator(self):
        "todo"