import unittest

from langchain.evaluation import load_evaluator, EvaluatorType
from langchain_core.outputs import LLMResult

from src.main.lab import chat_model, built_in_depth_evaluator, your_custom_criteria, textInput, \
    depth_criteria_passing_query, custom_mathematical_evaluator
from src.utilities.llm_testing_util import llm_connection_check, llm_wakeup

"""
This file will contain test cases for the automatic evaluation of your
solution in main/lab.py. You should not modify the code in this file. You should
also manually test your solution by running app.py.
"""

class TestLLMResponse(unittest.TestCase):

    """
    This function is a sanity check for the Language Learning Model (LLM) connection.
    It attempts to generate a response from the LLM. If a 'Bad Gateway' error is encountered,
    it initiates the LLM wake-up process. This function is critical for ensuring the LLM is
    operational before running tests and should not be modified without understanding the
    implications.
    Raises:
        Exception: If any error other than 'Bad Gateway' is encountered, it is raised to the caller.
    """
    def test_llm_sanity_check(self):
        try:
            response = llm_connection_check()
            self.assertIsInstance(response, LLMResult)
        except Exception as e:
            if 'Bad Gateway' in str(e):
                llm_wakeup()
                self.fail("LLM is not awake. Please try again in 3-5 minutes.")

    """
    This test will ensure that your work for the prompt evaluated by the "depth" built-in evaluator works properly.
    """
    def test_built_in_depth_evaluator(self):

        evaluator = load_evaluator(EvaluatorType.CRITERIA, llm=chat_model, criteria="depth")

        prediction = chat_model.invoke(textInput.format(userInput=depth_criteria_passing_query))
        eval_result = evaluator.evaluate_strings(
            prediction=prediction.content,
            input=depth_criteria_passing_query
        )

        # print(eval_result)

        self.assertEqual(eval_result["value"], "Y")

    """
    This test will ensure that your work for the custom criteria and evaluator instantiation works properly.
    """
    def test_custom_mathematical_evaluator_works(self):

        query = "What is 5 * 5?"

        eval_result = custom_mathematical_evaluator(query)

        # print(eval_result)

        self.assertEqual(eval_result["value"], "Y")
