import unittest

from langchain.evaluation import load_evaluator, EvaluatorType
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint

from src.main.lab import built_in_criteria, llm, chain, your_custom_criteria

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
        llm = HuggingFaceEndpoint(
            endpoint_url="https://z8dvl7fzhxxcybd8.eu-west-1.aws.endpoints.huggingface.cloud",
            huggingfacehub_api_token="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",
            task="text-generation",
            model_kwargs={
                "max_new_tokens": 1024
            }
        )

        self.assertIsInstance(llm, HuggingFaceEndpoint)

    """
    This test will ensure that your work for the prompt evaluated by the "depth" built-in evaluator works properly.
    """
    def test_built_in_evaluators(self):
        print(built_in_criteria["depth"][0])

        evaluator = load_evaluator(EvaluatorType.CRITERIA, llm=llm, criteria=built_in_criteria["depth"])

        prediction1 = chain.run(built_in_criteria["depth"])
        eval_result = evaluator.evaluate_strings(
            prediction=prediction1,
            input=built_in_criteria["depth"]
        )

        self.assertEquals(eval_result["value"], "Y")

    """
    This test will ensure that your work for the custom evaluator works properly.
    """
    def test_custom_evaluators(self):
        evaluator = load_evaluator(EvaluatorType.CRITERIA,criteria=your_custom_criteria,llm=llm)

        your_query = "What is 5 * 5?"
        your_prediction = llm.invoke(your_query)
        eval_result = evaluator.evaluate_strings(prediction=your_prediction, input=your_query)

        print("PROMPT : ", "NUMERIC")
        print("INPUT: ", your_query)
        print("OUTPUT:", your_prediction.split("What")[0].replace("\n",""))
        your_result = (eval_result["reasoning"].replace("\n", "").replace("[BEGIN RESPONSE]", "")[0])

        self.assertEqual(your_result, "Y")
