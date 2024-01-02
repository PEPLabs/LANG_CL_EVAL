from langchain.chains import LLMChain
from langchain.evaluation import load_evaluator, EvaluatorType
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

"""
This lab will guide you through using built-in evaluators, and creating a custom evaluator

https://python.langchain.com/docs/guides/evaluation/string/criteria_eval_chain
"""

"""
Defining our LLM, templates, prompt and chain. No need to edit these.
"""
llm = HuggingFaceEndpoint(
    endpoint_url="https://z8dvl7fzhxxcybd8.eu-west-1.aws.endpoints.huggingface.cloud",
    huggingfacehub_api_token="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",
    task="text2text-generation",
    model_kwargs={
        "max_new_tokens": 200
    }
)

template = """You are an informative chatbot that gives answers to valid questions"""

system_message_template = SystemMessagePromptTemplate.from_template(template)
human_message_template = HumanMessagePromptTemplate.from_template("{input}")
chat_prompt = ChatPromptTemplate.from_messages([system_message_template, human_message_template])

chain = LLMChain(
    llm=llm,
    prompt=chat_prompt,
)

"""
Your tasks for lab completion are below: 
-You will first define two prompts that will be evaluated by the built-in "depth" evaluator
-Then, you will define your own custom criteria that will evaluate whether a response returns a mathematical value
"""

# TODO: Add one prompt that will PASS the built-in "depth" criteria, and one prompt that will FAIL it.
"""In other words:
 -write one question that returns an insightful answer (an answer with depth)
 -and one question that returns a simple answer (an answer with no depth. maybe a simple calculation?)"""
depth_criteria_passing_prompt = "What is the meaning of life?"
depth_criteria_failing_prompt = "What is 2 + 2?"

# This is a sample custom criteria that evaluates the insightfulness of a response. Do not edit this.
sample_custom_criteria = {
    "spanish": "Does the output contain words from the spanish language?"
}

# TODO: create your own CUSTOM criteria that evaluates whether a response returns a MATHEMATICAL value
your_custom_criteria = {
    "numeric": "Does the output contain numeric or mathematical information?"
}

"""The following functions are responsible for evaluating the prompts. DO NOT EDIT THEM
DO read through the functions along with the console outputs to get an idea of what the function is doing.

The evaluator will return a VALUE of Y if the answer meets the criteria.
The evaluator will return a VALUE of N if the answer does not meet the criteria."""
def built_in_depth_evaluator(query: str):

    prediction = chain.run(query)

    evaluator = load_evaluator(EvaluatorType.CRITERIA, llm=llm, criteria="depth")
    eval_result = evaluator.evaluate_strings(prediction=prediction, input=query)

    print(eval_result)
    print("\nPROMPT: ", "DEPTH")

    print("INPUT: ", query)
    print("OUTPUT:", prediction.replace("Chatbot: ","").replace("\n","").split(".")[0])
    result = (eval_result["reasoning"].replace("\n", "").replace("[BEGIN RESPONSE]", "")[0])

    print("RESULT : ", eval_result["value"])

    if result == "Y":
        print("The output has depth")
    else:
        print("The output does not have depth")

    print("--------------------------------")

def custom_evaluator():

    evaluator = load_evaluator(EvaluatorType.CRITERIA,criteria=sample_custom_criteria,llm=llm)

    sample_query = "How do you say I am on fire in Spanish?"
    sample_prediction = llm.invoke(sample_query)
    eval_result = evaluator.evaluate_strings(prediction=sample_prediction, input=sample_query)

    print("\nPROMPT : ", "SPANISH")
    print("INPUT: ", sample_query)
    print("OUTPUT:", sample_prediction.split("What")[0].replace("\n",""))
    sample_result = (eval_result["reasoning"].replace("\n", "").split(":")[1][0])

    print("RESULT: ", sample_result)

    if sample_result == "Y":
        print("The output contains Spanish words")
    else:
        print("The output does not contain Spanish words")

    print("--------------------------------")

    evaluator = load_evaluator(EvaluatorType.CRITERIA,criteria=your_custom_criteria,llm=llm)

    your_query = "What is 5 * 5?"
    your_prediction = llm.invoke(your_query)
    eval_result = evaluator.evaluate_strings(prediction=your_prediction, input=your_query)

    print("PROMPT : ", "NUMERIC")
    print("INPUT: ", your_query)
    print("OUTPUT:", your_prediction.split("What")[0].replace("\n",""))
    your_result = (eval_result["reasoning"].replace("\n", "").replace("[BEGIN RESPONSE]", "")[0])

    print("RESULT: ", your_result)

    if your_result == "Y":
        print("The output contains numeric or mathematical information")
    else:
        print("The output does not contain numeric or mathematical information")

    print("--------------------------------")
