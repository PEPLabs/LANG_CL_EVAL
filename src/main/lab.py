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

template = """You are an informative chatbot that gives concise answers to valid questions when possible"""

system_message_template = SystemMessagePromptTemplate.from_template(template)
human_message_template = HumanMessagePromptTemplate.from_template("{input}")
chat_prompt = ChatPromptTemplate.from_messages([system_message_template, human_message_template])

chain = LLMChain(
    llm=llm,
    prompt=chat_prompt,
)

"""
Your tasks for lab completion are below: 
-You will first define some prompts that will be evaluated by the built-in "conciseness" evaluator
-Then, you will define your own custom criteria that will evaluate whether a response returns a mathematical value

Make sure to look at the samples for both tasks to get an idea of how to complete them
"""

# TODO: Add one prompt that will PASS the built-in "depth" criteria, and one prompt that will FAIL it.
# In other words:
# write a prompt that returns an insightful answer (an answer with depth)
# and one that returns a short black-and-white answer (an answer with no depth)
depth_criteria_passing_prompt = "What is the meaning of life?"
depth_criteria_failing_prompt = "What is 4 cubed?"

# This dictionary will use built-in criteria to evaluate LLM responses. Do not to edit this.
# It has a "conciseness" sample criteria already written, and the "depth" key will hold your work above.
built_in_criteria = {
    "conciseness": ["What is 4 cubed?", "Can you tell me some theories on the meaning of life?"],
    "depth": [depth_criteria_passing_prompt, depth_criteria_failing_prompt],
}

# This is a sample custom criteria that evaluates the insightfulness of a response. Do not edit this.
sample_custom_criteria = {
    "spanish": "Does the output contain words from the spanish language?"
}

# TODO: create your own custom criteria that evaluates whether a response returns a MATHEMATICAL value
your_custom_criteria = {
    "numeric": "Does the output contain numeric or mathematical information?"
}

"""The following functions are responsible for evaluating the prompts. DO NOT EDIT THEM
DO read through the functions along with the console outputs to get an idea of what the function is doing.

The evaluator will return a VALUE of Y and SCORE of 1 if the answer meets the criteria.
The evaluator will return a VALUE of N and SCORE of 0 if the answer does not meet the criteria."""
def built_in_evaluator():
    for criteria in built_in_criteria:
        evaluator = load_evaluator(EvaluatorType.CRITERIA, llm=llm, criteria=criteria)
        print("\n**{}**".format(criteria.upper()))

        for prompt in built_in_criteria[criteria]:
            prediction = chain.run(prompt)
            eval_result = evaluator.evaluate_strings(
                prediction=prediction,
                input=prompt
            )

            print("\nPROMPT : ", prompt)
            print("RESULT :\n","\n".join(prediction.replace("\n","").split(".")[:-1]))
            print("VALUE :", eval_result["value"])
            print("SCORE :", eval_result["score"])
            print("REASON :\n","\n".join(eval_result["reasoning"].replace("\n","").split(".")[:-1]))
            print("--------------------------------")
    print("********END OF BUILT-IN EVALUATORS********")

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
