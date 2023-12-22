import os

from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.evaluation import load_evaluator, EvaluatorType
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

"""
All requests to the LLM require some form of a key.
Other sensitive data has also been hidden through environment variables.
"""
api_key = os.environ['OPENAI_API_KEY']
base_url = os.environ['OPENAI_API_BASE']
version = os.environ['OPENAI_API_VERSION']

"""
This lab will guide you through _________

LINK GOES HERE
"""

"""
Defining our LLM, templates, prompt and chain. No need to edit these.
"""
llm = AzureChatOpenAI(model_name="gpt-35-turbo")

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

# TODO: Add one prompt that will PASS the built-in "depth" criteria, and one that will FAIL.
# In other words, one prompt that will return an in-depth insightful answer, and one that will return a short answer
depth_criteria = ["What is the meaning of life?", "What is 4 cubed?"]

# This dictionary will use built-in criteria to evaluate LLM responses. Do not to edit this.
# It has a "conciseness" sample criteria already written for you
built_in_criteria = {
    "conciseness": ["What is 4 cubed?", "Give me a detailed history of Japan"],  # <- Sample, do not edit
    "depth": depth_criteria,
}

# This is a sample custom criteria that evaluates the insightfulness of a response. Do not edit.
sample_custom_criteria = {
    "spanish": "Does the output contain words from the spanish language?"
}

# TODO: create your own custom criteria that evaluates whether a response returns a MATHEMATICAL value
your_custom_criteria = {
    "numeric": "Does the output contain numeric or mathematical information?"
}

# # This dictionary will use custom criteria to evaluate LLM responses. Do not to edit this
# custom_criteria = {
#     #sample_custom_criteria["insightful"]: ["What is the meaning of life? What are some theories?", "What is 4 cubed?"],
#     your_custom_criteria["numeric"]: ["What is 4 cubed?", "Tell me a joke about skeletons."]
# }

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

    your_query = "How do you say I am on fire in Spanish?"
    your_prediction = "Estoy en fuego"
    eval_result = evaluator.evaluate_strings(prediction=your_prediction, input=your_query)

    print("\nPROMPT : ", "SPANISH")
    print("RESULT :\n","\n".join(your_prediction.replace("\n","").split(".")[:-1]))
    print("VALUE :", eval_result["value"])
    print("SCORE :", eval_result["score"])
    print("REASON :\n","\n".join(eval_result["reasoning"].replace("\n","").split(".")[:-1]))
    print("--------------------------------")

    evaluator = load_evaluator(EvaluatorType.CRITERIA,criteria=your_custom_criteria,llm=llm)

    your_query = "What is 5 * 5?"
    your_prediction = "5 * 5 = 25"
    eval_result = evaluator.evaluate_strings(prediction=your_prediction, input=your_query)

    print("\nPROMPT : ", "NUMERIC")
    print("RESULT :\n","\n".join(your_prediction.replace("\n","").split(".")[:-1]))
    print("VALUE :", eval_result["value"])
    print("SCORE :", eval_result["score"])
    print("REASON :\n","\n".join(eval_result["reasoning"].replace("\n","").split(".")[:-1]))
    print("--------------------------------")
