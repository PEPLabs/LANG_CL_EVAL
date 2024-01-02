
"""
This file contains some sample code that will run the two methods found in lab.py.
No need to edit this file to complete the lab, but it will be helpful to run it to check your work.
"""
from src.main.lab import built_in_depth_evaluator, custom_evaluator, depth_criteria_failing_prompt, \
    depth_criteria_passing_prompt


def main():

    built_in_depth_evaluator(depth_criteria_passing_prompt)
    built_in_depth_evaluator(depth_criteria_failing_prompt)

    print("********END OF BUILT-IN EVALUATORS********")

    custom_evaluator()


if __name__ == '__main__':
    main()
