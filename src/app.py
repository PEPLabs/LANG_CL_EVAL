"""
This file contains some sample code that will run the methods found in lab.py.
No need to edit this file to complete the lab, but it will be helpful to run it to check your work.
"""
from src.main.lab import built_in_depth_evaluator, depth_criteria_passing_query, custom_spanish_evaluator, \
    custom_mathematical_evaluator


def main():

    built_in_depth_evaluator(depth_criteria_passing_query)

    custom_spanish_evaluator("How do you say 'my pants are on fire' in Spanish?")

    custom_mathematical_evaluator("What is 5 * 5?")


if __name__ == '__main__':
    main()
