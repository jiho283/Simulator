def judge_eq(true_answer_op, answer):
    """
    A function that judges whether the model's output is correct or not.
    Handles ambiguous cases. If ambiguous, we return the flag of ambiguity with `True`.

    Parameters:
        true_answer_op: one of (A), (B), (C), (D), (E)
        answer: generated text from the model

    Returns:
        result (str): 'Correct' or 'Wrong'
        is_ambiguous (bool): flag of cases whether the model is outputting ambiguous answer.
    """    
    is_ambiguous = False
    # is_ambiguous: count of cases when the model is outputting multiple options
    ## this is ambiguous because:
    ### (1) the model is just enumerating multiple options without actually answering the question
    ### (2) the model is reasoning with explicitly mentioning other options and gives the correct answer
    ### (3) the model is reasoning with explicitly mentioning other options and gives the wrong answer
    result = 'Wrong'
    if answer == "":
        is_ambiguous = False
        result = 'Wrong'
        return result, is_ambiguous
    elif true_answer_op in answer: # if the true answer is in the generated text
        op_list = ['(A)', '(B)', '(C)', '(D)', '(E)', '(a)', '(b)', '(c)', '(d)', '(e)']
        try:
            op_list.remove(true_answer_op)
            op_list.remove(true_answer_op.lower())
        except:
            print(f"UNEXPECTED ERROR: true_answer_op is set to {true_answer_op}.")
        for op in op_list:
            if op in answer: # if the other options are in the generated text
                result = 'Ambiguous'
                is_ambiguous = True
                return result, is_ambiguous
        result = 'Correct'
    return result, is_ambiguous

def distill_answer(answer):
    # answer: model output
    # returns: distilled answer from model output
    # should be called after measuring answer time
    candidate_list = ["(A)", "(B)", "(C)", "(D)", "(E)"]
    for candidate in candidate_list:
        if candidate in answer or candidate.lower() in answer:
            op_list = ['(A)', '(B)', '(C)', '(D)', '(E)', '(a)', '(b)', '(c)', '(d)', '(e)']
            op_list.remove(candidate)
            op_list.remove(candidate.lower())
            for op in op_list:
                if op in answer:
                    return "([/AMB/])" # when output has multiple options
            return candidate
    return "([N/A])"

def calibrate(true_answer_op, answer, question, distilled_answer, lenient=True):
    """Disambiguates ambiguous answers and evaluate model's answsers with a more lenient criteria.
    By using this function, you consider the followings to also be a correct answer from a model.
    - (lenient) When the model outputs only the description of the option. e.g. "Chandler" instead of "(C) Chandler"
    - (lenient) Various absentions when the true answer is "(E)". e.g., No options in the text, but the model answers "None of the options are correct"
    - (disambiguation) When it answers it correctly, but also has other options in the answer.

    Paramters:
        - true_answer_op (str): True answer. one of (A), (B), (C), (D), (E)
        - answer (str): Model's output
        - question (str): question asked along with MC options
        - distilled answer (str): output of `distill_answer`
        - lenient (bool): Whether to use a more lenient evaluating system.
    Returns:
        - result (str): Wrong, Correct, Ambiguous
        - is_ambiguous (bool): whether the result('Wrong') is still ambigous or not
        - calibrated_distilled_answer (str): calibrated version of the output of `distill_answer`
    """
    op2idx = {
        "(A)" : 1,
        "(B)" : 2,
        "(C)" : 3,
        "(D)" : 4,
        "(E)" : 5
    }
    true_text = question.split("\t")[op2idx[true_answer_op]].replace(true_answer_op, "").strip()
    if distilled_answer == "([/AMB/])":
        if "none of the options" in answer:
            calibrated_distilled_answer = "(E)"
            if true_answer_op == "(E)":
                result = "Correct"
            else:
                result = "Wrong"
            is_ambiguous = False
            return result, is_ambiguous, calibrated_distilled_answer
        for option in ['(A)', '(B)', '(C)', '(D)', '(E)']:
            if f"answer is {option}" in answer:
                calibrated_distilled_answer = option
                if true_answer_op == option:
                    result = "Correct"
                else:
                    result = "Wrong"
                is_ambiguous = False
                return result, is_ambiguous, calibrated_distilled_answer
            if "[EXPLANATION]" in answer.upper():
                explicit_answer = answer.upper().split("[EXPLANATION]")[0]
                if true_answer_op in explicit_answer:
                    op_list = ['(A)', '(B)', '(C)', '(D)', '(E)']
                    try:
                        op_list.remove(true_answer_op)
                    except:
                        print(f"UNEXPECTED ERROR: true_answer_op is set to {true_answer_op}.")
                    for op in op_list:
                        if op in answer: # if the other options are in the generated text
                            result = 'Ambiguous'
                            is_ambiguous = True
                            return result, is_ambiguous, "([/AMB/])"
                    result = 'Correct'
                    is_ambiguous = False
                    calibrated_distilled_answer = true_answer_op
                    return result, is_ambiguous, calibrated_distilled_answer
                elif true_text in explicit_answer and lenient: # lenient because we are allowing violation of the format
                    text_A = question.split("\t")[1].replace("(A)","").strip()
                    text_B = question.split("\t")[2].replace("(B)","").strip()
                    text_C = question.split("\t")[3].replace("(C)","").strip()
                    text_D = question.split("\t")[4].replace("(D)","").strip()
                    text_E = question.split("\t")[5].replace("(E)","").strip()
                    text_list = [text_A, text_B, text_C, text_D, text_E]
                    try:
                        text_list.remove(true_text)
                    except:
                        raise AssertionError(f"UNEXPECTED ERROR: true_text is set to {true_text}.")
                    for text in text_list:
                        if text in explicit_answer:
                            result = 'Ambiguous'
                            is_ambiguous = True
                            print(f"AMBIGUOUS. true text: {true_text}  answer: {answer}")
                            return result, is_ambiguous, "([/AMB/])"
                    result = "Correct"
                    is_ambiguous = False
                    calibrated_distilled_answer = true_answer_op
                    return result, is_ambiguous, calibrated_distilled_answer   
        else: 
            return "Wrong", True, "([/AMB/])"
            
    if distilled_answer == "([N/A])" and lenient:
        if "none of the options" in answer or "I cannot answer this question" in answer or "context does not" in answer or "cannot answer" in answer or "does not specify" in answer or "does not provide" in answer:
            calibrated_distilled_answer = "(E)"
            if true_answer_op == "(E)":
                result = "Correct"
            else:
                result = "Wrong"
            is_ambiguous = False
            return result, is_ambiguous, calibrated_distilled_answer
        if answer.replace("\n","").replace(" ","").replace("\t","") == "":
            is_ambiguous = False
            result = 'Wrong'
            return result, is_ambiguous, "([N/A])"
        if true_text in answer:    
            text_A = question.split("\t")[1].replace("(A)","").strip()
            text_B = question.split("\t")[2].replace("(B)","").strip()
            text_C = question.split("\t")[3].replace("(C)","").strip()
            text_D = question.split("\t")[4].replace("(D)","").strip()
            text_E = question.split("\t")[5].replace("(E)","").strip()
            text_list = [text_A, text_B, text_C, text_D, text_E]
            try:
                text_list.remove(true_text)
            except:
                raise AssertionError(f"UNEXPECTED ERROR: true_text is set to {true_text}.")
            for text in text_list:
                if text in answer:
                    result = 'Ambiguous'
                    is_ambiguous = True
                    print(f"AMBIGUOUS. true text: {true_text}  answer: {answer}")
                    return result, is_ambiguous, "([/AMB/])"
            result = "Correct"
            is_ambiguous = False
            calibrated_distilled_answer = true_answer_op
            return result, is_ambiguous, calibrated_distilled_answer
        else:
            return "Wrong", True, "([/AMB/])"
    if "<<<Timeout" in answer:
        is_ambiguous = False
        result = 'Wrong'
        return result, is_ambiguous, "<<<Timeout>>>"
    else:
        if distilled_answer == true_answer_op:
            result = "Correct"
        else:
            result = "Wrong"
        is_ambiguous = False
        return result, is_ambiguous, distilled_answer