import csv
import pathlib
import typing as tp

import data_exporting
import grid_info
import list_utils
import math_utils


def get_key_form_code(answer_keys: data_exporting.OutputSheet,
                      index: int) -> str:
    keys = answer_keys.data
    form_code_column_name = data_exporting.COLUMN_NAMES[
        grid_info.Field.TEST_FORM_CODE]
    try:
        # Get the index of the column that had the test form codes
        form_code_index = list_utils.find_index(keys[0], form_code_column_name)
        return keys[index + 1][form_code_index]
    except StopIteration:
        return "*"


def establish_key_dict(answer_keys: data_exporting.OutputSheet
                       ) -> tp.Dict[str, tp.List[str]]:
    try:
        answers_start_index = list_utils.find_index(answer_keys.data[0], "Q1")
    except StopIteration:
        raise ValueError(
            "Invalid key matrix passed to scoring functions. Answers columns must be named 'Q1' through 'QN'."
        )

    return {
        get_key_form_code(answer_keys, index): key[answers_start_index:]
        for index, key in enumerate(answer_keys.data[1:])
    }


def score_results(results: data_exporting.OutputSheet,
                  answer_keys: data_exporting.OutputSheet,
                  num_questions: int) -> data_exporting.OutputSheet:
    answers = results.data
    keys = establish_key_dict(answer_keys)
    form_code_column_name = data_exporting.COLUMN_NAMES[
        grid_info.Field.TEST_FORM_CODE]
    form_code_index = list_utils.find_index(answers[0], form_code_column_name)
    answers_start_index = list_utils.find_index(
        answers[0][form_code_index + 1:], "Q1") + form_code_index + 1
    virtual_fields: tp.List[grid_info.RealOrVirtualField] = [
        grid_info.VirtualField.SCORE, grid_info.VirtualField.POINTS
    ]
    columns = results.field_columns + virtual_fields
    scored_results = data_exporting.OutputSheet(columns, num_questions)

    for exam in answers[1:]:  # Skip header row
        fields = {
            k: v
            for k, v in zip(results.field_columns, exam[:answers_start_index])
        }
        form_code = exam[form_code_index]
        try:
            if "*" in keys:
                key = keys["*"]
            else:
                key = keys[form_code]
        except KeyError:
            fields[grid_info.VirtualField.
                   SCORE] = data_exporting.KEY_NOT_FOUND_MESSAGE
            fields[grid_info.VirtualField.
                   POINTS] = data_exporting.KEY_NOT_FOUND_MESSAGE
            scored_answers = []
        else:
            scored_answers = [
                int(actual == correct)
                for actual, correct in zip(exam[answers_start_index:], key)
            ]
            fields[grid_info.VirtualField.SCORE] = str(
                round(math_utils.mean(scored_answers) * 100, 2))
            fields[grid_info.VirtualField.POINTS] = str(sum(scored_answers))
        string_scored_answers = [str(s) for s in scored_answers]
        scored_results.add(fields, string_scored_answers)

    return scored_results


def verify_answer_key_sheet(file_path: pathlib.Path) -> bool:
    try:
        with open(str(file_path), newline='') as file:
            reader = csv.reader(file)
            keys_column_name = data_exporting.COLUMN_NAMES[
                grid_info.Field.TEST_FORM_CODE]
            names = next(reader)
            keys_column_name_index = list_utils.find_index(
                names, keys_column_name)
            list_utils.find_index(names[keys_column_name_index:], "Q1")
        return True
    except Exception:
        return False
