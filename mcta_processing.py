import typing as tp
import pathlib
from datetime import datetime
import itertools

from data_exporting import format_timestamp_for_file, save_csv, OutputSheet
import list_utils

"""Support for additional outputs used by the Multiple Choice Test Analysis software."""

def transform_and_save_mcta_output(answers_results: OutputSheet,
                                   keys_results: OutputSheet,
                                   files_timestamp: tp.Optional[datetime],
                                   output_folder: pathlib.Path):
    create_keys_files(keys_results, output_folder, files_timestamp)
    create_answers_files(answers_results, output_folder, files_timestamp)


def create_keys_files(keys_results: OutputSheet, output_folder: pathlib.Path, files_timestamp: tp.Optional[datetime]):
    form_code_col = keys_results.form_code_column_index

    for row in keys_results.data[1:]:
        code = row[form_code_col]
        csv_data = build_key_csv(row[keys_results.first_question_column_index:])
        save_mcta_csv(csv_data, output_folder, f"{code}_key", files_timestamp)


def create_answers_files(answers_results: OutputSheet,
                         output_folder: pathlib.Path,
                         files_timestamp: tp.Optional[datetime]):
    form_code_col = answers_results.form_code_column_index
    first_question_col = answers_results.first_question_column_index
    
    # Preserve the original index for naming students anonymously
    # List of tuples of (form code, original index, answers)
    answers_with_form_code = [(row[form_code_col], i, row[first_question_col:]) for (i, row) in enumerate(answers_results.data[1:])]
    
    # groupby requires sorted input
    sorted_by_code = sorted(answers_with_form_code, key=lambda x: x[0])
    grouped_by_code = itertools.groupby(sorted_by_code, key=lambda x: x[0])

    # Generate one output file for each form code in the answers data
    for code, group in grouped_by_code:
        group_data = [(original_index, answers) for (_, original_index, answers) in group]
        csv_data = build_answers_csv(group_data)
        # Test form code can be in [A|B] form if student selects A and B. The [|] are not safe for filename.
        file_safe_code = code.replace("[", "").replace("]", "").replace("|", "")
        save_mcta_csv(csv_data, output_folder, f"{file_safe_code}_results", files_timestamp)


def build_key_csv(answers: tp.List[str]) -> tp.List[tp.List[str]]:
    length = len(answers) - list_utils.count_trailing_empty_elements(answers)
    header = ["", "Answer", "Title", "Concept"]
    data = [[f"Q{i}", x, f"Q{i}", "unknown"] for i, x in enumerate(answers[:length], 1)]
    return [header] + data


def build_answers_csv(data: tp.List[tp.Tuple[int, tp.List[str]]]) -> tp.List[tp.List[str]]:
    header = [""] + [f"Q{i + 1}" for i in range(0, len(data[0][1]))]
    rows = [[f"Student{i}"] + answers for (i, answers) in data]
    return [header] + rows


def save_mcta_csv(data: tp.List[tp.List[str]],
                  path: pathlib.PurePath,
                  basefilename: str,
                  timestamp: tp.Optional[datetime]):
    filename = path / f"{format_timestamp_for_file(timestamp)}mcta_{basefilename}.csv"
    save_csv(data, filename)
