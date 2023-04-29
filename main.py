from hay_say_common import ROOT_DIR, PREPROCESSED_DIR, OUTPUT_DIR, CACHE_EXTENSION, get_model_path, clean_up, \
    construct_full_error_message, read_audio, save_audio_to_cache

from flask import Flask, request

import os.path
import traceback
import json
import subprocess
import base64
import shutil

ARCHITECTURE_NAME = 'so_vits_svc_4'
ARCHITECTURE_ROOT = os.path.join(ROOT_DIR, ARCHITECTURE_NAME)

RAW_COPY_FOLDER = os.path.join(ARCHITECTURE_ROOT, 'raw')
OUTPUT_COPY_FOLDER = os.path.join(ARCHITECTURE_ROOT, 'results')
INFERENCE_TEMPLATE_PATH = os.path.join(ARCHITECTURE_ROOT, 'inference_main_template.py')
INFERENCE_CODE_PATH = os.path.join(ARCHITECTURE_ROOT, 'inference_main.py')

INFERENCE_CODE_PATH_FOR_VEC768_LAYER12 = os.path.join(ROOT_DIR, 'so_vits_svc_4_Vec768-Layer12', 'inference_main.py')

PYTHON_EXECUTABLE = os.path.join(ROOT_DIR, '.venvs', 'so_vits_svc_4', 'bin', 'python')

app = Flask(__name__)


@app.route('/generate', methods=['POST'])
def generate() -> (str, int):
    code = 200
    message = ""
    try:
        input_filename_sans_extension, character, semitone_pitch, output_filename_sans_extension = parse_inputs()
        copy_input_audio(input_filename_sans_extension)
        execute_program(input_filename_sans_extension, character, semitone_pitch)
        copy_output(output_filename_sans_extension)
        clean_up(get_temp_files())
    except BadInputException:
        code = 400
        message = traceback.format_exc()
    except Exception:
        code = 500
        message = construct_full_error_message(ARCHITECTURE_ROOT, get_temp_files())

    # The message may contain quotes and curly brackets which break JSON syntax, so base64-encode the message.
    message = base64.b64encode(bytes(message, 'utf-8')).decode('utf-8')
    response = {
        "message": message
    }

    return json.dumps(response, sort_keys=True, indent=4), code


def parse_inputs():
    # todo: add the optional parameters "auto_pitch_correction", "cluster_model_path" (?), and "cluster_infer_ratio" (?)
    check_for_missing_keys()
    input_filename_sans_extension = request.json['Inputs']['User Audio']
    character = request.json['Options']['Character']
    semitone_pitch = request.json['Options']['Semitone Pitch']
    output_filename_sans_extension = request.json['Output File']
    check_types(input_filename_sans_extension, character, semitone_pitch, output_filename_sans_extension)
    return input_filename_sans_extension, character, semitone_pitch, output_filename_sans_extension


def check_for_missing_keys():
    missing_user_audio = ('Inputs' not in request.json.keys()) or ('User Audio' not in request.json['Inputs'].keys())
    missing_character = ('Options' not in request.json.keys()) or ('Character' not in request.json['Options'].keys())
    missing_semitone_pitch = ('Options' not in request.json.keys()) \
        or ('Semitone Pitch' not in request.json['Options'].keys())
    missing_output_filename = 'Output File' not in request.json.keys()
    if missing_user_audio or missing_character or missing_semitone_pitch or missing_output_filename:
        message = ('Missing "User Audio" \n' if missing_user_audio else '') \
                + ('Missing "Character" \n' if missing_character else '') \
                + ('Missing "Semitone Pitch" \n' if missing_semitone_pitch else '') \
                + ('Missing "Output File" +n' if missing_output_filename else '')
        raise BadInputException(message)


def check_types(input_filename_sans_extension, character, semitone_pitch, output_filename):
    wrong_type_user_audio = not isinstance(input_filename_sans_extension, str)
    wrong_type_character = not isinstance(character, str)
    wrong_type_semitone_pitch = not isinstance(semitone_pitch, int)
    wrong_type_output_filename = not isinstance(output_filename, str)
    if wrong_type_user_audio or wrong_type_character or wrong_type_semitone_pitch or wrong_type_output_filename:
        message = ('"User Audio" should be a string \n' if wrong_type_user_audio else '') \
                + ('"Character" should be a string \n' if wrong_type_character else '') \
                + ('"Semitone Pitch" should be an int \n' if wrong_type_semitone_pitch else '') \
                + ('"Output File" should be a string \n' if wrong_type_output_filename else '')
        raise BadInputException(message)


class BadInputException(Exception):
    pass


def get_model_and_config_paths(character):
    character_dir = get_model_path(ARCHITECTURE_NAME, character)
    model_filename, config_filename = get_model_and_config_filenames(character_dir)
    model_path = os.path.join(character_dir, model_filename)
    config_path = os.path.join(character_dir, config_filename)
    return model_path, config_path


def get_model_and_config_filenames(character_dir):
    return get_model_filename(character_dir), get_config_filename(character_dir)


def get_config_filename(character_dir):
    potential_name = os.path.join(character_dir, 'config.json')
    if not os.path.isfile(potential_name):
        raise Exception('Config file not found! Expecting a file with the name config.json in ' + character_dir)
    else:
        return potential_name


def get_model_filename(character_dir):
    potential_names = [file for file in os.listdir(character_dir) if file.startswith('G_')]
    if len(potential_names) == 0:
        raise Exception('Model file was not found! Expected a file with the name G_<number>.pth in ' + character_dir)
    if len(potential_names) > 1:
        raise Exception('Too many model files found! Expected only one file with the name G_<number>.pth in '
                        + character_dir)
    else:
        return potential_names[0]


def get_speaker(character):
    character_dir = get_model_path(ARCHITECTURE_NAME, character)
    config_filename = get_config_filename(character_dir)
    with open(config_filename, 'r') as file:
        config_json = json.load(file)
    speaker_dict = config_json['spk']
    speaker = get_speaker_key(character_dir, speaker_dict)
    return speaker


def get_speaker_key(character_dir, speaker_dict):
    all_speakers = speaker_dict.keys()
    if len(all_speakers) == 1:
        return list(all_speakers)[0]
    else:
        selected_speaker = get_speaker_from_speaker_config(character_dir)
        if selected_speaker not in all_speakers:
            raise Exception("The key \"" + selected_speaker + "\", from speaker.json, not found in config.json. "
                                                              "Expecting one of: " + str(list(all_speakers)))
        else:
            return selected_speaker


def get_speaker_from_speaker_config(character_dir):
    potential_json_path = os.path.join(character_dir, 'speaker.json')
    if not os.path.isfile(potential_json_path):
        raise Exception("speaker.json not found! If config.json has more than one speaker, then you must add a "
                        "speaker.json file to the character folder which specifies the desired speaker. The contents "
                        "of speaker.json should be a single entry in the following format: "
                        "{\"speaker\": <desired speaker name>}")
    else:
        with open(potential_json_path, 'r') as file:
            speaker_selector = json.load(file)
        return speaker_selector['speaker']


def copy_input_audio(input_filename_sans_extension):
    # todo: make sure we are supplying a file format that so_vits_svc_4 can use. WAV and FLAC are known to work.
    source = os.path.join(PREPROCESSED_DIR, input_filename_sans_extension + CACHE_EXTENSION)
    target = os.path.join(RAW_COPY_FOLDER, input_filename_sans_extension + CACHE_EXTENSION)
    try:
        shutil.copyfile(source, target)
    except Exception as e:
        raise Exception("Unable to copy file from Hay Say's audio cache to so_vits_svc_4's raw directory.") from e


def execute_program(input_filename_sans_extension, character, semitone_pitch):
    # todo: redirect stdout to a log file.
    model_path, config_path = get_model_and_config_paths(character)
    inference_path = determine_inference_path(config_path)
    subprocess.run([PYTHON_EXECUTABLE, inference_path,
                    '-m', model_path,
                    '-c', config_path,
                    '-n', input_filename_sans_extension + CACHE_EXTENSION,
                    '-t', str(semitone_pitch),
                    '-s', get_speaker(character)
                    ])


def determine_inference_path(config_path):
    # If it looks like the model was trained using the 4.0-Vec768-Layer12 branch, then use that code. Otherwise, use the
    # main 4.0 branch.
    inference_path = INFERENCE_CODE_PATH
    with open(config_path, 'r') as file:
        if 'contentvec_final_proj' in json.load(file).get('data').keys():
            inference_path = INFERENCE_CODE_PATH_FOR_VEC768_LAYER12
    return inference_path


def copy_output(output_filename_sans_extension):
    filename = get_output_filename()
    source_path = os.path.join(OUTPUT_COPY_FOLDER, filename)
    array_output, sr_output = read_audio(source_path)
    save_audio_to_cache(OUTPUT_DIR, output_filename_sans_extension, array_output, sr_output)


def get_output_filename():
    all_filenames = [file for file in os.listdir(OUTPUT_COPY_FOLDER)]
    if len(all_filenames) == 0:
        raise Exception('No output file was produced! Expected file to appear in ' + OUTPUT_COPY_FOLDER)
    elif len(all_filenames) > 1:
        message = 'More than one file was found in ' + OUTPUT_COPY_FOLDER + '! Please alert the maintainers of ' \
                  'Hay Say; they should be cleaning that directory every time output is generated. '
        try:
            clean_up(get_temp_files())
        except Exception as e:
            raise Exception(message + 'An attempt was made to clean the directory to correct this situation, but the '
                                      'operation failed.') from e
        raise Exception(message + 'The directory has now been cleaned. Please try generating your output again.')
    else:
        return all_filenames[0]


def get_temp_files():
    output_files_to_clean = [os.path.join(OUTPUT_COPY_FOLDER, file)
                             for file in os.listdir(OUTPUT_COPY_FOLDER)]
    input_files_to_clean = [os.path.join(RAW_COPY_FOLDER, file)
                            for file in os.listdir(RAW_COPY_FOLDER)]
    return output_files_to_clean + input_files_to_clean


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6576)
