#!/bin/python
# -*- coding: utf-8 -*-

"""Generate an srt or a text file with timecodes using vosk speech recognition."""


import os, sys, re
import wave
import tempfile
import json, srt
from tqdm import tqdm
from datetime import timedelta
from pydub import AudioSegment
#from pydub.playback import play
from vosk import KaldiRecognizer, Model, SetLogLevel
SetLogLevel(-1) #Disable vosk's unnecessary logs

# --------------------------------------------------------------------------- #
#                                    FLAGS                                    #
# --------------------------------------------------------------------------- #
FLAG_SRT = False
FLAG_TXT = False
FLAG_REAUDIO = False
FLAG_NO_TEXT = False
FLAG_NO_CODES = False


# --------------------------------------------------------------------------- #
#                                   GLOBALS                                   #
# --------------------------------------------------------------------------- #

TMP_DIR = os.path.join(tempfile.gettempdir(), "carbon")
if not os.path.exists(TMP_DIR):
    os.mkdir(TMP_DIR)

G_OUTPUT_NAME = ''
G_TIMER_OFFSET = 0
G_INTRO = 0
G_OUTRO = 0
G_LANG = ''
G_LANG_LIST = {
    "tr": "Turkish",
    "kz": "Kazakh",
    "en-gb": "English (England)",
    "en-us": "English (American)",
    "en-in": "English (Indian)",
    "fa": "Farsi",
    "fr": "French",
    "ja": "Japanese",
    "cn": "Chinese",
    "ru": "Russian",
    "ua": "Ukrainian",
    "cs": "Czech",
    "de": "German",
    "nl": "Dutch",
    "pl": "Polish",
    "es": "Spanish",
    "pt": "Portuguese/Brazilian Portuguese",
    "it": "Italian",
    "vn": "Vietnamese",
    "ca": "Catalan",
    "ar": "Arabic",
    "tl-ph": "Filipino",
    "eo": "Esperanto",
    "hi": "Hindi",
    "sv": "Swedish",
    "el-gr": "Greek"
}


# --------------------------------------------------------------------------- #
#                                  FUNCTIONS                                  #
# --------------------------------------------------------------------------- #

def failure(msg=''):
    """Optionally print an error message & exit."""
    if msg != '':
        print("Error:", msg)
    sys.exit(1)


def parse_time_format(time_):
    """Parse time in the format of HH:MM:SS (returns a list)."""
    _time_list = re.split(r'\:', time_)

    if (len(_time_list) > 3) or (len(_time_list) < 3):
        failure("Time format must be in the form of 'HH:MM:SS'")

    for i_ in range(0, 3):
        if not _time_list[i_].isdigit():
            failure(f"Not a valid time '{_time_list[i_]}'")
        else:
            _time_list[i_] = int(_time_list[i_])

    if _time_list[0] > 99:
        failure("Hours cannot be higher than 99!")
    elif _time_list[1] > 60:
        failure("Minutes cannot be higher than 60!")
    elif _time_list[2] > 60:
        failure("Seconds cannot be higher than 60!")

    return _time_list


def convert2wav(media_file, is_reaudio, intro_in_seconds=0, outro_in_seconds=0):
    """Write an audio in wave format (returns file name in the cache)"""
    new_name = os.path.basename(os.path.splitext(media_file)[0]) + ".wav"
    new_name = os.path.join(TMP_DIR, new_name)

    if (os.path.exists(new_name)) and not is_reaudio:
        return new_name

    #https://stackoverflow.com/questions/59102171/getting-timestamps-from-audio-using-python
    def match_target_amplitude(sound, target_dBFS):
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)

    #Get the audio
    print("Reading audio please wait...")
    _audio = AudioSegment.from_file(media_file)

    #Adjust it
    _audio = _audio.set_channels(1).set_frame_rate(16000)

    #Skip intro/outro
    if (intro_in_seconds > len(_audio)) or (outro_in_seconds > len(_audio)):
        failure("Specified time exceeds video length")

    if intro_in_seconds > 0:
        intro_in_seconds = int(intro_in_seconds*1000) #Convert to milliseconds
        _audio = _audio[intro_in_seconds:]

    if outro_in_seconds > 0:
        outro_in_seconds = int(outro_in_seconds*1000) #Convert to milliseconds
        _audio = _audio[:-outro_in_seconds]

    print("Writing audio please wait...")
    _audio.export(new_name, format="wav")

    return new_name

def start_speech_recognition (audio_file_, language_, is_progress_bar):
    """Speech recognition which uses vosk api."""

    file_size = os.path.getsize(audio_file_)
    audio = wave.open(audio_file_, "rb")
    json_result = []

    #Engine
    vosk_model = Model(None, None, language_)
    engine = KaldiRecognizer(vosk_model, audio.getframerate())
    engine.SetWords(True) #NOTE: Needed for timecodes

    #Progress Bar
    if is_progress_bar:
        prog_bar = tqdm(total=file_size, desc="Processing audio")

    #Start Recognition
    data=None
    data_length=None
    result_dict=None
    while True:
        data = audio.readframes(4000)
        data_length = len(data)
        prog_bar.update(data_length)

        if data_length == 0:
            break

        if engine.AcceptWaveform(data):
            result_dict = json.loads(engine.Result())
            json_result.append(result_dict)
        else:
            result_dict = json.loads(engine.PartialResult())

    #Get final bits of the audio and flush the pipeline
    result_dict = json.loads(engine.FinalResult())
    json_result.append(result_dict)

    if is_progress_bar:
        prog_bar.set_description("Done")
        tqdm.close(prog_bar)

    return json_result

def produce_output_file (
    json_object,
    name_without_extension,
    is_srt,
    is_txt,
    has_text,
    has_codes,
    intro_in_seconds=0,
    timer_offset_in_seconds=0
):
    """Produce either an srt or a txt file"""

    #Expected json object is like below (each 'result' is a raw word):
    #[
    # {
    #  'result':
    #    [
    #       {'conf': 0.523144, 'end': 1.53, 'start': 1.11, 'word': 'Lorem'},
    #       {'conf': 1.0, 'end': 2.04, 'start': 1.56, 'word': 'ipsum'},
    #       {'conf': 1.0, 'end': 2.19, 'start': 2.04, 'word': 'dolor'},
    #       {'conf': 1.0, 'end': 2.46, 'start': 2.19, 'word': 'sit'},
    #       {'conf': 1.0, 'end': 2.58, 'start': 2.46, 'word': 'amet'},
    #    ],
    #  'text': 'Lorem ipsum dolor sit amet'
    # },
    # {
    #  'result': ...
    # }...
    #]

    #Get raw_words
    raw_words = []
    for res in json_object:
        if not "result" in res:
            continue
        raw_words.append(res["result"])

    #Get words, start & end times (in seconds)
    lines = []
    for i_ in raw_words:
        line = []
        for j_ in i_:
            word = [
                j_["word"],
                (j_["start"]),
                (j_["end"])
            ]
            line.append(word)
        lines.append(line)

    #SRT File
    if is_srt and has_codes:
        srt_file = open(name_without_extension+r".srt", "w")

        subtitles = []

        if has_text:
            for i_ in range(0, len(lines)):
                s = srt.Subtitle(
                    index=i_,
                    content=' '.join([l[0] for l in lines[i_]]),
                    start=timedelta(seconds=(lines[i_][0][1]+intro_in_seconds)),
                    end=timedelta(seconds=(lines[i_][-1][-1]+intro_in_seconds))
                )
                subtitles.append(s)
        else:
            for i_ in range(0, len(lines)):
                s = srt.Subtitle(
                    index=i_,
                    content='PLACE HOLDER TEXT',
                    start=timedelta(seconds=(lines[i_][0][1]+intro_in_seconds)),
                    end=timedelta(seconds=(lines[i_][-1][-1]+intro_in_seconds))
                )
                subtitles.append(s)

        final_srt = srt.compose(subtitles)
        srt_file.write(final_srt)

        srt_file.close()

    #TXT File
    if is_txt:
        txt_file = open(name_without_extension+r".txt", "w")

        for i_ in range(0, len(lines)):

            if has_codes and has_text:
                txt_file.write(
                    str(
                        timedelta(
                            seconds=lines[i_][0][1]+timer_offset_in_seconds
                            )
                        ).split(".")[0]
                    )
                txt_file.write("\t")

                txt_file.write("[]\t")

                txt_file.write(' '.join([l[0] for l in lines[i_]]))
                txt_file.write('\n')

            elif has_codes:
                txt_file.write(
                    str(
                        timedelta(
                            seconds=lines[i_][0][1]+timer_offset_in_seconds
                            )
                        ).split(".")[0]
                    )
                txt_file.write("\t")

                txt_file.write("[]\t")

                txt_file.write('\n')

            elif has_text:
                txt_file.write(' '.join([l[0] for l in lines[i_]]))
                txt_file.write('\n')

        txt_file.close()


# --------------------------------------------------------------------------- #
#                          ARGUMENT HELPER FUNCTIONS                          #
# --------------------------------------------------------------------------- #

def help_prompt():
    """Print my custom help prompt."""
    print(f"Usage: {sys.argv[0]} [OPTIONS] MEDIA_FILE")
    print("Generate an srt or a text file with timecodes using speech recognition.")
    print(f"Example: {sys.argv[0]} -l en-gb -o out -t srt example.mp4")
    print("\nOptions:")
    print("  -h, --help\t\tprint this help message and exit")
    print("  -l, --lang\t\tlanguage of the speech")
    print("  -o, --output\t\tname of the output file (without extension)")
    print("  -t, --output-type\tfile type to generate (srt/txt)\n" \
          "\t\t\t each type can be combined with '+' sign.\n"
          "\t\t\t Example: [...] -t srt+txt example.mp3\n" \
          "\t\t\t will produce 'test.srt' & 'test.txt'.\n")
    print(
        "      --intro\t\tpart to omit from the beginning in HH:MM:SS format")
    print("      --outro\t\tpart to omit from the ending in HH:MM:SS format")
    print("      --timer-offset\toffset value in HH:MM:SS format (ignored in srt)\n" \
          "\t\t\t Example: [...] --timer-offset 00:12:00 foo.mpg\n" \
          "\t\t\t will add +00:12:00 to generated timecodes")
    print("      --no-text\t\tonly generate timecodes")
    print("      --no-timecodes\tonly generate text (ignores srt)")
    print("\nMiscellaneous:")
    print("  -r, --reaudio\t\tre-generate the audio in cache\n" \
          "\t\t\t '--intro' & '--outro' enables this flag")
    print("      --list-langs\tlist available languages and exit")


def exit_if_exceeds(array_index, array_max, msg):
    """If array_index exceeds array_max after
    incementing then exits with message,
    else returns array_index."""
    array_index += 1
    if array_index >= array_max:
        failure(msg)
    return array_index


# --------------------------------------------------------------------------- #
#                              ARGUMENT HANDLING                              #
# --------------------------------------------------------------------------- #

MEDIA_FILE = ''

argc = len(sys.argv)
arg_index = 1
cur_arg = ''
next_arg = ''
while arg_index < argc:

    cur_arg = sys.argv[arg_index]
    next_arg = ''

    if (cur_arg == "-h") or (cur_arg == "--help") or (argc <= 1):
        help_prompt()
        sys.exit()
    elif (cur_arg == "-l") or (cur_arg == "--lang"):
        arg_index = exit_if_exceeds(arg_index, argc, f"Option '{cur_arg}' requires an argument")

        next_arg = sys.argv[arg_index]
        if next_arg in G_LANG_LIST.keys():
            G_LANG = next_arg
        else:
            failure(f"Unknown language code '{next_arg}'")
    elif (cur_arg == "-o") or (cur_arg == "--output"):
        arg_index = exit_if_exceeds(arg_index, argc, f"Option '{cur_arg}' requires an argument")

        G_OUTPUT_NAME = sys.argv[arg_index]
    elif (cur_arg == "-t") or (cur_arg == "--output-type"):
        arg_index = exit_if_exceeds(arg_index, argc, f"Option '{cur_arg}' requires an argument")

        type_list = re.split(r'\+', sys.argv[arg_index].casefold())
        for i in type_list:
            if i == "srt":
                FLAG_SRT = True
            elif i == "txt":
                FLAG_TXT = True
            else:
                failure(f"Unknown output type '{i}'")
    elif cur_arg == "--intro":
        FLAG_REAUDIO = True
        arg_index = exit_if_exceeds(arg_index, argc, f"Option '{cur_arg}' requires an argument")

        next_arg = sys.argv[arg_index]
        time_list = parse_time_format(next_arg)

        #Calculate intro in seconds
        G_INTRO = int(time_list[0] * 60 * 60)
        G_INTRO += int(time_list[1] * 60)
        G_INTRO += int(time_list[2])
    elif cur_arg == "--outro":
        FLAG_REAUDIO = True
        arg_index = exit_if_exceeds(arg_index, argc, f"Option '{cur_arg}' requires an argument")

        next_arg = sys.argv[arg_index]
        time_list = parse_time_format(next_arg)

        #Calculate outro in seconds
        G_OUTRO = int(time_list[0] * 60 * 60)
        G_OUTRO += int(time_list[1] * 60)
        G_OUTRO += int(time_list[2])
    elif cur_arg == "--timer-offset":
        arg_index = exit_if_exceeds(arg_index, argc, f"Option '{cur_arg}' requires an argument")

        next_arg = sys.argv[arg_index]
        time_list = parse_time_format(next_arg)

        #Calculate offset in seconds
        G_TIMER_OFFSET = int(time_list[0] * 60 * 60)
        G_TIMER_OFFSET += int(time_list[1] * 60)
        G_TIMER_OFFSET += int(time_list[2])
    elif cur_arg == "--no-text":
        if FLAG_NO_CODES:
            failure("Option '--no-timecodes' cannot be used with '--no-text'")
        FLAG_NO_TEXT = True
    elif cur_arg == "--no-timecodes":
        if FLAG_NO_TEXT:
            failure("Option '--no-text' cannot be used with '--no-timecodes'")
        FLAG_NO_CODES = True
    elif (cur_arg == '-r') or (cur_arg == "--reaudio"):
        FLAG_REAUDIO = True
    elif cur_arg == "--list-langs":
        print("Codes:\t\tDefinitions:")
        for key, value in G_LANG_LIST.items():
            print(f" {key}\t\t {value}")
        sys.exit()
    else:
        MEDIA_FILE = sys.argv[arg_index]
        if not os.path.exists(MEDIA_FILE):
            failure(f"Media file '{MEDIA_FILE}' not found")
        elif os.path.isdir(MEDIA_FILE):
            failure(f"'{MEDIA_FILE}' is a directory")

    arg_index += 1

if (not FLAG_SRT and not FLAG_TXT) or \
   (G_OUTPUT_NAME == '') or \
   (G_LANG == ''):
    print("Error: Insufficient arguments")
    print(f"Try '{sys.argv[0]} --help' for more information.")
    sys.exit(1)


# --------------------------------------------------------------------------- #
#                                    MAIN                                     #
# --------------------------------------------------------------------------- #

audio_file = convert2wav(MEDIA_FILE, FLAG_REAUDIO, G_INTRO, G_OUTRO)
result_json = start_speech_recognition(audio_file, G_LANG, True)
produce_output_file(
    result_json,
    G_OUTPUT_NAME,
    FLAG_SRT,
    FLAG_TXT,
    not FLAG_NO_TEXT,
    not FLAG_NO_CODES,
    G_INTRO,
    G_TIMER_OFFSET
)
