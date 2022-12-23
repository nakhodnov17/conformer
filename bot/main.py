from aiogram import Bot, types
from aiogram.utils import executor
from aiogram.dispatcher import Dispatcher
from pydub import AudioSegment

from pytube import YouTube
from moviepy.editor import *
import os
from moviepy.editor import VideoFileClip
import sys

import dataset
from dataset import AudioDataset, collate_fn
from dataset import get_libri_speech_dataset, get_golos_dataset
import sentencepiece
from torch.utils.data import Dataset, DataLoader
import torch
import torchaudio
from confromer import Conformer
from metrics import ctc_greedy_decoding

BOT_TOKEN = '5892376937:AAFxHe9BFmU3MPCflZCeMV0Ms6kREGg91Bk'
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)


class Settings:
    answer_type = ""
    segment_length = 0
    general_part = 0

    def __init__(self):
        self.answer_type = "array"
        self.segment_length = 10000
        self.general_part = 2000


settings = Settings()

device = torch.device("cpu")

conformer = Conformer()
conformer.eval()
conformer.to(device)

weights = torch.load("conformer.pt", map_location=torch.device('cpu'))
conformer.load_state_dict(weights)

sp_tokenizer = sentencepiece.SentencePieceProcessor(model_file='nemo_tokenizer.model')


async def set_default_commands(dp):
    await dp.bot.set_my_commands(
        [
            types.BotCommand('start', 'Запустить бота'),
            types.BotCommand("text", "Текстовый формат ответа"),
            types.BotCommand("array", "Формат ответа в виде массива"),
            types.BotCommand("set_length", "Установить длину отрезка"),
            types.BotCommand("set_general_part", "Установить длину пересечения отрезков"),
        ]
    )


set_default_commands(dp)


def process_audios(model, tokenizer, paths):
    elements = []
    for path in paths:
        audio, audio_lenght = dataset.open_audio(path, 16000)
        elements.append(("", audio, audio_lenght, "", torch.tensor([]), 0))
    batch = collate_fn(elements)

    batch["audio"] = batch["audio"].to(device)
    batch["audio_len"] = batch["audio_len"].to(device)

    log_pb, enc_len, gp = model(batch["audio"], batch["audio_len"])

    return (ctc_greedy_decoding(log_pb, enc_len, len(tokenizer), tokenizer))


#@dp.message_handler(commands=['text', 'array', 'set_length', 'set_general_part'])
@dp.message_handler()
async def set_type(message: types.Message):
    s = message.text

    print("command " + message.text)
    command = message.text.split()[0]
    if (command == "/text"):
        settings.answer_type = "text"
        await message.answer("Вывод в форме текста")
    elif (command == "/array"):
        settings.answer_type = "array"
        await message.answer("Вывод в форме массива")
    elif (command == "/set_general_part"):
        try:
            if (int(message.text.split()[1])<0):
                print("Отрицательная длительность? Не уважаю")
                return
            if (int(message.text.split()[1])*2>=settings.segment_length):
                await message.answer("Слишком большое пересечение")
                return
            settings.general_part = int(message.text.split()[1])
            await message.answer("Установлено пересечение " + message.text.split()[1])
        except:
            await message.answer("Ввесдите число")
    elif (command == "/set_length"):
        try:
            if (int(message.text.split()[1])<5000 or settings.general_part*2>=int(message.text.split()[1])):
                await message.answer("Слишком маленькая длина отрезка для данного пересечения или длина отрезка меньше 5 сек")
                return
            settings.segment_length = int(message.text.split()[1])
            await message.answer("Установлена длина отрезка " + message.text.split()[1])
        except:
            await message.answer("Введите число")
    else:
        await text(message)

def get_audio(url):
    try:
        YouTube(url).streams.first().download()
        # 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
        yt = YouTube(url)
        yt.streams \
            .filter(progressive=True, file_extension='mp4') \
            .order_by('resolution') \
            .desc() \
            .first() \
            .download(output_path='./', filename='vidosik.mp4')

        video = VideoFileClip("vidosik.mp4")
        video.audio.write_audiofile(r"audioxol.mp3")
        aud = os.path.join('./', 'audioxol.mp3')

        try:
            os.remove(yt.title + ".3gpp")
        except:
            print("cant remove")
        return os.path.abspath(aud)
    except:
        print("error link")
        return ""




@dp.message_handler(content_types=[
    types.ContentType.VOICE,
    types.ContentType.AUDIO,
])
async def audio(message: types.Message):
    print("audio query")
    file = None
    try:
        file_id = message.voice.file_id
        file = await bot.get_file(file_id)
        file_path = file.file_path
        await bot.download_file(file_path, "audio." + file_path.split(".")[-1])
    except:
        file_id = message.audio.file_id
        file = await bot.get_file(file_id)
        file_path = file.file_path
        await bot.download_file(file_path, "audio." + file_path.split(".")[-1])
        print("gg")

    if (file == None): return
    paths = []
    song = AudioSegment.from_file("audio." + file_path.split(".")[-1])
    ten_seconds = 10 * 1000
    step = settings.segment_length-settings.general_part
    for i in range(0, len(song), step):
        cut = song[i:i + settings.segment_length]
        cut.export("CuttedAudio/cut" + str(i//8000 + 1) + ".wav", format="wav")
        paths.append("CuttedAudio/cut" + str(i//8000 + 1) + ".wav")
    print("ggg")

    ms = process_audios(conformer, sp_tokenizer, paths)
    print(ms)
    if (settings.answer_type == "array"):
        await message.answer(ms)
        return

    text = ""
    for ten_sec in ms:
        text += ten_sec

    for i in range(len(song) // ten_seconds + 1):
        os.remove("CuttedAudio/cut" + str(i + 1) + ".wav")

    await message.answer(text)


async def text(message: types.Message):
    print("text query")
    url = message.text
    link = get_audio(url)
    if link == "":
        return

    print(link)

    paths = []
    song = AudioSegment.from_file("audioxol.mp3")
    for i in range(0, len(song), settings.segment_length-settings.general_part):
        cut = song[i:i + settings.segment_length]
        cut.export("CuttedAudio/cut" + str(i + 1) + ".wav", format="wav")
        paths.append("CuttedAudio/cut" + str(i + 1) + ".wav")
    print("cutted")
    ms = process_audios(conformer, sp_tokenizer, paths)
    print(ms)
    if (settings.answer_type == "array"):
        await message.answer(ms)
        return

    text = ""
    for ten_sec in ms:
        text += ten_sec
    print("text: ", text)
    if (text == ""):
        text = "No text"
    # for i in range(len(song) // ten_seconds + 1):
    # os.remove("CuttedAudio/cut" + str(i + 1) + ".wav")
    await message.answer(text)


executor.start_polling(
    dp,
    skip_updates=True,
)
