from aiogram import Bot, types
from aiogram.utils import executor
from aiogram.dispatcher import Dispatcher
import moviepy
from pydub import AudioSegment

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

device = torch.device("cpu")

conformer = Conformer()
conformer.to(device)

weights = torch.load("conformer.pt", map_location=torch.device('cpu'))
conformer.load_state_dict(weights)

sp_tokenizer = sentencepiece.SentencePieceProcessor(model_file='nemo_tokenizer.model')
def process_audios(model, tokenizer, paths):
    elements = []
    for path in paths:
        audio, audio_lenght = dataset.open_audio(path, 16000)
        elements.append(("", audio, audio_lenght, "", torch.tensor([]), 0))
    batch = collate_fn(elements)

    batch["audio"] = batch["audio"].to(device)
    batch["audio_len"] = batch["audio_len"].to(device)


    model.eval()
    log_pb, enc_len, gp = model(batch["audio"], batch["audio_len"])

    return (ctc_greedy_decoding(log_pb, enc_len, len(tokenizer), tokenizer))


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

    if (file != None):
        paths = []
        song = AudioSegment.from_file("audio." + file_path.split(".")[-1])
        ten_seconds = 10 * 1000
        for i in range (len(song)//ten_seconds+1):
            cut = song[i*ten_seconds:(i+1)*ten_seconds]
            cut.export("CuttedAudio/cut" + str(i+1) + ".wav", format="wav")
            paths.append("CuttedAudio/cut" + str(i + 1) + ".wav")
            print(len(cut))
        print("ggg")

    await message.answer(process_audios(conformer, sp_tokenizer, paths))


executor.start_polling(
    dp,
    skip_updates=True,
)



