from transformers import pipeline


def get_speech_model():
    pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small.en"
    )   
    return pipe


def get_transcript_speech(pipe, sample):
    prediction = pipe(sample, batch_size=4)["text"]
    return "".join(i for i in prediction if ord(i) < 128)
