from dotenv import load_dotenv
import os
import whisper
from openai import OpenAI
from gtts import gTTS

#Configurações
AUDIO_INPUT = "audio/request.wav"
LANGUAGE = "pt"

load_dotenv()

#Whisper
print("Transcrevendo áudio...")
model = whisper.load_model("small")
result = model.transcribe(AUDIO_INPUT, fp16=False)
transcription = result["text"]
print("Usuário:", transcription)

#ChatGPT
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Responda de forma clara e objetiva."},
        {"role": "user", "content": transcription}
    ]
)

chatgpt_response = response.choices[0].message.content
print("ChatGPT:", chatgpt_response)

#gTTS
print("Gerando áudio...")
tts = gTTS(text=chatgpt_response, lang=LANGUAGE)
tts.save("audio/response.mp3")

print("Resposta em áudio salva com sucesso!")
