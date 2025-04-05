from google.cloud import texttospeech
import os

# Set Google Application Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r'C:/PROJECT/transit-vision-2e6350ac393a.json'

# Initialize the Text-to-Speech client
client = texttospeech.TextToSpeechClient()

# Input text for synthesis
# text = "കോട്ടയം ഭാഗത്തേക്കുള്ള ബസ്സ് എത്തി ചേർന്നിരിക്കുന്നു"
text = "കോട്ടയം ഭാഗത്തേക്കുള്ള ബസ്സ് എത്തിച്ചേർന്നിരിക്കുന്നു"

# User-defined parameters
language_code = "ml-IN"  # Malayalam (India)
voice_gender = texttospeech.SsmlVoiceGender.NEUTRAL  # Options: MALE, FEMALE, NEUTRAL
# voice_type = "Wavenet"  # Options: "Standard", "Wavenet", "Chirp3" (if available)
voice_name = "ml-IN-Chirp3-HD-Kore"  # Check Google TTS voices for available options
# ml-IN-Chirp3-HD-Aoede
# ml-IN-Chirp3-HD-Charon
# ml-IN-Chirp3-HD-Fenrir
## ml-IN-Chirp3-HD-Kore
## ml-IN-Chirp3-HD-Leda
# ml-IN-Chirp3-HD-Orus
# ml-IN-Chirp3-HD-Puck
## ml-IN-Chirp3-HD-Zephyr
# ml-IN-Standard-A
# ml-IN-Standard-B
# ml-IN-Standard-C
# ml-IN-Standard-D
# ml-IN-Wavenet-A
# ml-IN-Wavenet-B
# ml-IN-Wavenet-C
# ml-IN-Wavenet-D


speaking_rate = 0.9  # Speech speed (1.0 is normal, <1 is slower, >1 is faster)
pitch = 0.0  # Adjust pitch (-20.0 to 20.0, 0 is default)
volume_gain_db = 0.0  # Volume adjustment (-16.0 to 16.0 dB, 0 is default)

# Configure the text input
synthesis_input = texttospeech.SynthesisInput(text=text)

# Configure voice parameters
voice = texttospeech.VoiceSelectionParams(
    language_code=language_code,
    ssml_gender=voice_gender,
    name=voice_name  # Use voice_name if you want a specific voice
)

# Configure audio output with various adjustments
audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3,  # MP3 format
    # speaking_rate=speaking_rate,  # Adjust speech speed
    pitch=pitch,  # Adjust pitch
    volume_gain_db=volume_gain_db  # Adjust volume gain
)

# Generate speech synthesis
response = client.synthesize_speech(
    input=synthesis_input,
    voice=voice,
    audio_config=audio_config
)

# Save the output audio file
output_file = "C:/PROJECT/audio/output.mp3"
with open(output_file, "wb") as out:
    out.write(response.audio_content)
    print(f"Audio content written to file '{output_file}'")

# Play the audio file (works on Windows)
os.system(f"start {output_file}")
