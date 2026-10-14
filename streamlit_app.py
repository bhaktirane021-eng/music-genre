import os
import tempfile
import numpy as np
import librosa
import torch
import torch.nn as nn
import streamlit as st

SR, SEG_SEC, N_MELS, N_FFT, HOP, N_FRAMES = 22050, 3, 128, 2048, 512, 130
GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
EMOJI = {"blues": "🎷", "classical": "🎻", "country": "🤠", "disco": "🪩",
         "hiphop": "🎤", "jazz": "🎺", "metal": "🤘", "pop": "🎧",
         "reggae": "🌴", "rock": "🎸"}


def audio_to_segments(path):
    y, _ = librosa.load(path, sr=SR, mono=True, duration=30)
    seg_len = SR * SEG_SEC
    segs = []
    for i in range(len(y) // seg_len):
        chunk = y[i * seg_len:(i + 1) * seg_len]
        mel = librosa.feature.melspectrogram(
            y=chunk, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
        segs.append(librosa.power_to_db(mel, ref=np.max)[:, :N_FRAMES])
    return np.array(segs, dtype=np.float32)


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                         nn.ReLU(), nn.MaxPool2d(2))


class CNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(block(1, 32), block(32, 64),
                                      block(64, 128), block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.4), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


@st.cache_resource
def load_model():
    here = os.path.dirname(os.path.abspath(__file__))
    model = CNN()
    model.load_state_dict(torch.load(os.path.join(here, "model.pt"), map_location="cpu"))
    model.eval()
    norm = np.load(os.path.join(here, "norm.npz"))
    return model, float(norm["mean"]), float(norm["std"])


st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵")
st.title("🎵 Music Genre Classifier")
st.caption("Song upload kara, CNN model genre olkhel (~90% accuracy on GTZAN).")

model, mean, std = load_model()
file = st.file_uploader("Song upload kara (.wav / .mp3)", type=["wav", "mp3", "ogg", "flac"])

if file is not None:
    st.audio(file)
    if st.button("Genre olkha", type="primary"):
        suffix = os.path.splitext(file.name)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.getvalue())
            path = tmp.name
        try:
            with st.spinner("Song vishleshan hot aahe..."):
                segs = audio_to_segments(path)
                if len(segs) == 0:
                    st.error("Song khup chhota aahe. Kimaan 3 second cha song dya.")
                    st.stop()
                x = torch.tensor((segs - mean) / std).unsqueeze(1)
                with torch.no_grad():
                    probs = torch.softmax(model(x), 1).mean(0).numpy()
        finally:
            os.unlink(path)

        order = np.argsort(probs)[::-1]
        top = GENRES[order[0]]
        st.success(f"{EMOJI[top]} **{top.upper()}**  ({probs[order[0]] * 100:.0f}%)")
        for i in order[:5]:
            st.write(f"{EMOJI[GENRES[i]]} {GENRES[i]}")
            st.progress(float(probs[i]))
