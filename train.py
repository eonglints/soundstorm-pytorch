import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from soundstorm_pytorch import SoundStorm, ConformerWrapper


DEMO_WAV = "demo_data/116_288045_000003_000000.wav"
BANDWIDTH = 6.0  # 6.0 in the paper but with 12 quantizers and 50fps
# 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32)
NUM_QUANTIZERS = 8  # 12 in the paper
LENGTH_SECONDS = 0.25
NUM_LAYERS = 6  # 12 in the paper
DIM = 2048  # 4096 in the paper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Instantiate a pretrained EnCodec model
encodec_model = EncodecModel.encodec_model_24khz()
assert BANDWIDTH in encodec_model.target_bandwidths

encodec_model.set_target_bandwidth(BANDWIDTH)
codebook_size = encodec_model.quantizer.bins
framerate = encodec_model.frame_rate

# Optional other useful attributes:
# normalize = encodec_model.normalize
# name = encodec_model.name
# num_quantizers = encodec_model.quantizer.n_q # this is always 32 for some reason
# bits_per_codebook = encodec_model.bits_per_codebook
# bandwidth = encodec_model.bandwidth
# target_bandwidths = encodec_model.target_bandwidths

encodec_model = encodec_model.to(device)

print(f"Using {NUM_QUANTIZERS} quantizers ({BANDWIDTH}kbps)")
print(f"Using {codebook_size} codebooks")

# Load and pre-process the audio waveform
wav, sr = torchaudio.load(DEMO_WAV)
wav = convert_audio(wav, sr, encodec_model.sample_rate, encodec_model.channels)
wav = wav.unsqueeze(0)
wav = wav.to(device)

# Extract discrete codes from EnCodec
with torch.no_grad():
    encoded_frames = encodec_model.encode(wav)
codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
# Shape of codes: [btz, num quantizers, length in frames]

# Model expects size: [bsz,  length in frames, num quantizers]
codes = codes.permute(0, 2, 1)  # NOTE, unusual that this worked without permutation

assert codes.shape[-1] == NUM_QUANTIZERS

print(f"Shape of codes before slicing: {codes.shape}")

if LENGTH_SECONDS:
    codes = codes[:, : int(LENGTH_SECONDS * framerate), :]

print(f"Shape of codes before slicing: {codes.shape}")


conformer = ConformerWrapper(
    codebook_size=codebook_size,
    num_quantizers=NUM_QUANTIZERS,
    conformer=dict(dim=DIM, depth=NUM_LAYERS),
)

model = SoundStorm(
    conformer,
    steps=18,  # 18 steps, as in original maskgit paper
    schedule="cosine",  # currently the best schedule is cosine
)

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
running_loss = 0.0
last_loss = 0.0
model.train()
print("Training...")
for i in range(1000):
    optimizer.zero_grad()

    loss, _ = model(codes)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if i % 100 == 99:
        last_loss = running_loss / 100  # loss per batch
        print(f"batch {i+1} loss: {last_loss}")
        running_loss = 0.0


# get your pre-encoded codebook ids from the soundstream from a lot of raw audio
# codes = torch.randint(0, 1024, (2, 1024))
# codes = codes.unsqueeze(0)  # expects size: [bsz, 1, 1024]
# codes = codes.to(device)

# do the below in a loop for a ton of data
# model can now generate in 18 steps. ~2 seconds sounds reasonable
# generated = model.generate(1024, batch_size=2)  # (2, 1024)
