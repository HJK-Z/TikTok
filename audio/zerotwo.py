import librosa
import numpy as np
import pygame


def normalize(val, omn, omx):
    return (val - omn) / (omx - omn)


def getColorNorm(height, mn, mx):
    return (0, 0, 255 * normalize(height, mn, mx))


class AudioBar:
    def __init__(
        self,
        x,
        y,
        freq,
        color,
        width=50,
        min_height=10,
        max_height=100,
        min_decibel=-80,
        max_decibel=0,
    ):
        self.x, self.y, self.freq = x, y, freq
        self.color = color
        self.width, self.min_height, self.max_height = width, min_height, max_height
        self.height = min_height
        self.min_decibel, self.max_decibel = min_decibel, max_decibel
        self.__decibel_height_ratio = (self.max_height - self.min_height) / (
            self.max_decibel - self.min_decibel
        )

    def update(self, dt, decibel):
        desired_height = decibel * self.__decibel_height_ratio + self.max_height
        speed = (desired_height - self.height) / 0.1
        self.height += speed * dt
        self.color = getColorNorm(self.height, self.min_height, self.max_height)
        self.height = np.clip(self.min_height, self.max_height, self.height)

    def render(self, screen):
        pygame.draw.rect(
            screen,
            self.color,
            (
                self.x,
                self.y + screen.get_height() - self.height,
                self.width,
                self.height,
            ),
        )


filename = "audio\zerotwo.wav"
time_series, sample_rate = librosa.load(filename)
stft = np.abs(librosa.stft(time_series, hop_length=512, n_fft=2048 * 4))

spectrogram = librosa.amplitude_to_db(stft, ref=np.max)
frequencies = librosa.core.fft_frequencies(n_fft=2048 * 4)

times = librosa.core.frames_to_time(
    np.arange(spectrogram.shape[1]), sr=sample_rate, hop_length=512, n_fft=2048 * 4
)

time_index_ratio = len(times) / times[len(times) - 1]
frequencies_index_ratio = len(frequencies) / frequencies[len(frequencies) - 1]


def get_decibel(target_time, freq):
    return spectrogram[int(freq * frequencies_index_ratio)][
        int(target_time * time_index_ratio)
    ]


pygame.init()
pygame.display.set_caption("#obsessed with this")
infoObject = pygame.display.Info()

screen_w = 600
screen_h = 400
screen = pygame.display.set_mode([screen_w, screen_h])

step = 100
mx = 8000 - step
mn = 100

bars = []
frequencies = np.arange(mn, mx + step, step)
r = len(frequencies)

width = screen_w / r
x = (screen_w - width * r) / 2

for c in frequencies:
    bars.append(AudioBar(x, 0, c, (0, 0, 0), max_height=400, width=width))
    x += width

t = pygame.time.get_ticks()
getTicksLastFrame = t

pygame.mixer.music.load(filename)
pygame.mixer.music.play(0)

running = True
while running:
    t = pygame.time.get_ticks()
    deltaTime = (t - getTicksLastFrame) / 1000.0
    getTicksLastFrame = t

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for b in bars:
        db = get_decibel(pygame.mixer.music.get_pos() / 1000.0, b.freq)
        b.update(deltaTime, db)
        b.render(screen)

    pygame.display.flip()

pygame.quit()
