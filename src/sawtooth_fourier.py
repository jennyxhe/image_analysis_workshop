"""Build a sawtooth wave from sine waves and plot the amplitude and phase of each.

A sawtooth wave rises steadily and then drops back at once.  Like the square
wave in fourier_series_demo.py, it can be made by adding up sine waves:

    sawtooth(t) = (2 / pi) x [ sin(1 x 2 pi f t) / 1
                             - sin(2 x 2 pi f t) / 2
                             + sin(3 x 2 pi f t) / 3
                             - ... ]

Unlike the square wave, it needs ALL multiples (1, 2, 3, 4, ...) of the basic
frequency f, not only the odd ones.  The wave with multiple k has amplitude
2 / (pi x k).  The minus signs mean that every second wave is turned upside
down, which is the same as a phase of 180 degrees.

Run it from the top folder of the workshop repository with:

    uv run src/sawtooth_fourier.py

The picture is saved as Pictures/fourier/sawtooth_fourier.png.
Change N_WAVES below to use more or fewer sine waves.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

N_WAVES = 20        # how many sine waves to add up
PERIOD = 1.0        # seconds for one "tooth"
TIME = np.linspace(0, 2 * PERIOD, 2000)  # show two teeth

REPO = Path(__file__).resolve().parent.parent
OUT_FILE = REPO / "Pictures" / "fourier" / "sawtooth_fourier.png"

WAVE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
               "#008300", "#6250d6", "#e34948"]
TARGET_COLOR = "#555555"


def sawtooth(t):
    """Rises from -1 to +1 during one period, then drops back to -1 at once.
    It is 0 at t = 0, so the drops happen half way through each period."""
    return 2 * ((t / PERIOD + 0.5) % 1) - 1


def sawtooth_recipe(n_waves):
    """The frequency, amplitude and phase of each sine wave in the sawtooth."""
    multiples = np.arange(1, n_waves + 1)            # 1, 2, 3, 4, ...
    freqs = multiples / PERIOD
    signed_amps = (2 / np.pi) * (-1.0) ** (multiples + 1) / multiples
    amps = np.abs(signed_amps)                       # size of each wave
    phases = np.where(signed_amps > 0, 0.0, np.pi)   # 0 or 180 degrees
    return freqs, amps, phases


def sine(t, freq, amp, phase):
    """One sine wave: amp x sin(2 pi x freq x t + phase)."""
    return amp * np.sin(2 * np.pi * freq * t + phase)


def tidy(ax, title):
    ax.set_title(title, loc="left", fontsize=11)
    ax.axhline(0, color="#999999", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    freqs, amps, phases = sawtooth_recipe(N_WAVES)
    waves = [sine(TIME, f, a, p) for f, a, p in zip(freqs, amps, phases)]

    fig = plt.figure(figsize=(13, 8))
    grid = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], hspace=0.4, wspace=0.25)
    ax_sum = fig.add_subplot(grid[0, 0])
    ax_parts = fig.add_subplot(grid[1, 0])
    ax_amp = fig.add_subplot(grid[0, 1])
    ax_phase = fig.add_subplot(grid[1, 1])

    # Top left: the sawtooth, and the sum of the first few / all sine waves.
    tidy(ax_sum, "The sum gets closer to the sawtooth as more waves are added")
    ax_sum.plot(TIME, sawtooth(TIME), color=TARGET_COLOR, lw=1.5, ls="--",
                label="sawtooth")
    steps = sorted({1, 3, N_WAVES})
    for n, color in zip(steps, ["#c8c8c8", "#eb6834", "#2a78d6"]):
        ax_sum.plot(TIME, np.sum(waves[:n], axis=0), color=color, lw=2,
                    label=f"sum of {n} wave{'s' if n > 1 else ''}")
    ax_sum.set_ylim(-1.5, 1.5)
    ax_sum.legend(loc="upper left", frameon=False, fontsize=9, ncol=4)

    # Bottom left: the first few waves on their own.
    n_show = min(5, N_WAVES)
    tidy(ax_parts, f"The first {n_show} sine waves (each one smaller and faster)")
    for k in range(n_show):
        ax_parts.plot(TIME, waves[k], color=WAVE_COLORS[k], lw=1.8,
                      label=f"{freqs[k]:g} Hz, phase {np.degrees(phases[k]):.0f}°")
    ax_parts.set_ylim(-1.5, 1.5)
    ax_parts.set_xlabel("time (s)")
    ax_parts.legend(loc="upper left", frameon=False, fontsize=8, ncol=3)

    # Right: amplitude and phase of every wave ("stem" plots).
    tidy(ax_amp, "Amplitude of each wave")
    tidy(ax_phase, "Phase of each wave")
    for k, (f, a, p) in enumerate(zip(freqs, amps, phases)):
        color = WAVE_COLORS[k] if k < n_show else "#8a8a8a"  # rest in gray
        ax_amp.plot([f, f], [0, a], color=color, lw=2)
        ax_amp.plot(f, a, "o", color=color, ms=6)
        ax_phase.plot([f, f], [0, np.degrees(p)], color=color, lw=2)
        ax_phase.plot(f, np.degrees(p), "o", color=color, ms=6)
    ax_amp.set_ylim(0, 0.8)
    ax_phase.set_ylim(-20, 200)
    ax_phase.set_yticks([0, 90, 180])
    ax_phase.set_ylabel("degrees")
    ax_phase.set_xlabel("frequency (Hz)")
    for ax in (ax_amp, ax_phase):
        ax.set_xlim(0, freqs[-1] + 1)

    fig.suptitle(f"A sawtooth wave built from {N_WAVES} sine waves",
                 fontsize=13)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=100)
    print(f"Saved {OUT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()
