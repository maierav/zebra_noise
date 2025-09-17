#!/usr/bin/env python3
"""
Test script to compare regular zebranoise with trippy-style smoothing
"""

import zebranoise

# Colab-friendly encoding settings
colab_codec_args = ["-pix_fmt", "yuv420p", "-profile:v", "baseline", "-level", "3.0"]

# Generate a short test stimulus
print("Generating regular zebranoise...")
zebranoise.zebra_noise("test_regular.mp4", 
                      xsize=320, ysize=256, tdur=10, 
                      fps=30, seed=42)

print("Creating trippy-style zebranoise stimulus...")
stim = zebranoise.PerlinStimulus(xsize=320, ysize=256, tdur=10, 
                                xyscale=0.2, tscale=50, seed=42)

# Ultra-smooth trippy-like filters with extra blurring
ultra_smooth_filters = [
    ("temporal_smooth", 21),      # More temporal smoothing
    ("trippy_zebra_v2", 8, 0.08, 0.03),  # Very smooth spatial + soft edges
    ("blur", 1.5),                # Extra Gaussian blur
]

print("Generating trippy-zebra at 100% Michelson contrast...")
stim.save_video("trippy_zebra_C100.mp4", 
               filters=ultra_smooth_filters + [("michelson_contrast", 1.0, 0.5)],
               codec="libx264", codec_args=colab_codec_args)

print("Generating trippy-zebra at 33% Michelson contrast...")
stim.save_video("trippy_zebra_C33.mp4", 
               filters=ultra_smooth_filters + [("michelson_contrast", 0.33, 0.5)],
               codec="libx264", codec_args=colab_codec_args)

print("Generating trippy-zebra at 5% Michelson contrast...")
stim.save_video("trippy_zebra_C05.mp4", 
               filters=ultra_smooth_filters + [("michelson_contrast", 0.05, 0.5)],
               codec="libx264", codec_args=colab_codec_args)

print("Done! Trippy-style zebranoise with Michelson contrast control:")
print("- test_regular.mp4: Standard zebranoise (hard edges)")
print("- trippy_zebra_C100.mp4: 100% Michelson contrast (full range)")
print("- trippy_zebra_C33.mp4: 33% Michelson contrast")
print("- trippy_zebra_C05.mp4: 5% Michelson contrast (subtle)")

# Display the first video as an example
try:
    from IPython.display import Video, display
    print("\nDisplaying 33% contrast version:")
    display(Video("trippy_zebra_C33.mp4", embed=True, width=640))
except ImportError:
    print("Not in Jupyter/Colab environment - videos saved to files")