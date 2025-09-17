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
                      xsize=320, ysize=256, tdur=5, 
                      fps=30, seed=42)

print("Generating trippy-style zebranoise (smooth edges)...")
stim = zebranoise.PerlinStimulus(xsize=320, ysize=256, tdur=5, 
                                xyscale=0.2, tscale=50, seed=42)

# New trippy_zebra filter: smooth_factor, comb_freq, sigmoid_temp
stim.save_video("test_trippy_zebra.mp4", 
               filters=[("trippy_zebra", 4, 0.08, 10)],
               codec="libx264", codec_args=colab_codec_args)

print("Generating very smooth trippy zebra...")
stim.save_video("test_very_trippy.mp4", 
               filters=[("trippy_zebra", 8, 0.08, 5)],
               codec="libx264", codec_args=colab_codec_args)

print("Generating ultra-smooth trippy zebra...")
stim.save_video("test_ultra_trippy.mp4", 
               filters=[("temporal_smooth", 15), ("trippy_zebra", 6, 0.08, 3)],
               codec="libx264", codec_args=colab_codec_args)

print("Generating tanh-smooth version...")
stim.save_video("test_tanh_smooth.mp4", 
               filters=[("trippy_zebra_v2", 4, 0.08, 0.05)],
               codec="libx264", codec_args=colab_codec_args)

print("Done! All videos should now play in Colab:")
print("- test_regular.mp4: Standard zebranoise (hard edges)")
print("- test_trippy_zebra.mp4: Trippy-style smooth edges (fixed)")
print("- test_very_trippy.mp4: Very smooth edges")
print("- test_ultra_trippy.mp4: Ultra-smooth with temporal smoothing")
print("- test_tanh_smooth.mp4: Tanh-based ultra-smooth transitions")

# Display the first video as an example
try:
    from IPython.display import Video, display
    print("\nDisplaying first video:")
    display(Video("test_regular.mp4", embed=True, width=640))
except ImportError:
    print("Not in Jupyter/Colab environment - videos saved to files")