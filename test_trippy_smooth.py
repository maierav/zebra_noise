#!/usr/bin/env python3
"""
Test script to compare regular zebranoise with trippy-style smoothing
"""

import zebranoise

# Generate a short test stimulus
print("Generating regular zebranoise...")
zebranoise.zebra_noise("test_regular.mp4", 
                      xsize=320, ysize=240, tdur=5, 
                      fps=30, seed=42)

print("Generating trippy-smoothed zebranoise...")
stim = zebranoise.PerlinStimulus(xsize=320, ysize=240, tdur=5, 
                                xyscale=0.2, tscale=50, seed=42)
stim.save_video("test_trippy_smooth.mp4", 
               filters=[("trippy_smooth", 4), ("comb", 0.08)])

print("Generating temporally smoothed version...")
stim.save_video("test_temporal_smooth.mp4", 
               filters=[("temporal_smooth", 15), ("comb", 0.08)])

print("Generating combined smooth version...")
stim.save_video("test_combined_smooth.mp4", 
               filters=[("trippy_smooth", 4), ("temporal_smooth", 11), ("blur", 1), ("comb", 0.08)])

print("Done! Compare the videos:")
print("- test_regular.mp4: Standard zebranoise")
print("- test_trippy_smooth.mp4: Spatial trippy-style smoothing")
print("- test_temporal_smooth.mp4: Temporal Hanning smoothing")
print("- test_combined_smooth.mp4: Combined spatial + temporal smoothing")