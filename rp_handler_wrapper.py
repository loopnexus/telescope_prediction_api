"""RunPod-compatible handler wrapper."""

import runpod
from rp_handler import handler as real_handler  # Import the actual client handler

# Wrapper handler to conform to RunPod's serverless signature
def handler(job):
    print("[DEBUG] RunPod job received with keys:", list(job.keys()))
    
    job_input = job.get("input", {})
    print("[DEBUG] job['input'] keys:", list(job_input.keys()))
    
    return real_handler({"input": job_input})

# Start the RunPod serverless worker
runpod.serverless.start({"handler": handler})
