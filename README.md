---
title: Construction Site Safety Detector
emoji: 👷
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Construction Site Safety Detector

A YOLOv8n model fine-tuned to detect **workers, helmets, safety vests and gloves** in videos.
It flags workers who may not be following safety rules, for example someone with no helmet or no vest.

Upload a video and the app returns an annotated video along with a safety summary.

**Note:** `best.pt` (the trained weights) must be in the root of this Space.
