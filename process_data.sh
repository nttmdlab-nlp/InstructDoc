#!/bin/bash
API_KEY=$1

# ===== KIE =====
python data_preprocessors/docile.py
python data_preprocessors/klc.py
python data_preprocessors/deepform.py
python data_preprocessors/funsd.py
python data_preprocessors/pwc.py
python data_preprocessors/wildreceipt.py
python data_preprocessors/cord.py
python data_preprocessors/sroi.py

# ===== Single-page QA =====
python data_preprocessors/visualmrc.py
python data_preprocessors/websrc.py --api_key $API_KEY
python data_preprocessors/ocrvqa.py --api_key $API_KEY
python data_preprocessors/docvqa.py
python data_preprocessors/hwsquad.py

# ===== Single-page QA w/ Discrete Reasoning =====
python data_preprocessors/tatdqa.py
python data_preprocessors/wtq.py

# ===== Single-page QA w/ Visual Reasoning =====
python data_preprocessors/iconqa.py
python data_preprocessors/ai2d.py
python data_preprocessors/scienceqa.py
python data_preprocessors/textbook.py

# ===== Single-page QA w/ Discrete and Visual Reasoning =====
python data_preprocessors/infographicvqa.py
python data_preprocessors/chartqa.py --api_key $API_KEY

# ===== Multi-page QA w/ Multi-hop, Discrete, and Visual Reasoning =====
python data_preprocessors/slidevqa.py --api_key $API_KEY
python data_preprocessors/dude.py

# ===== Document NLI =====
python data_preprocessors/tabfact.py

# ===== Dialogue =====
python data_preprocessors/llavar.py --api_key $API_KEY

# ===== Captioning =====
python data_preprocessors/scicap.py --api_key $API_KEY
python data_preprocessors/screen2words.py --api_key $API_KEY

# ===== Classification =====
python data_preprocessors/rvlcdip.py --api_key $API_KEY

# ===== ITM =====
python data_preprocessors/rvlcdip_io.py --api_key $API_KEY
python data_preprocessors/docvqa_iq.py

# ===== DLA =====
python data_preprocessors/docbank.py
python data_preprocessors/doclaynet.py
