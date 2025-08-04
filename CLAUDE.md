# UNSW NB15 Data Sampler Task

## EXACT Instructions

Create a UNSW NB15 data sampler notebook that does EXACTLY what the CIC-data-sampler.ipynb does, with ONLY these changes:

### Changes Allowed:
1. **Data source paths**: Change to UNSW NB15 paths:
   - PCAP files: `gs://ai-cyber/datasets/unsw-nb15/pcap/pcaps 17-2-2015/`
   - CSV labels: `gs://ai-cyber/datasets/unsw-nb15/csv/`

2. **Label discovery method**: Instead of extracting labels from folder names, extract attack categories from CSV files (`attack_cat` field)

3. **Image formats**: Use ONLY `5channel_32x32` format (not all 6 formats like CIC)

### MUST Keep Exactly The Same:
- **All imports** - copy exactly from CIC notebook, no modifications
- **PCAP processing functions** - `read_pcap_packets()` function identical to CIC
- **Image encoding functions** - `encode_payload_multiformat()` identical to CIC (with multiview method)
- **Storage class** - `MultiFormatDataWriter` identical to CIC
- **Sampling process** - identical loop structure: process each label, collect 12k samples per label
- **Progress logs** - same output format: "📦 Processing label:", "✓ Collected X samples for Y"
- **Train/val/test splits** - same 15% each
- **All configuration parameters** - same payload_bytes, shard_size, etc.

### The Core Logic:
```
for label in sorted(attack_categories_from_csv):
    print(f"📦 Processing label: {label}")
    for file_info in pcap_files:
        if label_counts[label] >= 12000:
            break
        # Process packets and assign to current label
        # Same packet processing as CIC
        # Same image generation as CIC
        # Same storage as CIC
    print(f"✓ Collected {label_counts[label]} samples for {label}")
```

## What NOT To Do:
- Do NOT add protobuf fixes or error handling beyond what CIC has
- Do NOT change PCAP processing logic
- Do NOT add complex packet-to-flow matching
- Do NOT change the storage functions
- Do NOT change imports
- Do NOT overcomplicate anything

## Result Should Be:
A notebook that produces the same output structure as CIC, same progress logs, same processing approach, but with UNSW attack categories as labels instead of CIC folder names.