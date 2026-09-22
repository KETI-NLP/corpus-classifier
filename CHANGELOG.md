# Changes

## 0.2.0

- Load the native Transformers export from the pinned public Hugging Face model. Model and CLI now share a custom PretrainedConfig/PreTrainedModel implementation.
- Keep original unmerged trained parameters, tokenizer, pooling, thresholds and joint decoding; preserve legacy v0.1.0 bundle loading.
- Verify native model shards and custom-code hashes; bind nested Python source changes to resumable jobs.
- Document direct AutoClass loading with trust_remote_code=True and the installed-code/offline CLI path.
- Validate exact reference logits, 96 synthetic documents on two GPUs, zero-forward completed-job resume, Parquet relations and 15 CPU tests. These are engineering checks, not new quality or large-scale throughput measurements.
