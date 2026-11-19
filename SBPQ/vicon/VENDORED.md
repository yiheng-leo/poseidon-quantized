# VICON runtime provenance

The VICON runtime is included directly in `SBPQ/vicon/`. Importing models,
building calibration data, optimizing SBPQ, loading exported models and running
evaluation require no external source checkout or source-path environment
variables. Model checkpoints and datasets remain user-supplied files.

Sources are the experimental VICON snapshot and integration helpers distributed
in [realcrane/SafeSVD](https://github.com/realcrane/SafeSVD), inspected at commit
`9afbcac`. VICON's original upstream is [Eydcao/VICON](https://github.com/Eydcao/VICON).
The experimental snapshot includes changes beyond the upstream release; the
imported architecture retains its checkpoint names, shapes and configuration.

| Original source | Local implementation |
| --- | --- |
| `VICON/src/models.py` | `models.py` |
| `VICON/src/model_utils.py` | `modeling_utils.py` |
| `VICON/src/dataset.py` | `dataset.py` |
| `VICON/src/dataset_utils.py` | `dataset_utils.py` |
| `VICON/src/crop_utils.py` | `crop_utils.py` |
| `safe_svd/helper/collector/vicon/vicon_attention.py` | `attention.py` |
| `safe_svd/helper/collector/vicon/vicon_calibration_dataset.py` | `calibration.py` |
| `safe_svd/helper/collector/vicon/vicon_file_order.py` | `file_order.py` |
| `VICON/configs/{model,datasets,loss}/default.yaml` | `configs/` with the same groups |
| `VICON/configs/research_calibration_order.json` | `configs/research_calibration_order.json` |
| `evaluation/vicon/checkpoints.py` | `checkpoints.py` |
| `evaluation/vicon/physics.py` | `physics.py` |

`source_hashes.json` records original file hashes. MIT license notices are
retained in `LICENSE-VICON.txt` and `LICENSE-SafeSVD.txt`.

Local changes:

- Package imports are relative and refer only to this repository.
- The cropped model permanently uses SafeSVD's non-inplace positional-embedding
  formulation. It supports gradients and MC replay without temporary class
  patching. Attention projection names and numerical equations are preserved.
- `model_utils.py` loads the original state dict strictly and makes the encoder
  Python forward explicit so Linear hooks remain active during inference.
- `data_utils.py` retains the original bc normalization and query/channel/type
  masks without constructing VICON's training optimizer or importing plotting
  libraries. Unused torchvision imports were removed from cropping helpers.
- `configs/default.yaml` retains train/test seeds and worker configuration;
  unused training-loop/optimizer/plotting settings were omitted.
- `evaluation.py` implements the research evaluator's separate loss/physics
  passes, advancing dataset streams, masked loss and relative-loss denominator.
  Paired mode resets those streams. Physical metrics come from the original
  evaluator. Output uses the current repository's compact reporting helper.
- Shared artifact deserialization maps the previous external class names to
  these local classes so full-model artifacts can be opened locally.

`tests/fixtures/vicon_reference.json` contains calibration/normalization values
and deterministic model/evaluation references captured from the original
SafeSVD code on the synthetic test fixture. Tests check those values, attention
equivalence, MC gradients, exact restart and a fresh-process public-command run
that rejects any attempted `VICON` or `safe_svd` import.
Research-mode references also cover complete test traversals (including partial
tail batches) for both original and quantized models, sharing advancing streams
across the four loss/physics passes.
