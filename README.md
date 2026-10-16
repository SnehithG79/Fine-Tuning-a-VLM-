# Fine Tuning a ViT

**LoRA fine-tuning of a Vision Transformer for chest X-ray classification**

Oct 12, 2026 · @Snehith

A pretrained Vision Transformer (ViT-base) fine-tuned with LoRA sorts chest X-rays into four categories with **93.5% test accuracy (macro-F1 0.94)** on a single T4 GPU, training only about **0.7%** of its weights.

> **Not a medical device.** This is a research prototype, not a diagnostic device. Clinical use would need patient-level validation, regulatory review and clinician oversight.

## Table of contents

- [Why this project](#why-this-project)
- [Method](#method)
- [Configuration](#configuration)
- [Epochs and early stopping](#epochs-and-early-stopping)
- [Results](#results)
- [Error analysis](#error-analysis)
- [Limitations](#limitations)
- [Improvements and further scope](#improvements-and-further-scope)
- [Implementation notes](#implementation-notes)
- [Sources](#sources)

## Why this project

A chest X-ray is a widely used, low-cost first test for lung disease, but reading it takes trained radiologists, who are scarce in many low-resource settings. A small model that sorts images into likely categories could, after proper clinical validation, help triage cases so that specialist time goes to the images that need it first.

The project links to the UN Sustainable Development Goals in three ways:

- **SDG 3, Good Health and Well-being.** COVID-19 and viral pneumonia are communicable respiratory diseases (target 3.3), and decision support for imaging could widen access to quality essential health services (target 3.8).
- **SDG 10, Reduced Inequalities.** Everything used is open and free: the Kaggle dataset, the Hugging Face model and the PEFT library. LoRA trains on one T4 GPU, so small teams and universities without large clusters can reproduce and adapt the work.
- **SDG 9, Industry, Innovation and Infrastructure.** Parameter-efficient fine-tuning updates under 1% of the weights, which cuts the compute, energy and hardware cost of building a specialised model.

The project also serves a learning goal: practising transfer learning, parameter-efficient fine-tuning, and honest evaluation on a realistic medical-imaging task.

## Method

The work ran as a single notebook on a T4 GPU, in eight steps.

1. **Data.** The Kaggle COVID-19 Radiography Database has four classes: COVID, Lung_Opacity, Normal and Viral Pneumonia. Mask folders were excluded, and each class was capped at 1,000 images (4,000 in total) to keep the first run quick.
2. **Split.** A stratified 70/15/15 split gave 2,800 training, 600 validation and 600 test images (150 per class in validation and test).
3. **Preprocessing.** Grayscale X-rays were converted to three channels, resized to 224 × 224 and normalised with the ViT's own statistics through its image processor.
4. **Model.** `google/vit-base-patch16-224-in21k` (about 85.8 million parameters), pretrained on ImageNet-21k only, with a new, randomly initialised four-way classification head.
5. **LoRA.** Small low-rank adapters were attached to the attention layers. The new classification head was trained in full, because it starts random.
6. **Training.** Hugging Face `Trainer` with fp16 mixed precision, evaluation every 50 steps, and the best checkpoint kept by validation macro-F1.
7. **Evaluation.** Accuracy, per-class precision, recall and F1, and a confusion matrix on the untouched test set.
8. **Inference.** The saved adapter and head were reloaded and used to classify single images with a confidence score.

## Configuration

Only the small LoRA adapters and the classification head were trained; the pretrained ViT layers stayed frozen.

| Setting | Value | Reason |
|---|---|---|
| Base model | `google/vit-base-patch16-224-in21k` | General visual features from ImageNet-21k |
| LoRA rank / alpha / dropout | 16 / 16 / 0.1 | Small adapters; dropout regularises them |
| LoRA target layers | `q_proj`, `v_proj` (attention) | Names depend on the library version |
| Fully trained layer | classifier head | New head starts random |
| Trainable parameters | about 0.59 million (about 0.7%) | Computed from the configuration |
| Optimiser and learning rate | AdamW, 5e-4 | LoRA tolerates a higher rate than full fine-tuning |
| Schedule | Cosine decay, warm-up for 10% of steps | Stabilises the start, settles the end |
| Batch size | 32 (88 steps per epoch) | Fits the T4 in fp16 |
| Precision | fp16 mixed | The T4 has no bfloat16 |
| Gradient clipping | 1.0 | Limits unstable updates |
| Evaluation and checkpoints | Every 50 steps; best by macro-F1; keep 2 | Selects the best model, not the last |
| Seed | 42 | Repeatable splits and training |

## Epochs and early stopping

The first run was too short, so the epoch limit was raised from 3 to 7 and early stopping with patience 3 was added as a safeguard.

**First run: 3 epochs (264 steps).** Three epochs is a common starting point for LoRA on a pretrained ViT. Validation loss fell from 0.822 to 0.318 and accuracy rose from 68.0% to 89.3%. Both were still improving at the last step, and training and validation loss were close together (about 0.32). That pattern means underfitting: the model stopped because training ended, not because it had learned what it could.

**Second run: 7 epochs (616 steps).** Longer training cut validation loss to 0.219 and lifted accuracy to 92.7% (macro-F1 0.927). Over the last two evaluations (steps 600 and 616) validation loss (0.2194) and accuracy (92.67%) did not change. This is a plateau, not overfitting, because validation loss was not rising and accuracy was not falling. Training loss (0.08 to 0.12) did sit below validation loss (0.22), a modest gap that shows the model starting to fit the training images more closely than unseen ones.

**Early stopping with patience 3.** Choosing an epoch count by hand is guesswork. The Trainer was given `EarlyStoppingCallback(early_stopping_patience=3)`, which works like this:

- At each evaluation (every 50 steps) it compares validation macro-F1, the metric used to choose the best checkpoint, with the best value so far.
- If the metric fails to improve for 3 evaluations in a row (150 steps, about 1.7 epochs), training stops.
- With `load_best_model_at_end=True`, the final model is the best checkpoint, not the last one.

Patience of 3 tolerates the small dips normal in noisy validation scores, yet stops training long before overfitting does damage.

Other safeguards already in the configuration: LoRA itself (under 1% of weights trainable), LoRA dropout of 0.1, cosine learning-rate decay and gradient clipping. Weight decay, label smoothing and data augmentation were not used (see [Improvements](#improvements-and-further-scope)).

**Outcome:** the training log ends at step 616, exactly 7 epochs (7 × 88 steps), so the run reached its epoch limit and the callback did not end it sooner.

## Results

The model classified **561 of 600** test images correctly (**93.5%**), close to the 92.7% validation score, which suggests that choosing the best checkpoint on validation did not inflate the result.

| Run | Epochs (steps) | Validation loss | Validation accuracy | Validation macro-F1 |
|---|---|---|---|---|
| 1 | 3 (264) | 0.318 | 89.3% | 0.894 |
| 2 | 7 (616) | 0.219 | 92.7% | 0.927 |

Test-set results for the final model, 150 images per class:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| COVID | 0.98 | 0.94 | 0.96 |
| Lung_Opacity | 0.90 | 0.91 | 0.90 |
| Normal | 0.89 | 0.89 | 0.89 |
| Viral Pneumonia | 0.98 | 0.99 | 0.99 |
| **Macro average** | **0.94** | **0.93** | **0.94** |

Confusion matrix (rows are actual, columns are predicted):

| Actual \ Predicted | COVID | Lung_Opacity | Normal | Viral Pneumonia |
|---|---|---|---|---|
| COVID | 141 | 4 | 5 | 0 |
| Lung_Opacity | 2 | 137 | 11 | 0 |
| Normal | 1 | 12 | 134 | 3 |
| Viral Pneumonia | 0 | 0 | 1 | 149 |

Viral Pneumonia and COVID are classified best. Normal and Lung_Opacity are weakest.

## Error analysis

Of the 39 test errors, 23 are confusions between Lung_Opacity and Normal, two classes with similar appearance on a chest X-ray.

- **Lung_Opacity and Normal:** 11 opacities were called Normal and 12 Normal images were called opacity.
- **Missed COVID:** 5 COVID images were called Normal and 4 were called Lung_Opacity. In a screening setting, a sick patient called Normal is the costliest error, so recall on the disease classes is the number to improve.
- **Viral Pneumonia:** 149 of 150 were correct. This may reflect a genuinely distinct appearance, or differences in how the images were collected (a hypothesis that a saliency check such as Grad-CAM would test).

## Limitations

The 93.5% figure describes this dataset and this split, and it probably overstates real-world performance.

- **No patient identifiers.** Images from the same patient may sit in both training and test sets, which inflates scores.
- **Balanced test set.** Each class has 150 test images by design. In practice Normal images far outnumber the others, so real-world precision for the disease classes would differ.
- **Possible shortcuts.** If classes came from different hospitals or scanners, the model could be using image quality or markers instead of lung findings.
- **Small test set.** With 600 images, a 93.5% score carries roughly ±2 percentage points of uncertainty (95% interval), and the work used one split and one seed.
- **Partial data.** At most 1,000 images per class were used; the rest of the dataset was left out.
- **Domain gap.** The backbone was pretrained on everyday photos, not medical images.
- **Not a medical device.** The model must not be used for diagnosis.

## Improvements and further scope

The highest-value next step is a trustworthy evaluation, then better generalisation, then wider scope.

| Next step | What it gives | Effort |
|---|---|---|
| Patient-level split or an external test set | An honest estimate of real-world performance | Needs patient IDs or another dataset |
| Grad-CAM saliency maps | Checks the model looks at the lungs, not markers or borders | Low |
| Data augmentation (small rotations, crops, brightness; no horizontal flips) | Smaller gap between training and validation | Low |
| Weight decay (0.05) and label smoothing (0.1) | More regularisation against overfitting | Low |
| Use all images and class weights | More examples for Normal and Lung_Opacity | Low |
| Larger LoRA: r=32, add `o_proj`, `fc1`, `fc2` | More capacity to adapt from photos to X-rays | Medium |
| Medical-pretrained backbone (for example RAD-DINO or BiomedCLIP; check licence and size) | Narrows the photo-to-X-ray domain gap | Medium |
| Several seeds, confidence intervals, full fine-tuning and CNN baselines | A rigorous comparison | Medium |
| Confidence calibration and an "uncertain" option | Safer use: low-confidence cases go to a human | Medium |
| Demo app (for example Gradio) | Lets non-programmers try the model | Low |
| Instruction-tuned vision-language model | Text answers to questions about an image | High |
| Extend to CT scans with infection masks | Slice-level classification or segmentation | High |

## Implementation notes

Three library-version problems appeared before training ran, all fixed without changing the method.

| Error | Cause | Fix |
|---|---|---|
| `NoMatchingPeftModuleError` | The installed `transformers` names ViT attention layers `q_proj`, `k_proj`, `v_proj`, `o_proj`, not `query` and `value` | Listed the model's linear layer names and targeted `q_proj` and `v_proj` |
| `ImportError: incompatible torchao` | The preinstalled `torchao` 0.10.0 is older than the 0.16.0 that PEFT requires | Uninstalled `torchao` (unused) and restarted the runtime |
| `TypeError: unexpected argument warmup_ratio` | The argument was removed in the installed `transformers` | Used `warmup_steps` set to 10% of the training steps |

## Sources

Checked while writing this report:

- UN, Sustainable Development Goal 3: Good Health and Well-being (goal title and targets 3.3 and 3.8)
- `google/vit-base-patch16-224-in21k`, Hugging Face model card (ImageNet-21k pretraining, 224 × 224 input, no fine-tuned classification head)

Background references, not re-opened for this report:

- Kaggle, COVID-19 Radiography Database
- UN, Sustainable Development Goal 10: Reduced Inequalities
- Dosovitskiy et al., *An Image is Worth 16x16 Words* (the Vision Transformer)
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*
- Hugging Face PEFT documentation

All scores in this report come from the user's own training and test runs; the trainable-parameter count is computed from the LoRA configuration.
