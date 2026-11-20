# Experiment Log

Keep one section per meeting. Add the tables and plots you make from
`runs/*/results.json` and `test-results.json` (save the plots in `evidence/`),
link the pools' contact sheets, and explain what you see. A result that shows no
effect is still a result: say so and say why.

## Meeting 0 — Exploration and predictions

- **What this was:** an exercise before any training or generation: look at
  the data and the tools, then write down what I expect each meeting to show.
  These are guesses made in advance, to be checked against the real results
  later. Details in `notebooks/meeting0.ipynb`.
- **What the data shows:**
  - Egyptian Mau has 15 training photos against 70 for every other breed; the
    validation and test photos are balanced (20 and 30 per breed).
  - In the frozen ImageNet backbone, Egyptian Mau's closest breed by far is
    Bengal (cosine 0.858); Abyssinian (0.586) and Russian Blue (0.530) are much
    further away.
  - Without any training, a nearest-centroid classifier already finds 0.85 of
    the validation Maus; 1-NN, which feels the imbalance, finds 0.75.
  - 13 of the 15 training Maus are silver, side-on, whole-cat photos, often on
    studio backdrops. Validation and test also contain close-ups, bronze cats
    and cats lying or curled up. Three validation Maus (#155, #15, #148) are
    closer to another breed than to Egyptian Mau before training.
- **What I expect from each meeting:**
  - **Meeting 1 (oversample, augment):** the imbalanced baseline will be poor
    (Mau recall about 0.6 or below, errors mostly to Bengal). Oversampling will
    fix most of it, because the main problem is the class prior. Augmentation
    will add little, because every training image is already cropped, flipped
    and re-lit. The hard photos will stay hard.
  - **Meeting 2 (text-to-image):** the riskiest method. The model may draw a
    generic spotted cat instead of a Mau, which could push Bengals to "Mau"
    without helping the real Maus. I expect at best a small gain over
    oversampling.
  - **Meeting 3 (img2img, IP-Adapter):** safer, because the images start from
    real Maus. Low img2img strength will behave like oversampling; higher
    strength and IP-Adapter can add new poses, which is what the hard photos
    need.
  - **Meeting 4 (inpainting, outpainting):** new backgrounds will help less than
    new cats, because the hard Maus are hard because of the cat's view and
    coat, not the scene. Outpainting may help with distant, small cats.
  - **Final comparison:** balancing will give the largest single gain; synthetic
    data, if it helps at all, will add a few points on top.
- **Conclusion:** the first problem is the class prior, not the look of the
  breed. The second is missing views of the cat. Synthetic data is worth using
  only if it adds those views without costing Bengal recall, so Bengal recall
  is the number to watch in every meeting.

## Meeting 1 — Real-data baselines

- **Question:** How much do oversampling and ordinary augmentation help Egyptian
  Mau recall compared with the imbalanced data?
- **How `oversample()` works:**
- **How `augment()` works, and why these transforms:**
- **Commands:**
- **Results on validation:** (your table and plot)
- **What I see:**
- **Test results of the three baselines:**

## Meeting 2 — Text-to-image generation

- **Question:**
- **Setting changed in each pool (one at a time):**
- **Commands:**
- **Contact sheets:** (good and bad images, with the failure types you name)
- **Results on validation:** (table and plot)
- **What I see:**
- **Pool I take forward, and why:**
- **Its test result, compared with the baselines:**

## Meeting 3 — Generation from training photos

- **Question:** How close to the real photos should the synthetic images be?
- **img2img strength values and IP-Adapter scale values tried:**
- **Commands:**
- **Contact sheets:**
- **Results on validation:**
- **What I see:**
- **Best run of each mode, and its test result:**

## Meeting 4 — Editing training photos

- **Question:** Does the classifier need new cats or new backgrounds?
- **Background prompts and outpainting scales tried:**
- **Commands:**
- **Contact sheets:**
- **Results on validation:**
- **What I see:**
- **Best run of each mode, and its test result:**

## Final comparison

- **Runs chosen in Meetings 1–4:**
- **Results:** (your table and plot of their `test-results.json`)
- **Did the validation ranking hold on the test photos?**
- **Recommendation:** use, revise, or reject synthetic data for this task.
