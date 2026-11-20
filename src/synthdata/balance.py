"""Meeting 1: the two classical ways to add Egyptian Mau training examples.

Both functions are yours to write; `uv run synth-train --method oversample` or
`--method augment` uses them.

augment() has two variants, chosen with the SYNTHDATA_AUGMENT environment
variable (default "mild"). Train each as its own run:

    uv run synth-train --name m1-augment --method augment --tag augment=mild
    SYNTHDATA_AUGMENT=strong uv run synth-train --name m1-augment-strong \
        --method augment --tag augment=strong

Note: train.py's TRAIN_TRANSFORM (random crop 75-100%, flip, jitter 0.15) is
applied to every training image AFTER augment(), so these transforms add on
top of it.
"""

import os
import random

from PIL import Image
from torchvision import transforms

GREY = (
    124,
    116,
    104,
)  # ImageNet mean colour, so rotated corners are not black

AUGMENTS = {
    # Mild: changes a cat could plausibly have in another real photo (pose,
    # framing, lighting). Adds variety while keeping the breed's coat pattern
    # and colour intact, so it should beat plain oversampling.
    "mild": transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=GREY
            ),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.15
            ),
        ]
    ),
    # Strong: large rotations/shifts, hue shift, grayscale and blur. The spotted
    # silver coat is what separates Egyptian Mau from similar breeds (Bengal,
    # Abyssinian...), and hue/grayscale/blur damage exactly that cue, while big
    # shifts can push the cat out of the frame. Expect no gain or a loss vs
    # "mild": the lesson is that augmentation must preserve the class signal.
    "strong": transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=35, translate=(0.2, 0.2), scale=(0.7, 1.3), fill=GREY
            ),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(5, sigma=(0.5, 2.0))], p=0.3
            ),
        ]
    ),
}

VARIANT = os.environ.get("SYNTHDATA_AUGMENT", "mild")
if VARIANT not in AUGMENTS:
    raise SystemExit(
        f"SYNTHDATA_AUGMENT={VARIANT!r}: use one of {list(AUGMENTS)}"
    )


def oversample(
    minority_photos: list[str], budget: int, rng: random.Random
) -> list[str]:
    """Choose which Egyptian Mau training photos to repeat.

    Return a list of exactly `budget` names taken from `minority_photos`; names
    may repeat. Repeat the photos as evenly as possible and use `rng` for every
    random choice, so a run can be reproduced from its seed.
    """
    if not minority_photos:
        raise ValueError("oversample() needs at least one minority photo")
    shuffled = minority_photos.copy()
    rng.shuffle(shuffled)
    base, extra = divmod(budget, len(shuffled))
    result = []
    for i, photo in enumerate(shuffled):
        result.extend([photo] * (base + (1 if i < extra else 0)))
    return result


def augment(image: Image.Image) -> Image.Image:
    """Return a randomly transformed copy of one Egyptian Mau training photo.

    The training script calls this every time an added copy is loaded, so each
    epoch sees a new variant. The variant (see AUGMENTS) is not saved by
    synth-train: give each one its own --name and --tag augment=<variant>, or a
    resumed run can mix variants.
    """
    return AUGMENTS[VARIANT](image)
