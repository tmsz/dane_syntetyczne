"""Shared routine for training sessions on a CUDA machine (Google Colab T4).

Every meeting's train and test notebooks (colab/mN_*.ipynb) use these steps:

    authenticate()          first: mount Drive and make sure GitHub accepts pushes
    setup()                 pull the code, install, download the data, check the GPU
    generate_pool(name, ...) synth-generate one pool on the GPU (skips existing images)
    train_run(name, ...)    synth-train one run on the GPU (finished seeds are skipped)
    check_devices(names)    every seed of these runs was trained on cuda
    test_run(name)          synth-test one run (skipped if already tested)
    save_to_github(msg)     commit and push runs/, evidence/, pools/
    mount_drive()           mount Google Drive at /content/drive
    save_models(name)       copy runs/<name>/seed-K/model.pt to Drive
    save_images(name)       copy pools/<name>/images/ to Drive
    restore_models(name)    copy them back from Drive
    check_environment()     all of the above checks, without training

model.pt and pool images go to Drive (.gitignore keeps them out of git); everything
else goes to git. restore_models() stops if a run's models are missing from Drive.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/tmsz/dane_syntetyczne.git"
SECRETS = []  # strings never printed (a GitHub token typed in this session)


def mask(text):
    """Hide every secret in SECRETS from text before it is printed."""
    for secret in SECRETS:
        text = text.replace(secret, "***")
    return text


def sh(*cmd, env=None):
    """Run a command, stream its output (secrets masked), stop on failure.

    The output is read and printed here because Colab's kernel may not show what
    child processes write; PYTHONUNBUFFERED makes Python children print live.
    """
    print("$", mask(" ".join(map(str, cmd))), flush=True)
    env = {**(env or os.environ), "PYTHONUNBUFFERED": "1"}
    with subprocess.Popen(
        list(map(str, cmd)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=REPO_DIR,
    ) as process:
        for line in process.stdout:
            print(mask(line), end="", flush=True)
    if process.returncode:
        raise subprocess.CalledProcessError(
            process.returncode, mask(" ".join(map(str, cmd)))
        )


def require_gpu():
    """Stop unless a CUDA GPU is available, and print its name."""
    import torch

    if not torch.cuda.is_available():  # CPU training is slow and ran out of memory
        raise RuntimeError(
            "No CUDA GPU on this machine. In VS Code, connect to a new "
            "Colab server with a T4 GPU and run again."
        )
    print("GPU:", torch.cuda.get_device_name(0))


def setup():
    """Latest code from GitHub, dependencies, Oxford-IIIT Pet in ./data, a GPU."""
    os.chdir(REPO_DIR)
    try:
        sh("git", "pull", "--ff-only", "--quiet")
    except subprocess.CalledProcessError:
        print("⚠ git pull failed; continuing with the current checkout.")
    sh(sys.executable, "-m", "pip", "install", "--quiet", "-e", ".")  # deps from pyproject.toml

    from torchvision.datasets import OxfordIIITPet

    for part in ("trainval", "test"):  # skipped when already extracted
        OxfordIIITPet(root=str(REPO_DIR / "data"), split=part, download=True)
    if str(REPO_DIR / "src") not in sys.path:
        sys.path.insert(0, str(REPO_DIR / "src"))
    from synthdata.data import image_path, load_split, trimap_path

    split = load_split()
    missing = [
        p.image
        for p in split
        if not (image_path(p.image).exists() and trimap_path(p.image).exists())
    ]
    assert not missing, f"{len(missing)} photos missing from ./data, e.g. {missing[0]}"
    print(f"Data: {len(split)} course photos in ./data")
    require_gpu()


def generate_pool(name, prompts, options=()):
    """synth-generate one pool on the GPU. Images already in pools/<name> are skipped.

    Images stay out of git (.gitignore); save_images() copies them to Drive. The
    manifest, pool.json and contact sheet go to git.
    """
    require_gpu()
    print(f"\n=== generate {name} ===", flush=True)
    cmd = ["synth-generate", "--name", name, "--device", "cuda"]
    for prompt in prompts:
        cmd += ["--prompt", prompt]
    sh(*cmd, *options)


def train_run(name, method, env=None, tags=(), pool=None, workers=2):
    """synth-train one run on the GPU. Seeds that already have results are skipped.

    workers=2: Colab's free machines have 2 CPU cores; more workers only use more RAM.
    """
    require_gpu()
    print(f"\n=== train {name} ===", flush=True)
    cmd = [
        "synth-train",
        "--name",
        name,
        "--method",
        method,
        "--device",
        "cuda",
        "--workers",
        workers,
    ]
    if pool:
        cmd += ["--pool", pool]
    for tag in tags:
        cmd += ["--tag", tag]
    sh(*cmd, env={**os.environ, **(env or {})})


def check_devices(names):
    """Stop if any seed of these runs was not trained on cuda (run.json records it)."""
    devices = {
        f"{name}/{f.parent.name}": json.loads(f.read_text())["device"]
        for name in names
        for f in sorted((REPO_DIR / "runs" / name).glob("seed-*/run.json"))
    }
    wrong = {k: v for k, v in devices.items() if v != "cuda"}
    assert not wrong, f"trained without the GPU: {wrong}"
    print(f"All {len(devices)} seeds trained on cuda.")


def test_run(name):
    """synth-test one run on the test photos; skipped if it already has test results."""
    if (REPO_DIR / "runs" / name / "test-results.json").exists():
        print(f"{name}: already tested, skipped")
        return
    print(f"\n=== test {name} ===", flush=True)
    sh("synth-test", "--run", name, "--device", "cuda")


def save_to_github(message, paths=("runs", "evidence", "pools")):
    """Commit and push results so they outlive the Colab runtime.

    Sets its own commit identity, rebases onto anything pushed from the Mac, never
    waits for a password prompt, and asks once for a token if the push is refused.
    """
    git = ["git", "-C", str(REPO_DIR)]
    no_prompt = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}  # fail instead of hanging
    present = [p for p in paths if (REPO_DIR / p).exists()]
    sh(*git, "add", "--", *present)
    if subprocess.run([*git, "diff", "--cached", "--quiet"]).returncode == 0:
        print("Nothing new to save.")
        return
    sh(
        *git,
        "-c",
        "user.name=Tomasz Lepszy",
        "-c",
        "user.email=colab@users.noreply.github.com",
        "commit",
        "--quiet",
        "-m",
        message,
    )
    sh(*git, "pull", "--rebase", "--quiet")
    push = subprocess.run(
        [*git, "push", "--quiet"], env=no_prompt, capture_output=True, text=True
    )
    if push.returncode == 0:
        print(f"✅ Saved to GitHub: {message}")
        return
    print(mask(push.stderr))
    ask_token("Push refused.")  # no (valid) token in the clone: ask once, retry
    sh(*git, "push", "--quiet", env=no_prompt)
    print(f"✅ Saved to GitHub: {message}")


def can_push():
    """True if this clone can push to GitHub without asking for a password."""
    push = subprocess.run(
        ["git", "-C", str(REPO_DIR), "push", "--dry-run", "--quiet"],
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        capture_output=True,
        text=True,
    )
    return push.returncode == 0


def ask_token(reason):
    """Ask once for a GitHub token and store it in the clone's remote URL."""
    from getpass import getpass

    token = getpass(f"{reason} GitHub token (hidden; Enter = give up): ")
    if not token:
        raise RuntimeError("No GitHub token: results could not be pushed.")
    SECRETS.append(token)
    url = REPO_URL.replace("https://", f"https://{token}@")
    subprocess.run(
        ["git", "-C", str(REPO_DIR), "remote", "set-url", "origin", url], check=True
    )


def authenticate():
    """Every interactive prompt first, before any download or training.

    Mounts Google Drive and makes sure GitHub accepts pushes (asks for a token if
    not), so a long session never stops later to wait for a login.
    """
    mount_drive()
    if not can_push():
        ask_token("GitHub refuses pushes from this clone.")
        assert can_push(), "GitHub still refuses pushes: check the token's rights"
    print("✅ Google Drive mounted, GitHub push access OK")


DRIVE_RUNS = Path("/content/drive/MyDrive/dane_syntetyczne/runs")


def mount_drive():
    """Mount Google Drive at /content/drive (asks for access once per runtime)."""
    if not Path("/content/drive/MyDrive").exists():
        try:
            from google.colab import drive

            drive.mount("/content/drive")
        except Exception as error:
            raise RuntimeError(
                "Could not mount Google Drive. In VS Code, run the command "
                "'Colab: Mount Google Drive to Server...', run the cell it adds, "
                "then run this cell again."
            ) from error
    DRIVE_RUNS.mkdir(parents=True, exist_ok=True)


def _models(name):
    """The model.pt files of runs/<name>, one per seed."""
    return sorted((REPO_DIR / "runs" / name).glob("seed-*/model.pt"))


def save_models(name):
    """Copy runs/<name>/seed-K/model.pt to Drive, same layout, and verify the copies."""
    models = _models(name)
    assert len(models) == 5, f"{name}: expected 5 model.pt, found {len(models)}"
    for src in models:
        dst = DRIVE_RUNS / src.relative_to(REPO_DIR / "runs")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        assert dst.stat().st_size == src.stat().st_size, f"bad copy: {dst}"
    print(f"{name}: 5 models saved to Drive")


DRIVE_POOLS = DRIVE_RUNS.parent / "pools"


def save_images(name):
    """Copy pools/<name>/images/ to Drive and verify every copy."""
    images = sorted((REPO_DIR / "pools" / name / "images").glob("*.png"))
    images = [p for p in images if not p.name.endswith(".tmp.png")]
    assert images, f"{name}: no images in pools/{name}/images; generate it first"
    dst_dir = DRIVE_POOLS / name / "images"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in images:
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        assert dst.stat().st_size == src.stat().st_size, f"bad copy: {dst}"
    print(f"{name}: {len(images)} images saved to Drive at {dst_dir}")


def restore_models(name):
    """Copy runs/<name>/seed-K/model.pt back from Drive; stop unless all 5 are there."""
    for src in sorted((DRIVE_RUNS / name).glob("seed-*/model.pt")):
        dst = REPO_DIR / "runs" / src.relative_to(DRIVE_RUNS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
    found = len(_models(name))
    assert found == 5, f"{name}: {found}/5 models on Drive; train it first"
    print(f"{name}: 5 models restored from Drive")


def check_environment():
    """Everything a session needs, without training: code, data, GPU, Drive, push access."""
    setup()  # pull, install, data, GPU
    for tool in ("synth-train", "synth-test"):
        sh(tool, "--help")
    mount_drive()
    probe = DRIVE_RUNS / ".write-test"
    probe.write_text("ok")
    assert probe.read_text() == "ok", "Drive is not writable"
    probe.unlink()
    print("Drive: writable at", DRIVE_RUNS)
    assert can_push(), "cannot push to GitHub"
    print("GitHub: push access OK")
    print("✅ Environment ready")
