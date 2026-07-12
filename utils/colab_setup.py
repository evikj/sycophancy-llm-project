"""
colab_setup.py
===============
Everything that needs to run ONCE at the start of every new Colab session.
Wrapped in functions so you never have to copy the same boilerplate into
every notebook - just call setup_environment() in the first cell.
"""


def mount_drive():
    """Mount Google Drive. If already mounted, this is a no-op."""
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)


def clone_or_update_repo(repo_url: str, local_path: str = "/content/repo", branch: str = "main"):
    """
    Clone the project's GitHub repo into Colab's local (ephemeral) disk,
    or pull the latest commit if it's already there from an earlier cell
    run in this same session.

    This is what keeps CODE version-controlled on GitHub while DATA and
    RESULTS live on Drive (see config.py for the distinction).

    Returns the local path so it can be added to sys.path.
    """
    import subprocess
    from pathlib import Path

    local_path = Path(local_path)
    if local_path.exists() and (local_path / ".git").exists():
        subprocess.run(["git", "-C", str(local_path), "pull"], check=True)
    else:
        subprocess.run(["git", "clone", "-b", branch, repo_url, str(local_path)], check=True)
    return local_path


def install_requirements(requirements_path: str = None):
    """Install required libraries quietly."""
    import subprocess
    import sys

    if requirements_path:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", requirements_path],
            check=True,
        )
    else:
        pkgs = ["transformers", "accelerate", "bitsandbytes", "datasets", "peft"]
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *pkgs],
            check=True,
        )


def setup_environment(repo_url: str = None, local_path: str = "/content/repo", branch: str = "main"):
    """
    Convenience wrapper for the first cell of every notebook.

    If repo_url is given, the repo is cloned/updated into local_path and
    that path is returned so the caller can do:
        sys.path.insert(0, str(repo_path))
    """
    mount_drive()
    repo_path = None
    if repo_url:
        repo_path = clone_or_update_repo(repo_url, local_path, branch)
        install_requirements(str(repo_path / "requirements.txt"))
    else:
        install_requirements()
    print("Environment ready: Drive mounted, libraries installed.")
    return repo_path
