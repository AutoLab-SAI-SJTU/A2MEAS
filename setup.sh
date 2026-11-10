#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash setup.sh [--benchmark robotwin|behavior|all] [--adapter]
                     [--policy-only] [--skip-downloads | --download-only]

Install pinned environments and download models and simulator assets.
  --benchmark       Select environments and resources (default: robotwin).
  --adapter         Include the BEHAVIOR Adapter, SAM and judge.
  --policy-only     Omit RoboTwin SAM; evaluate with ROBOTWIN_RSI=false.
  --skip-downloads  Install environments without downloading resources.
  --download-only   Prepare resources using already installed environments.
  --help            Show this help without changing files.

A2MEAS_RUNTIME selects the runtime directory (default: ../a2meas-runtime).
Installation uses Conda once to provision complete shared Python runtimes;
evaluation does not require Conda activation or container Python development headers.
Hugging Face uses the standard HF_HOME/HF_TOKEN settings and saved credentials.
HELP
}
benchmark=robotwin
adapter=false
policy_only=false
skip_downloads=false
download_only=false
while (( $# )); do
  case "$1" in
    --benchmark)
      if (( $# < 2 )); then usage >&2; exit 2; fi
      benchmark="$2"; shift 2 ;;
    --adapter) adapter=true; shift ;;
    --policy-only) policy_only=true; shift ;;
    --skip-downloads) skip_downloads=true; shift ;;
    --download-only) download_only=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
case "$benchmark" in robotwin|behavior|all) ;; *) usage >&2; exit 2 ;; esac
if { $adapter && [[ "$benchmark" == robotwin ]]; } ||
   { $policy_only && [[ "$benchmark" == behavior ]]; } ||
   { $download_only && $skip_downloads; }; then
  usage >&2
  exit 2
fi
minimum_glibc=2.28
if [[ "$benchmark" != robotwin ]]; then minimum_glibc=2.35; fi
glibc_version="$(getconf GNU_LIBC_VERSION)"
glibc_version="${glibc_version#glibc }"
if ! printf '%s\n' "$minimum_glibc" "$glibc_version" | sort -V -C; then
  echo "$benchmark requires glibc >= $minimum_glibc; found $glibc_version" >&2
  exit 1
fi
export PIP_DISABLE_PIP_VERSION_CHECK=1
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

runtime_dir="${A2MEAS_RUNTIME:-../a2meas-runtime}"
if [[ -z "${A2MEAS_RUNTIME:-}" && ( -e runtime || -L runtime ) ]]; then
  runtime_dir=runtime
fi
export A2MEAS_RUNTIME="$(realpath -m "$runtime_dir")"
export PIP_CACHE_DIR="$A2MEAS_RUNTIME/cache/pip"
export HF_HUB_CACHE="$A2MEAS_RUNTIME/cache/huggingface/hub"
export HF_XET_CACHE="$A2MEAS_RUNTIME/cache/huggingface/xet"
export PYTHONPYCACHEPREFIX="$A2MEAS_RUNTIME/cache/python"
if [[ -e runtime || -L runtime ]]; then
  if [[ "$(realpath -m runtime)" != "$A2MEAS_RUNTIME" ]]; then
    echo "runtime points to a different directory than A2MEAS_RUNTIME" >&2
    exit 1
  fi
fi
python310=python3.10
python311=python3.11
if ! $download_only; then
  versions=(3.10.19)
  if [[ "$benchmark" != robotwin ]]; then versions+=(3.11.13); fi
  for version in "${versions[@]}"; do
    series="${version%.*}"
    prefix="$A2MEAS_RUNTIME/python${series//./}"
    if [[ ! -x "$prefix/bin/python" ]]; then
      conda create --prefix "$prefix" --override-channels -c conda-forge "python=$version" --yes
    fi
  done
fi
[[ ! -x "$A2MEAS_RUNTIME/python310/bin/python" ]] || python310="$A2MEAS_RUNTIME/python310/bin/python"
[[ ! -x "$A2MEAS_RUNTIME/python311/bin/python" ]] || python311="$A2MEAS_RUNTIME/python311/bin/python"

if ! $download_only; then
"$python310" - <<'PY'
from pathlib import Path
import subprocess
import sys
import sysconfig

include = sysconfig.get_path("include")
if not (Path(include) / "Python.h").is_file():
    sys.exit("Missing Python development headers in the managed runtime; recreate runtime/python310 with setup.sh.")
subprocess.run(["gcc", "-fsyntax-only", "-x", "c", "-I", include, "-"],
               input="#include <Python.h>\n", text=True, check=True)
PY

fi
if [[ ! -e runtime && ! -L runtime ]]; then
  mkdir -p "$A2MEAS_RUNTIME"
  ln -s "$(realpath --relative-to="$PWD" "$A2MEAS_RUNTIME")" runtime
fi
if [[ -L resources ]]; then
  echo "resources must be a directory inside this checkout, not a symlink" >&2
  exit 1
fi
mkdir -p resources

create_environment() {
  local interpreter="$1" target="$2" executable
  if [[ -x "$target/bin/python" && ! -f "$target/pyvenv.cfg" ]]; then
    return
  fi
  for executable in python python3 python3.10 python3.11; do
    if [[ -L "$target/bin/$executable" ]]; then
      unlink "$target/bin/$executable"
    fi
  done
  "$interpreter" -m venv "$target"
}

install_model() {
  create_environment "$python310" "$A2MEAS_RUNTIME/model"
  "$A2MEAS_RUNTIME/model/bin/python" -m pip install pip==25.3
  "$A2MEAS_RUNTIME/model/bin/python" -m pip install \
    torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128 -c requirements/model.txt

  dependencies=requirements/model.txt
  if $adapter || { [[ "$benchmark" != behavior ]] && ! $policy_only; }; then
    dependencies=requirements/grounding.txt
  fi
  "$A2MEAS_RUNTIME/model/bin/python" -m pip install -r "$dependencies" "$harness"

  "$A2MEAS_RUNTIME/model/bin/python" -m pip check
}

install_robotwin() {
  create_environment "$python310" "$A2MEAS_RUNTIME/simulator"
  "$A2MEAS_RUNTIME/simulator/bin/python" -m pip install pip==25.3
  "$A2MEAS_RUNTIME/simulator/bin/python" -m pip install \
    torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 -c benchmarks/robotwin/requirements-simulator.txt
  "$A2MEAS_RUNTIME/simulator/bin/python" -m pip install \
    -r benchmarks/robotwin/requirements-simulator.txt "$harness"
  "$A2MEAS_RUNTIME/simulator/bin/python" - <<'PY'
from pathlib import Path
import os
import imageio_ffmpeg
path = Path(os.environ['A2MEAS_RUNTIME']) / 'simulator/bin/ffmpeg'
path.unlink(missing_ok=True)
path.symlink_to(os.path.relpath(imageio_ffmpeg.get_ffmpeg_exe(), path.parent))
PY

  mkdir -p "$A2MEAS_RUNTIME/cache/setup" "$A2MEAS_RUNTIME/simulator/oidn"
  curl --fail --location --output "$A2MEAS_RUNTIME/cache/setup"/oidn.tar.gz \
    https://github.com/RenderKit/oidn/releases/download/v2.3.3/oidn-2.3.3.x86_64.linux.tar.gz
  (cd "$A2MEAS_RUNTIME/cache/setup" && sha256sum --check "$OLDPWD/configs/robotwin/oidn.sha256")
  tar -xzf "$A2MEAS_RUNTIME/cache/setup"/oidn.tar.gz -C "$A2MEAS_RUNTIME/simulator/oidn" --strip-components=1
  rm "$A2MEAS_RUNTIME/cache/setup"/oidn.tar.gz
  "$A2MEAS_RUNTIME/simulator/bin/python" -m pip check
}

install_behavior() {
  sdk="$A2MEAS_RUNTIME/behavior_sdk"
  mkdir -p resources/behavior
  export OMNIGIBSON_DATA_PATH="$PWD/resources/behavior"
  if [[ -e "$sdk" || -L "$sdk" ]]; then
    if [[ ! -e "$sdk/.git" ]] || [[ "$(git -C "$sdk" rev-parse --verify HEAD)" != d3a1b651f1b83c2f339f3e7bf47d212c429ad228 ]]; then
      echo "Existing BEHAVIOR SDK cannot be verified; preserved without changes: $sdk" >&2
      exit 1
    fi
    git -C "$sdk" apply --reverse --check "$PWD/third_party/behavior/evaluation.patch"
  else
    (
      sdk_staging="$(mktemp -d "$A2MEAS_RUNTIME/behavior_sdk.XXXXXX")"
      trap 'rm -rf -- "$sdk_staging"' EXIT
      git init "$sdk_staging"
      git -C "$sdk_staging" remote add origin https://github.com/StanfordVL/BEHAVIOR-1K.git
      git -C "$sdk_staging" sparse-checkout set --no-cone /setup.sh /LICENSE /OmniGibson/ /bddl3/ /joylo/ /docs/assets/OmniGibson_logo.png
      git -C "$sdk_staging" fetch --depth 1 origin d3a1b651f1b83c2f339f3e7bf47d212c429ad228
      git -C "$sdk_staging" checkout --detach FETCH_HEAD
      git -C "$sdk_staging" apply "$PWD/third_party/behavior/evaluation.patch"
      # Do not replace a directory created by another installer in the meantime.
      mv -T -n -- "$sdk_staging" "$sdk"
      if [[ -d "$sdk_staging" ]]; then
        echo "BEHAVIOR SDK appeared during installation; preserved without changes: $sdk" >&2
        exit 1
      fi
    )
  fi
  create_environment "$python311" "$A2MEAS_RUNTIME/behavior"
  "$A2MEAS_RUNTIME/behavior/bin/python" -m pip install 'setuptools>=71,<81' wheel
  nvidia_eula=()
  if [[ "${OMNI_KIT_ACCEPT_EULA:-}" == YES ]]; then nvidia_eula=(--accept-nvidia-eula); fi
  (cd "$sdk" && PATH="$A2MEAS_RUNTIME/behavior/bin:$PATH" CONDA_PREFIX="$A2MEAS_RUNTIME/behavior" \
    bash setup.sh --confirm-no-conda --omnigibson --bddl --joylo --eval --cuda-version 12.8 "${nvidia_eula[@]}")
  "$A2MEAS_RUNTIME/behavior/bin/python" -m pip install -r benchmarks/behavior/requirements-simulator.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
  # Requests uses charset-normalizer; an unrelated chardet install can override it.
  "$A2MEAS_RUNTIME/behavior/bin/python" -m pip uninstall -y chardet
  "$A2MEAS_RUNTIME/behavior/bin/python" -m benchmarks.behavior.prepare_runtime
  # The SDK overrides Isaac Sim's incompatible websockets pin for evaluation.
  "$A2MEAS_RUNTIME/behavior/bin/python" -c \
    'import omnigibson; from omnigibson.eval.policies import WebsocketPolicy; from gello.utils.og_teleop_utils import load_available_tasks'
}

install_judge() {
  create_environment "$python311" "$A2MEAS_RUNTIME/judge"
  "$A2MEAS_RUNTIME/judge/bin/python" -m pip install --index-url https://pypi.org/simple pip==25.3
  "$A2MEAS_RUNTIME/judge/bin/python" -m pip install \
    torch==2.13.0+cu130 torchvision==0.28.0+cu130 \
    --index-url https://download.pytorch.org/whl/cu130
  "$A2MEAS_RUNTIME/judge/bin/python" -m pip install --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu130 -r benchmarks/behavior/requirements-judge.txt
  "$A2MEAS_RUNTIME/judge/bin/python" -m pip check
}

if ! $download_only; then
  harness="vla-eval @ git+https://github.com/allenai/vla-evaluation-harness.git@f740abc542cd36e5ae8c542752a0372759344476"
  install_model
  if [[ "$benchmark" != behavior ]]; then install_robotwin; fi
  if [[ "$benchmark" != robotwin ]]; then
    install_behavior
    if $adapter; then install_judge; fi
  fi
fi
if $skip_downloads; then
  exit 0
fi

model_python="$A2MEAS_RUNTIME/model/bin/python"
if [[ ! -x "$model_python" ]]; then
  echo "Model environment missing; run setup.sh without --download-only first" >&2
  exit 1
fi
if [[ "$benchmark" != robotwin ]]; then
  if [[ ! -x "$A2MEAS_RUNTIME/behavior/bin/python" ]]; then
    echo "BEHAVIOR simulator environment missing; install it before downloading resources" >&2
    exit 1
  fi
  if $adapter && [[ ! -x "$A2MEAS_RUNTIME/judge/bin/python" ]]; then
    echo "Judge environment missing; install with --adapter before downloading resources" >&2
    exit 1
  fi
  "$model_python" - <<'PYSDK'
from a2meas.behavior import SDK_ROOT, validate_sdk
validate_sdk(SDK_ROOT)
PYSDK
fi
if { $adapter || { [[ "$benchmark" != behavior ]] && ! $policy_only; }; } &&
   [[ ! -f resources/grounding/sam.pt ]]; then
  if ! "$model_python" -c 'from huggingface_hub import get_token; raise SystemExit(0 if get_token() else 1)'; then
    "$A2MEAS_RUNTIME/model/bin/hf" auth login
  fi
fi
if [[ "$benchmark" != behavior ]]; then
  if [[ ! -x "$A2MEAS_RUNTIME/simulator/bin/python" ]]; then
    echo "RoboTwin simulator environment missing; install it before downloading resources" >&2
    exit 1
  fi
  sam=()
  if ! $policy_only; then sam=(--sam); fi
  "$model_python" serve.py robotwin prepare "${sam[@]}"
  assets=resources/robotwin/simulator/assets
  if [[ -L "$assets" ]] || ! cmp -s configs/robotwin/assets.sha256 "$assets/installation.sha256"; then
    "$A2MEAS_RUNTIME/model/bin/hf" download TianxingChen/RoboTwin2.0 \
      background_texture.zip embodiments.zip objects.zip --repo-type dataset \
      --revision 3dc3b798668feb99ac61cc9086d84cbcc3d79186 \
      --local-dir "$A2MEAS_RUNTIME/cache/robotwin-assets"
    sha256sum --check configs/robotwin/assets.sha256
    (
      mkdir -p -- "$(dirname -- "$assets")"
      assets_staging="$(mktemp -d resources/robotwin/simulator/assets.XXXXXX)"
      trap 'rm -rf -- "$assets_staging"' EXIT
      for archive in background_texture embodiments objects; do
        "$model_python" -m zipfile -e "$A2MEAS_RUNTIME/cache/robotwin-assets/$archive.zip" "$assets_staging"
      done
      find "$assets_staging" -name __MACOSX -prune -exec rm -rf -- {} +
      find "$assets_staging" -type f \( -name .DS_Store -o -name '._*' \) -delete
      find "$assets_staging" -type f -name 'curobo*_tmp.yml' -delete
      cp configs/robotwin/assets.sha256 "$assets_staging/installation.sha256"
      rm -rf -- "$assets"
      mv -- "$assets_staging" "$assets"
    )
  fi
  "$A2MEAS_RUNTIME/simulator/bin/python" -m benchmarks.robotwin.prepare \
    --robotwin-root resources/robotwin/simulator
fi
if [[ "$benchmark" != robotwin ]]; then
  extra=()
  if $adapter; then extra=(--adapter); fi
  "$model_python" serve.py behavior prepare "${extra[@]}"
  if $adapter; then
    "$model_python" - <<'PYJUDGE'
from a2meas import RESOURCE_ROOT
from a2meas.behavior.resources import prepare_judge
prepare_judge(RESOURCE_ROOT / "judge")
PYJUDGE
  fi
  mkdir -p resources/behavior
  sdk="$A2MEAS_RUNTIME/behavior_sdk"
  PYTHONPATH="$sdk/OmniGibson:$sdk/bddl3:$sdk/joylo${PYTHONPATH:+:$PYTHONPATH}" \
    OMNIGIBSON_DATA_PATH="$PWD/resources/behavior" \
    OMNIGIBSON_NO_OMNIVERSE=1 "$A2MEAS_RUNTIME/behavior/bin/python" -c '
from a2meas.behavior import DATA_ROOT
from a2meas.behavior.resources import prepare_dataset
prepare_dataset(DATA_ROOT)
'
fi
