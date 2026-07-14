#!/usr/bin/env bash
# PTM-CoScientist - Dev Deploy (변경된 것만 빌드 & 재시작)
# - git: 마지막 dev-deploy 커밋 대비 git diff
# - 로컬 편집: 마지막 dev-deploy 이후 파일 mtime (uncommitted)
# - 감지 범위: src/, webui/, config/, docker-compose.yml, .env
# Usage: ./scripts/dev-deploy.sh [--all]

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LAST_DEV_BUILD="$REPO_ROOT/.last-dev-build"
LAST_DEV_COMMIT="$REPO_ROOT/.last-dev-build-commit"

FIND_EXCLUDE=(
  -not -path "*/__pycache__/*"
  -not -path "*/.git/*"
  -not -path "*/.venv/*"
  -not -path "*/venv/*"
  -not -name "*.pyc"
  -not -name "*.egg-info"
  -not -path "*/data/*"
)

# 경로 → 컴포넌트 이름
_component_for_path() {
  local f="$1"
  [[ -z "$f" ]] && return 0
  case "$f" in
    src/*|config/*)    echo "api" ;;
    webui/*)           echo "webui" ;;
    docker-compose.yml) echo "compose-file" ;;
    .env)              echo "dotenv" ;;
  esac
}

# git: 마지막 dev-deploy 커밋 이후 + 워킹트리/스테이징 변경
get_changed_components_git() {
  local result=()
  local old_commit="" diff_files=""

  if [[ -f "$LAST_DEV_COMMIT" ]]; then
    old_commit=$(tr -d ' \n\r' < "$LAST_DEV_COMMIT")
  fi

  if [[ -n "$old_commit" ]]; then
    diff_files=$(git diff --name-only "$old_commit" HEAD 2>/dev/null || true)
  fi
  diff_files+=$'\n'$(git diff --name-only HEAD 2>/dev/null || true)
  diff_files+=$'\n'$(git diff --name-only --cached HEAD 2>/dev/null || true)

  while IFS= read -r f; do
    local c
    c=$(_component_for_path "$f")
    [[ -n "$c" ]] && result+=("$c")
  done <<< "$diff_files"

  printf '%s\n' "${result[@]}" | sort -u
}

# mtime: 마지막 dev-deploy 이후 수정된 파일
get_changed_components_mtime() {
  local result=()
  local marker="$LAST_DEV_BUILD"

  # src/ or config/ → api
  for dir in src config; do
    [[ ! -d "$REPO_ROOT/$dir" ]] && continue
    if [[ ! -f "$marker" ]]; then
      result+=("api")
    else
      if find "$REPO_ROOT/$dir" -type f -newer "$marker" "${FIND_EXCLUDE[@]}" 2>/dev/null | grep -q .; then
        result+=("api")
      fi
    fi
  done

  # webui/ → webui
  if [[ -d "$REPO_ROOT/webui" ]]; then
    if [[ ! -f "$marker" ]]; then
      result+=("webui")
    else
      if find "$REPO_ROOT/webui" -type f -newer "$marker" "${FIND_EXCLUDE[@]}" 2>/dev/null | grep -q .; then
        result+=("webui")
      fi
    fi
  fi

  # docker-compose.yml / .env
  if [[ -f "$marker" ]]; then
    for f in docker-compose.yml .env; do
      [[ -f "$REPO_ROOT/$f" ]] || continue
      if [[ "$REPO_ROOT/$f" -nt "$marker" ]]; then
        [[ "$f" == ".env" ]] && result+=("dotenv") || result+=("compose-file")
      fi
    done
  fi

  printf '%s\n' "${result[@]}" | sort -u
}

get_changed_components() {
  local git_changed mtime_changed
  git_changed=$(get_changed_components_git)
  mtime_changed=$(get_changed_components_mtime)

  if [[ -z "$git_changed" && -z "$mtime_changed" ]]; then return; fi
  printf '%s\n' $git_changed $mtime_changed | sort -u
}

# ─── Main ────────────────────────────────────────────────────────────────────

FORCE_ALL=false
for arg in "$@"; do
  [[ "$arg" == "--all" ]] && FORCE_ALL=true
done

echo "=== PTM-CoScientist Dev Deploy ==="

if $FORCE_ALL; then
  CHANGED=("api" "webui")
  echo "Building all (--all)"
else
  CHANGED=($(get_changed_components))
  if [[ ${#CHANGED[@]} -eq 0 ]]; then
    echo "변경 없음 (git/mtime). --all 로 강제 빌드 가능."
    [[ -f "$LAST_DEV_COMMIT" ]] && echo "  Last: $(cat "$LAST_DEV_COMMIT" | cut -c1-12)"
    echo "  HEAD: $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
    exit 0
  fi
  echo "Changed: ${CHANGED[*]}"
  [[ -f "$LAST_DEV_COMMIT" ]] && echo "  Since: $(cat "$LAST_DEV_COMMIT" | tr -d ' \n\r' | cut -c1-12)"
  echo "  HEAD:  $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
fi

# api와 webui는 같은 이미지(ptm-coscientist-api)를 공유하므로
# api가 변경되면 한 번만 빌드하면 webui도 최신화됨
BUILD_SERVICES=()
for c in "${CHANGED[@]}"; do
  case "$c" in
    api)          BUILD_SERVICES+=(coscientist-api) ;;
    webui)        ;;                                  # api 이미지 공유, api 빌드로 충분
    compose-file) BUILD_SERVICES+=(coscientist-api) ;;
    dotenv)       ;;
  esac
done
BUILD_SERVICES=($(printf '%s\n' "${BUILD_SERVICES[@]}" | sort -u))

RESTART_SERVICES=()
for c in "${CHANGED[@]}"; do
  case "$c" in
    api)          RESTART_SERVICES+=(coscientist-api) ;;
    webui)        RESTART_SERVICES+=(coscientist-webui) ;;
    dotenv)       RESTART_SERVICES+=(coscientist-api coscientist-webui) ;;
    compose-file) RESTART_SERVICES+=(coscientist-api coscientist-webui) ;;
  esac
done
RESTART_SERVICES=($(printf '%s\n' "${RESTART_SERVICES[@]}" | sort -u))

# webui 재시작 대상인데 api도 바뀌었다면 → api가 webui보다 먼저 올라와야 하므로
# up -d 는 depend_on을 알아서 처리함 (별도 조작 불필요)

if [[ ${#BUILD_SERVICES[@]} -eq 0 ]]; then
  echo "Build: (skip)"
else
  echo "Building: ${BUILD_SERVICES[*]}"
  docker compose build "${BUILD_SERVICES[@]}"
fi

if [[ ${#RESTART_SERVICES[@]} -eq 0 ]]; then
  echo "Warning: nothing to restart."
else
  echo "Restarting: ${RESTART_SERVICES[*]}"
  docker compose up -d "${RESTART_SERVICES[@]}"
fi

touch "$LAST_DEV_BUILD"
git rev-parse HEAD > "$LAST_DEV_COMMIT" 2>/dev/null || true

HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
echo "Done. (Hash: $HASH)"
