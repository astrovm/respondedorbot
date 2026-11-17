#!/usr/bin/env bash
# Run checks without host credentials, services, sockets, or user configuration.
set -euo pipefail

if [[ ${1:-} != --inside ]]; then
    command -v podman >/dev/null || { echo 'Missing tool: podman' >&2; exit 1; }
    repo=$(cd "$(dirname "$0")/.." && pwd)
    if [[ -r /proc/meminfo ]]; then
        memory_mb=$(awk '/MemTotal/ {print int($2 / 2048)}' /proc/meminfo)
    elif command -v sysctl >/dev/null; then
        memory_mb=$(($(sysctl -n hw.memsize) / 2097152))
    else
        echo 'Cannot determine memory. Set up a supported Podman host.' >&2
        exit 1
    fi
    image=localhost/respondedorbot-checks
    nice -n 10 podman build --memory "${memory_mb}m" --memory-swap "${memory_mb}m" \
        -t "$image" -f "$repo/scripts/Containerfile.test" "$repo/scripts"
    exec nice -n 10 podman run --rm \
        --memory "${memory_mb}m" --memory-swap "${memory_mb}m" \
        --cpus "${CARGO_BUILD_JOBS:-2}" --volume "$repo:/app" \
        --env CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}" \
        --env CARGO_HOME=/app/target/agent/cargo-home \
        --env CARGO_TARGET_DIR=/app/target/agent/current \
        "$image" bash scripts/check-container.sh --inside "$@"
fi

[[ -f /run/.containerenv || -f /.dockerenv ]] || {
    echo 'The inner checks require a container.' >&2
    exit 1
}
shift
mode=${1:-check}
if (($#)); then shift; fi

pg_bin=$(find /usr/lib/postgresql -name initdb -print -quit)
pg_bin=${pg_bin%/initdb}
pg_data=$(mktemp -d)
chown postgres:postgres "$pg_data"
runuser -u postgres -- "$pg_bin/initdb" -D "$pg_data" \
    --locale-provider=icu --icu-locale=en-US --auth=trust >/tmp/initdb.log
runuser -u postgres -- "$pg_bin/pg_ctl" -D "$pg_data" \
    -l /tmp/postgres.log -o '-h 127.0.0.1' -w start
redis-server --bind 127.0.0.1 --save '' --appendonly no --daemonize yes
cleanup() {
    redis-cli shutdown nosave
    runuser -u postgres -- "$pg_bin/pg_ctl" -D "$pg_data" -m fast -w stop
    rm -rf "$pg_data"
}
trap cleanup EXIT
runuser -u postgres -- "$pg_bin/createdb" respondedorbot_test
export TEST_DATABASE_URL='postgresql://postgres@127.0.0.1:5432/respondedorbot_test?sslmode=disable'
export TEST_REDIS_HOST=127.0.0.1 TEST_REDIS_PORT=6379

case "$mode" in
    test) cargo test --locked "$@" ;;
    lint) cargo fmt --all -- --check
        cargo clippy --locked --workspace --all-targets --all-features -- -D warnings ;;
    check|coverage)
        if [[ $mode == check ]]; then
            cargo fmt --all -- --check
            cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
        fi
        cargo llvm-cov --locked --workspace --all-features --no-report
        status=0
        for package in bot-core bot-adapters botd; do
            cargo llvm-cov report -p "$package" --fail-uncovered-lines 0 --show-missing-lines || status=$?
        done
        exit "$status"
        ;;
    *) echo "Unknown check mode: $mode" >&2; exit 1 ;;
esac
