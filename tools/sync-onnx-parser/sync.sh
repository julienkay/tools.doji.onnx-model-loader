#!/usr/bin/env bash
# Syncs the internal ONNX parser copies in Runtime/InferenceEngine/Onnx from
# tools.doji.formats.onnx.
#
# The converter compiles into Unity.InferenceEngine through an asmref, and that
# assembly cannot reference Doji.Onnx. So the loader keeps internal copies of
# Onnx.cs (regenerated with visibility=internal) and ProtoReader.cs. Each copy
# records the SHA-256 of its source; OnnxParserCopyTests fails when they differ.
#
# Usage: tools/sync-onnx-parser/sync.sh [--check] [path/to/tools.doji.formats.onnx]
#   --check  Do not write. Exit 1 when a copy is out of date.
# Needs: protoc and the .NET 8 SDK on the PATH.
set -euo pipefail

check=false
if [[ "${1:-}" == "--check" ]]; then
    check=true
    shift
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
formats_repo="$(cd "${1:-$repo_root/../tools.doji.formats.onnx}" && pwd)"
source_runtime="$formats_repo/tools.doji.formats.onnx/Runtime"
target_dir="$repo_root/tools.doji.onnx-model-loader/Runtime/InferenceEngine/Onnx"
plugin_project="$formats_repo/protoc-gen-unity"

for path in "$source_runtime/Onnx.cs" "$source_runtime/ProtoReader.cs" "$formats_repo/onnx/onnx.proto3" "$plugin_project"; do
    [[ -e "$path" ]] || { echo "Not found: $path" >&2; exit 1; }
done

# Native Windows tools (protoc.exe) do not understand Git Bash paths such as /tmp.
native_path() {
    if command -v cygpath >/dev/null; then cygpath -m "$1"; else printf '%s' "$1"; fi
}

sha256() {
    # Hash without carriage returns, so CRLF and LF checkouts give the same value.
    if command -v sha256sum >/dev/null; then
        tr -d '\r' < "$1" | sha256sum | cut -d' ' -f1
    else
        tr -d '\r' < "$1" | shasum -a 256 | cut -d' ' -f1
    fi
}

header() {
    printf '// Synced from tools.doji.formats.onnx by tools/sync-onnx-parser/sync.sh. Do not edit.\n'
    printf '// Source: tools.doji.formats.onnx/Runtime/%s, %s.\n' "$1" "$2"
    printf '// source-sha256: %s\n' "$3"
}

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# Onnx.cs: regenerate from the schema with internal top-level types.
dotnet build "$plugin_project" -c Release -nologo -v quiet >/dev/null
plugin="$plugin_project/bin/Release/net8.0/protoc-gen-unity"
[[ -e "$plugin.exe" ]] && plugin="$plugin.exe"
mkdir -p "$work/gen"
protoc --plugin=protoc-gen-unity="$(native_path "$plugin")" \
    --unity_out=visibility=internal:"$(native_path "$work/gen")" \
    -I "$(native_path "$formats_repo/onnx")" "$(native_path "$formats_repo/onnx/onnx.proto3")"
{
    header "Onnx.cs" "regenerated with visibility=internal" "$(sha256 "$source_runtime/Onnx.cs")"
    tr -d '\r' < "$work/gen/Onnx.cs"
} > "$work/Onnx.cs"

# ProtoReader.cs: copy with internal top-level classes.
{
    header "ProtoReader.cs" "top-level classes made internal" "$(sha256 "$source_runtime/ProtoReader.cs")"
    tr -d '\r' < "$source_runtime/ProtoReader.cs" | sed -E 's/^    public static class /    internal static class /'
} > "$work/ProtoReader.cs"

for file in Onnx.cs ProtoReader.cs; do
    if grep -qE '^    public ([a-z]+ )*(class|struct|enum|interface|delegate) ' "$work/$file"; then
        echo "$file still has a public top-level type; update this script." >&2
        exit 1
    fi
done

status=0
for file in Onnx.cs ProtoReader.cs; do
    target="$target_dir/$file"
    if [[ -f "$target" ]] && diff -q <(tr -d '\r' < "$target") "$work/$file" >/dev/null; then
        echo "$file: up to date"
    elif $check; then
        echo "$file: out of date" >&2
        status=1
    else
        cp "$work/$file" "$target"
        echo "$file: updated"
    fi
done
exit $status
