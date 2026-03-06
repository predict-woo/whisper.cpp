## Sentry Debug Symbols (whisper.node)

whisper.node은 `../whisper.cpp/examples/addon.node/`에서 cmake-js로 빌드되는 네이티브 C++ 애드온이다.
Sentry에서 whisper.node 크래시를 symbolicate하려면 PDB가 바이너리와 동일한 debug ID를 공유해야 한다.

### 빌드 및 업로드 흐름

```bash
# 1. RelWithDebInfo로 빌드 (최적화 + PDB 생성)
cd ../whisper.cpp/examples/addon.node
npx cmake-js compile --config RelWithDebInfo

# 2. 빌드된 바이너리를 npm 패키지에 복사
node ../../npm/scripts/copy-binary.js
# 또는 수동: copy build/RelWithDebInfo/addon.node.node → ../../npm/packages/win32-x64/whisper.node

# 3. PDB를 Sentry에 업로드
cd ../../../alt
npx sentry-cli debug-files upload --org clap-k5 --project alt-electron \
  "../whisper.cpp/examples/addon.node/build/RelWithDebInfo/"
```

### 핵심 규칙

- **반드시 `RelWithDebInfo`** 로 빌드해야 한다. `Release`는 debug ID가 0으로 채워져 PDB 매칭이 불가능하다.
- **바이너리와 PDB는 동일 빌드**에서 나와야 한다. 다시 빌드하면 debug ID가 바뀌므로 PDB도 다시 업로드해야 한다.
- `copy-binary.js`는 `.node` 파일만 복사한다. PDB는 Sentry에 직접 업로드하면 되며 앱에 포함할 필요 없다.
- 릴리스할 때마다 이 과정을 반복해야 한다 (빌드 → 바이너리 복사 → PDB 업로드).

### 검증

```bash
# 출시할 바이너리의 debug ID 확인 (0이 아닌 GUID여야 함)
npx sentry-cli debug-files check "../whisper.cpp/npm/packages/win32-x64/whisper.node"
```