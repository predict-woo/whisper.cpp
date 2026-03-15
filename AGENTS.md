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

## Core ML W8A8 Encoder (Whisper large-v3-turbo)

기본 경로는 `iOS18` 타깃의 W8A8 ANE encoder export다. 기본 export 산출물은:

- `models/coreml-encoder-large-v3-turbo-w8a8-ane-ios18-bwss.mlpackage`
- `models/coreml-encoder-large-v3-turbo-w8a8-ane-ios18-bwss.mlmodelc`

### Export / Compile

```bash
cd ../whisper.cpp
source .venv-w8a8-ssp/bin/activate

# RedHat compressed-tensors checkpoint에서 Core ML package 생성
python models/export-whisper-coreml-w8a8.py

# mlpackage -> mlmodelc compile
xcrun coremlc compile \
  models/coreml-encoder-large-v3-turbo-w8a8-ane-ios18-bwss.mlpackage \
  models/
```

필요하면 다른 target을 명시할 수 있다:

```bash
python models/export-whisper-coreml-w8a8.py --target iOS17 --output models/coreml-encoder-large-v3-turbo-w8a8-ane.mlpackage
```

### Run / Verify

`whisper-cli`는 기본 encoder bundle 대신 `WHISPER_COREML_ENCODER_PATH` override를 사용할 수 있다.

```bash
WHISPER_COREML_ENCODER_PATH=models/coreml-encoder-large-v3-turbo-w8a8-ane-ios18-bwss.mlmodelc \
./build/bin/whisper-cli --coreml -m models/ggml-large-v3-turbo-q5_0.bin -f samples/jfk.wav
```

확인 포인트:

- 로그에 `loading Core ML model from 'models/coreml-encoder-large-v3-turbo-w8a8-ane-ios18-bwss.mlmodelc'` 가 보여야 한다.
- 로그에 `Core ML model loaded` 가 보여야 한다.
- `--coreml` 없이는 Core ML encoder를 사용하지 않는다.
- `system_info: COREML = 1` 는 지원이 빌드되었다는 뜻이지, 실제 사용 중이라는 뜻은 아니다.
