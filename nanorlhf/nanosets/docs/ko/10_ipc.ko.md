# IPC (Inter-Process Communication)
이 문서에서는 nanosets에서 사용하는 IPC(Inter-Process Communication) 파일 포맷을 설명합니다. 목표는 `Table`을 파일로 직렬화/역직렬화할 때 가능한 한 Zero-copy에 가깝게 데이터를 다루는 것입니다. 특히 읽기(read) 경로에서 `mmap` 기반 메모리 매핑을 사용해 불필요한 복사를 줄이는 것이 핵심입니다.

## 1. `mmap`을 이해하기 위한 최소한의 배경

### RAM은 "커널 공간"과 "유저 공간"으로 나뉩니다
현대 운영체제에서 RAM은 개념적으로 두 영역으로 나뉘어 있다고 생각하면 됩니다.

- 커널 공간(Kernel space): 운영체제(OS)가 사용하는 메모리 영역입니다. 디스크 캐시, I/O 버퍼, 장치 제어 등 "시스템 전체를 관리하는 코드"가 여기서 동작합니다.
- 유저 공간(User space): 우리가 실행하는 일반 프로그램(Python 포함)이 사용하는 메모리 영역입니다. 각 프로세스는 자기 유저 공간을 갖고, 다른 프로세스의 유저 공간과는 기본적으로 분리됩니다.

이 구분의 목적은 안전성과 격리입니다. 프로그램이 실수로 OS 메모리를 망가뜨리면 시스템 전체가 위험해지기 때문에, OS는 일반 프로그램이 커널 공간을 직접 건드리지 못하게 합니다.

### 파일을 보통 읽으면 왜 "복사"가 두 번 일어날까요?
보통 Python에서 파일을 읽으면(`fp.read()` 같은 방식) 개념적으로 이런 흐름이 됩니다.

```text
[Disk] → Copy → [Kernel space] → Copy → [User space]
```

- 첫 번째 Copy: 디스크에서 읽은 데이터가 커널 공간(커널의 페이지 캐시/버퍼)에 올라옵니다.
- 두 번째 Copy: 그 데이터를 프로그램이 쓰는 유저 공간으로 다시 복사합니다.

이 "두 번의 복사"는 CPU와 메모리 대역폭을 추가로 사용합니다. 데이터가 커질수록 비용이 커집니다.

### `mmap`은 무엇을 줄여주나요?
`mmap`(memory mapping)은 위 흐름에서 "두 번째 Copy(커널 → 유저)"를 줄이는 기법입니다.

`mmap`을 쓰면, 운영체제가 파일 데이터를 유저 공간으로 "복사해서 넘겨주는" 대신, 유저 공간의 주소(가상 주소 공간)에 커널 공간의 파일 페이지를 "매핑"해줍니다.

```text
[Disk] → Copy → [Kernel space] ↔ [User virtual address space]
```

중요한 포인트:
- 파일 데이터 자체는 커널 공간의 페이지 캐시에 있고
- 유저 프로그램은 "그 데이터를 가리키는 주소"를 통해 접근합니다.
- 그래서 유저 공간에 "또 하나의 큰 복사본"을 만들지 않습니다.

### 페이지(page)란 무엇인가요?
운영체제는 메모리를 바이트 단위로 관리하지 않고, 일정한 크기의 "페이지(page)" 단위로 관리합니다.
대부분의 시스템에서 페이지 크기는 4 KB입니다.

즉, 디스크에서 데이터를 RAM으로 가져올 때도 1바이트씩 가져오는 게 아니라 "페이지 단위로" 가져옵니다.
커널 공간에는 이런 페이지들이 저장되는 영역이 있는데 이를 페이지 캐시(page cache)라고 부릅니다.

### `mmap`을 하면 파일 전체가 한 번에 RAM에 올라오나요?
아닙니다. `mmap`은 "즉시 파일 전체를 읽어오는 것"이 아니라, "주소를 매핑해두는 것"입니다.

실제로 프로그램이 매핑된 주소를 "처음으로 읽는 순간"에, 운영체제가 해당 페이지를 디스크에서 읽어 페이지 캐시에 올립니다. 이때 발생하는 이벤트가 페이지 폴트(page fault)입니다.

이를 요구 페이징(demand paging) 또는 지연 로딩(lazy loading)이라고 합니다.

```text
Before 1st access:
    [User virtual address space] → Page Fault → OS → [Disk] → Copy → [Kernel space]

After 1st access:
    [User virtual address space] ↔ [Kernel space]
```

정리하면:
- `mmap()` 호출 시점: "주소 공간만 준비", 실제 파일 데이터는 아직 안 읽었을 수 있음
- 첫 접근 시점: 페이지 폴트 발생, OS가 필요한 페이지만 디스크에서 읽어옴
- 이후 접근: 이미 캐시에 올라온 페이지는 빠르게 접근 가능

| Stage            | User Space (Process Memory)               | Kernel Space (Page Cache)           |
|------------------|-------------------------------------------|-------------------------------------|
| `mmap()` called  | Space for virtual addresses reserved      | No file data loaded (still on disk) |
| After 1st access | Address → Kernel page mapping established | File page loaded into page cache    |
| Later accesses   | Reads from mapped addresses               | File page remains in page cache     |

### `mmap`과 `memoryview`는 어떻게 다른가요?
둘 다 Zero-copy 접근을 가능하게 해주지만, 레벨이 다릅니다.

- `mmap`: OS 레벨에서 동작합니다.
  - 디스크 I/O, 페이지 캐시, 페이지 폴트 같은 것과 연결됩니다.
  - 커널 ↔ 유저 사이 Copy를 줄이는 것이 핵심입니다.

- `memoryview`: Python 레벨에서 동작합니다.
  - 이미 메모리에 있는 바이트/버퍼 객체(예: `bytes`, `bytearray`, `mmap` 객체 등)에 대해
    Zero-copy "뷰"를 제공합니다.
  - 디스크에서 읽어오는 일 자체를 처리하지는 않습니다.

요약:
- `mmap`: OS-level Zero-copy (disk ↔ virtual memory)
- `memoryview`: Python-level Zero-copy (RAM ↔ Python object)

## 2. IPC는 무엇이고 왜 필요한가요?
IPC는 보통 프로세스 간 통신을 의미하지만, 여기서는 좀 더 넓게 런타임/프로세스/언어 사이에서 데이터를 주고받기 위한 바이너리 포맷이라는 의미로 사용합니다.

nanosets에서 IPC가 해결하려는 문제는 다음과 같습니다.

- `Table`(스키마 + 배치 + 컬럼 배열들)을 파일로 저장
- 다시 로드할 때 Python 객체를 재구성하되, 값 버퍼(`values`/`offsets`/`validity`/`indices`)는 가능한 한 Zero-copy로 재사용
- Arrow 스타일(버퍼 기반, `validity` bitmap, `offsets` 기반) 구조에 맞춰 언어/런타임에 독립적인 레이아웃 유지

즉, "사람이 읽기 쉬운 rows-only(JSON)"가 아니라, "버퍼 레이아웃을 유지하면서 빠르게 다시 붙이는" 포맷입니다.

## 3. 파일 레이아웃
`write_table(fp, table)`이 생성하는 파일 레이아웃은 다음과 같습니다.

- `MAGIC` (5 bytes)
- Header Length (4 bytes, little-endian `uint32`)
- Header (JSON bytes)
- Buffers (raw binary blobs; `values`/`offsets`/bitmaps/`indices` 등)

### MAGIC의 역할
`MAGIC`은 파일 포맷을 빠르게 식별하기 위한 고정 바이트 시퀀스입니다.

- 이 구현에서는 `MAGIC = b"NANO0"`
- 의미: nanosets IPC 파일이며 버전 0이라는 표식

### Header(JSON)의 역할
Header는 다음 정보를 담는 메타데이터입니다.

- `Schema`: 필드 이름, dtype, nullable
- `RecordBatch`es: 배치 길이, 각 컬럼(`array`)의 타입(`kind`)과 구성 정보
- `buffers`: blob들의 (`offset`, `length`) 목록

JSON을 쓰는 이유는 사람이 읽기 쉬움, 구현 단순성, 언어 간 파싱 용이성입니다. 데이터 본체는 raw blob으로 두고 메타만 JSON으로 둬서 오버헤드를 최소화합니다.

## 4. `write_table`의 전체 흐름
`write_table(fp, table)`은 크게 두 단계를 수행합니다.

### Blob(버퍼) 수집
배열들을 순회하면서 필요한 `Buffer`들을 모두 `blobs: List[memoryview]`에 쌓습니다.  
`add_buffer(b: Buffer) -> int`는 blob 배열에 `b.data`를 추가하고 그 인덱스를 반환합니다.

여기서 blob은 각 `Array` 내부의 물리 버퍼들입니다.

- `PrimitiveArray`: `values`
- `StringArray`: `offsets` + `values`
- `ListArray`: `offsets` + `child`(재귀)
- `StructArray`: `children`(재귀)
- `TensorArray`: 특수 처리(아래 참조)
- validity bitmap: `validity.buffer`
- indices buffer(뷰): `indices`

### Header 생성 후 파일에 기록
buffers의 누적 길이를 기반으로 offsets를 계산해 `header["buffers"]`에 기록합니다.

그리고 파일에 다음을 순서대로 씁니다.

```python
fp.write(MAGIC)
fp.write(struct.pack("<I", len(header_bytes)))
fp.write(header_bytes)
for blob in blobs:
    fp.write(blob)
```

## 5. `read_table`: `mmap` 기반 역직렬화가 왜 중요한가요?
`read_table(path)`는 파일을 열고 `mmap`으로 전체 파일을 메모리에 매핑한 뒤, header를 읽고 buffers를 `memoryview(mm)`에서 슬라이스로 뷰를 만들어 재구성합니다.

이때 핵심은:
- buffers를 Python `bytearray`/`bytes`로 "다시 만들지 않는다"
- 파일을 매핑한 `mm` 위에서 `memoryview` 슬라이스로 "뷰만 만든다"
- 결과적으로 값 버퍼(`values`/`offsets`/`validity`/`indices`)가 파일 위에 그대로 얹힌다(Zero-copy 지향)

### 파일 파싱 순서
1) `MAGIC` 검사  
2) header length 읽기 (4 bytes, `"<I"`)  
3) header JSON bytes 읽고 `json.loads`  
4) `header["buffers"]`에 따라 raw blob 영역을 `memoryview`로 잡고, 각 blob을 `Buffer.from_memoryview`로 슬라이스

```python
data_start = mm.tell()
base_view = memoryview(mm)[data_start : data_start + total]

buffers: List[Buffer] = []
for buffer in header["buffers"]:
    start = buffer["offset"]
    end = start + buffer["length"]
    buffers.append(Buffer.from_memoryview(base_view[start:end]))
```

여기서 `Buffer`는 내부적으로 `memoryview`를 들고 있으므로, 전체 read 경로가 `mmap` + `memoryview`로 이어져 Zero-copy에 가깝게 동작합니다.

## 6. `decode_array`: 메타데이터로부터 배열 재구성
`decode_array(inputs)`는 헤더 메타데이터를 보고 어떤 `Array`를 만들지 결정하고, 필요한 버퍼를 `buffers[idx]`에서 꺼내 붙입니다.

중요한 관점은 이것입니다.
- `Array` 객체는 새로 만들지만
- 그 `Array`가 들고 있는 값 버퍼는 "파일에서 복사한 새 메모리"가 아니라
- `mmap` 위의 뷰(`Buffer(memoryview(mm)[...])`)를 그대로 사용한다(Zero-copy 지향)

### validity 복원도 Zero-copy 지향으로 이뤄짐
validity가 있으면 다음처럼 `Bitmap`이 `Buffer`를 받아 복원됩니다.

```python
validity_buffer = buffers[inputs["validity"]]
validity = Bitmap(validity_length, validity_buffer)
```

즉, validity bitmap도 파일 매핑 메모리를 그대로 참조합니다.

## 7. `TensorArray`와 `torch.frombuffer`: `mmap`과 결합되는 핵심 포인트
`TensorArray`는 IPC에서 "row마다 텐서를 따로 저장"하지 않고, 텐서들을 하나로 `stack`한 raw bytes를 `values` blob 하나로 저장합니다.

읽을 때 중요한 함수가 `torch.frombuffer`입니다.

### `torch.frombuffer`는 무엇을 하나요?
`torch.frombuffer(buffer, dtype=..., count=...)`는 "바이트 버퍼를 기반으로 텐서를 만든다"는 API입니다.
가능한 경우, 이 과정에서 바이트를 새로 복사하지 않고 **기존 버퍼 위에 텐서 뷰(view)** 를 만듭니다(Zero-copy 지향).

이 문서에서의 의미는 다음과 같습니다.

- `values_buffer.data`는 `mmap` 파일의 `memoryview`일 수 있음
- `torch.frombuffer(values_buffer.data, ...)`를 하면
- "파일 매핑된 메모리 위에" 텐서가 올라가는 형태(뷰)가 될 수 있음
- 즉 큰 텐서 데이터를 다시 복사해서 새로 만들지 않는 설계를 지향함

### 코드에서 실제로 하는 일
```python
base_1d = torch.frombuffer(values_buffer.data, dtype=scalar_dtype, count=total_elems)
base_block = base_1d.view(base_length, *elem_shape) if elem_shape else base_1d.view(base_length)

base_tensors: List[torch.Tensor] = [base_block[i] for i in range(base_length)]
return TensorArray(base_tensors, validity, indices)
```

- `torch.frombuffer(...)`로 1D 텐서를 만들고
- `.view(...)`로 `(base_length, *shape)`로 reshape해서
- row 단위로 `base_block[i]`를 꺼내 리스트를 만든 뒤 `TensorArray`로 감쌉니다.

## 8. IPC: 3줄 요약
- `mmap`은 파일을 유저 공간으로 복사해 들고 오는 대신, 커널 페이지 캐시의 파일 데이터를 유저 가상 주소 공간에 매핑해 "두 번째 Copy(커널→유저)"를 줄입니다(Zero-copy 지향).
- `read_table`은 `mmap` + `memoryview` 슬라이스로 `Buffer`들을 만들어, `values`/`offsets`/`validity`/`indices`를 파일 위에서 Zero-copy로 재참조하는 것을 목표로 합니다.
- `TensorArray`는 raw bytes 하나로 저장되고, 읽을 때 `torch.frombuffer`를 사용해 가능한 경우 파일 매핑 버퍼 위에 텐서 뷰를 구성합니다(Zero-copy 지향).