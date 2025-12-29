# IPC
이 문서에서는 nanosets에서 사용하는 IPC(Inter-Process Communication) 파일 포맷을 설명합니다. 
목표는 `Table`을 파일로 직렬화/역직렬화할 때, 가능한 한 Zero-copy에 가깝게 데이터를 다루고, 특히 읽기(read) 경로에서 `mmap` 기반의 메모리 매핑을 통해 불필요한 복사를 줄이는 것입니다.

## 1. IPC는 무엇이고 왜 필요한가요?
IPC는 보통 프로세스 간 통신을 의미하지만, 여기서는 좀 더 넓게 런타임, 프로세스, 언어 사이에서 데이터를 주고받을 수 있는 바이너리 포맷이라는 의미로 사용합니다.

이 코드의 IPC가 해결하려는 문제는 다음과 같습니다.

- `Table`(스키마 + 배치 + 컬럼 배열들)을 파일로 저장
- 다시 로드할 때 Python 객체를 재구성하되, 값 버퍼(`values`/`offsets`/`validity`/`indices` 등)는 가능한 한 복사 없이 재사용
- Arrow 스타일(버퍼 기반, `validity` bitmap, `offsets` 기반) 구조에 맞춰 언어/런타임에 독립적인 레이아웃 유지

## 2. 파일 레이아웃
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

## 3. `write_table`의 전체 흐름
`write_table(fp, table)`은 크게 두 단계를 수행합니다.

### Blob(버퍼) 수집
배열들을 순회하면서 필요한 `Buffer`들을 모두 `blobs: List[memoryview]`에 쌓습니다.  
`add_buffer(b: Buffer) -> int`는 blob 배열에 `b.data`를 추가하고 그 인덱스를 반환합니다.

여기서 중요한 점은 blob이 개별 `Array`의 내부 buffer들이라는 것입니다.

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

## 4. `encode_array`: 배열 메타데이터 직렬화 규칙
`encode_array(array: Array)`는 `array` 타입에 따라 header에 들어갈 메타 정보를 만들고 필요한 버퍼들을 blobs로 등록합니다.

공통적으로 들어가는 메타데이터:

- `dtype`: `{"kind": dtype.name}`
- `length`: `array.length`
- `validity`가 있으면:
  - `validity`: validity buffer blob 인덱스
  - `validity_length`: `len(array.validity)`
- `indices`가 있으면:
  - `indices`: indices buffer blob 인덱스

### PrimitiveArray
- `kind = "primitive"`
- `values = add_buffer(array.values)`

### StringArray
- `kind = "string"`
- `offsets = add_buffer(array.offsets)`
- `values = add_buffer(array.values)`

### ListArray
- `kind = "list"`
- `offsets = add_buffer(array.offsets)`
- `child = encode_array(array.child)` (재귀적으로 child array도 메타로 들어감)

### StructArray
- `kind = "struct"`
- `names = array.field_names`
- `children = [encode_array(ch) for ch in array.children]`

### TensorArray
`TensorArray`는 일반적인 `values` buffer 하나로 끝나지 않아서 별도 함수 `encode_tensor_array`로 처리합니다.

## 5. `TensorArray`의 IPC 표현
`TensorArray`는 내부가 `List[Optional[torch.Tensor]]`이기 때문에, IPC 저장 시 각 row tensor를 개별 blob으로 저장하는 방식은 비효율적입니다.  
대신 여기서는 다음 설계를 사용합니다.

- `base_length`: `TensorArray`의 base storage 길이(= `len(array.tensors)`)
- dtype/shape/device를 메타로 저장
- `base_length`개의 텐서를 하나의 큰 contiguous 블록으로 `stack`
- 그 raw bytes를 `values` blob 하나로 저장

### dtype/shape/device 일관성 체크
`Array`에서 `None`이 아닌 가장 먼저 등장하는 tensor를 prototype로 잡고, `base_tensors` 전체가 동일한 dtype/shape/device인지 검사합니다.

또한 이 구현은 IPC에서 CPU tensor만 지원합니다.

- `prototype.device.type != "cpu"` 이면 에러

### None 처리 방식
`None` row가 있을 수 있으므로 `stack`을 만들기 위해 placeholder tensor가 필요합니다.

- `None`이면 `torch.zeros(elem_shape, dtype=scalar_dtype, device=device)`를 넣음
- non-contiguous tensor는 `.contiguous()`로 맞춤
- 최종적으로 `torch.stack(...).contiguous()`로 큰 1개 블록 생성

그 다음:

- `raw_bytes = stacked_tensor.numpy().tobytes(order="C")`
- 이를 `Buffer`로 감싸서 `values` blob으로 추가

### `TensorArray` 메타에 저장되는 항목
- `kind = "tensor"`
- `base_length`
- `tensor_dtype` (문자열, 예: `float32`)
- `tensor_shape` (리스트)
- `device` (문자열)
- `values` (blob index)

`base_length`가 0이거나 prototype이 없으면 `values`를 `None`으로 둡니다.

## 6. `read_table`: `mmap` 기반 역직렬화
`read_table(path)`는 파일을 열고 `mmap`으로 전체 파일을 메모리에 매핑한 뒤, header를 읽고 buffers를 `memoryview(mm)`에서 슬라이스로 뷰를 만들어 재구성합니다.

핵심은 buffers를 Python `bytearray`로 복사하지 않고, `mmap`에 대한 `memoryview` 슬라이스로 `Buffer`를 만든다는 점입니다.

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

여기서 `Buffer`는 내부적으로 `memoryview`를 들고 있으므로, 전체 read 경로가 `mmap` + `memoryview`로 이어져 zero-copy에 가깝게 동작합니다.

## 7. `decode_array`: 메타데이터로부터 배열 재구성
`decode_array(inputs)`는 `encode_array`의 반대 동작을 합니다.

### dtype 복원
```python
data_type = DataType(inputs["dtype"]["kind"])
```

### validity 복원
`validity`가 있으면:

- `validity_buffer = buffers[inputs["validity"]]`
- `validity_length = inputs.get("validity_length", logical_length)`
- `validity = Bitmap(validity_length, validity_buffer)`

여기서 `Bitmap`도 `Buffer`를 받아 zero-copy로 초기화될 수 있습니다.

### indices 복원
`indices`가 있으면:

- `indices = buffers[inputs["indices"]]`

### kind별 복원 규칙
`ListArray`와 `StringArray`는 `offsets` buffer를 사용하기 때문에 `base_length` 계산 시 `(len(offsets) // 4) - 1`와 같이 계산합니다. (`offsets`는 `int32` 배열이므로 4바이트 단위임)

- primitive:
  - `values_buf = buffers[inputs["values"]]`
  - `item_size = FMT[data_type][1]`
  - `base_length = len(values_buf) // item_size`
  - `PrimitiveArray(data_type, base_length, values_buf, validity, indices)`

- string:
  - `offsets = buffers[inputs["offsets"]]`
  - `values = buffers[inputs["values"]]`
  - `base_length = (len(offsets) // 4) - 1`
  - `StringArray(offsets, base_length, values, validity, indices)`

- list:
  - `offsets = buffers[inputs["offsets"]]`
  - `child = decode_array(inputs["child"])`
  - `base_length = (len(offsets) // 4) - 1`
  - `ListArray(offsets, base_length, child, validity, indices)`

- struct:
  - `names = inputs["names"]`
  - `children = [decode_array(cm) for cm in inputs["children"]]`
  - `StructArray(names, children, validity)`

- tensor:
  - `decode_tensor_array(inputs, validity, indices)`

## 8. `decode_tensor_array`: `TensorArray` 재구성
`TensorArray`는 저장 시 stacked bytes로 저장했으므로, 복원 시에는 다음 과정을 밟습니다.

중요한 포인트는 `torch.frombuffer`는 가능한 경우 buffer를 복사하지 않고 tensor view를 만들 수 있다는 점입니다. 
즉, `mmap`된 파일 데이터 위에 텐서가 올라가는 형태가 됩니다(조건/환경에 따라 내부 동작이 달라질 수는 있지만, 설계 의도는 zero-copy 지향입니다).

- `base_length`, `tensor_dtype`, `tensor_shape`, `values_idx` 읽기
- `values` buffer를 가져오기
- `torch.frombuffer(values_buf.data, dtype=scalar_dtype, count=total_elems)`로 1D 텐서 생성
- `base_length`와 `elem_shape`로 view 해서 `(base_length, *elem_shape)` 블록으로 만들기
- row별 텐서를 리스트로 만들어 `TensorArray` 생성

```python
base_1d = torch.frombuffer(values_buf.data, dtype=scalar_dtype, count=total_elems)
base_block = base_1d.view(base_length, *elem_shape) if elem_shape else base_1d.view(base_length)

base_tensors: List[torch.Tensor] = [base_block[i] for i in range(base_length)]
return TensorArray(base_tensors, validity, indices)
```

## 9. `Schema` / `RecordBatch` / `Table` 복원
header의 `schema.fields`로 `Field`들을 만들고 `Schema`를 구성합니다.
그 다음 `batches`를 순회하며 각 배치의 `columns` 메타를 `decode_array`로 복원하고 `RecordBatch`를 만들고, 최종적으로 `Table(batches)`를 반환합니다.

## 10. IPC: 3줄 요약
- 파일은 `MAGIC` + JSON header + raw buffers(blobs)로 구성되고, header는 buffers의 `offset`/`length`와 배열 메타를 담습니다.
- write는 배열 내부의 `values`/`offsets`/`validity`/`indices` 등을 blob으로 모으고, read는 `mmap` + `memoryview`로 blob을 zero-copy에 가깝게 다시 참조합니다.
- `TensorArray`는 row tensor들을 `stack`한 raw bytes 하나로 저장하고, read에서는 `torch.frombuffer`로 복원해 파일 버퍼 위에 텐서 뷰를 구성합니다.