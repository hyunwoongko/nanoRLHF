# StringArray

이 문서에서는 `StringArray`가 무엇인지, 왜 필요한지, 그리고 내부 구현이 어떤 설계 의도를 가지는지 설명합니다. `StringArray`는 문자열을 파이썬 객체 리스트로 들고 있지 않고, **연속된 바이트 버퍼(Buffer)** 와 **offsets(int32) 버퍼**로 저장하며, 결측치(null)는 **Validity Bitmap**으로 분리하여 관리하는 Arrow 스타일의 문자열 배열 타입입니다.

## 1. StringArray는 무엇인가요?

`StringArray`는 아래 구성 요소로 문자열 컬럼을 표현합니다.

- `values: Buffer`  
  모든 문자열을 UTF-8로 인코딩한 바이트들을 하나의 연속된 바이트 버퍼에 이어붙여 저장합니다.

- `offsets: Buffer`  
  각 문자열이 `values` 안에서 어디서 시작하고 어디서 끝나는지 나타내는 int32 오프셋 배열입니다. 길이는 항상 `physical_length + 1`이며, `i`번째 문자열은 `[offsets[i], offsets[i+1])` 범위를 차지합니다.

- `validity: Optional[Bitmap]`  
  결측치 정보를 비트 단위로 저장합니다. 1은 유효, 0은 null을 의미합니다. null이 전혀 없다면 `None`으로 둘 수 있습니다.

- `indices: Optional[Buffer]`  
  `PrimitiveArray`와 마찬가지로, values/offsets를 복사하지 않고도 임의 선택이나 뷰(view) semantics를 만들기 위한 int32 인덱스 매핑 버퍼입니다. indices가 존재하면 이 배열은 non-contiguous 뷰로 동작하며, 논리 인덱스가 실제 문자열 위치(physical index)로 매핑됩니다.

정리하면, `StringArray`는 문자열들을 하나의 큰 UTF-8 바이트 버퍼에 모아두고, offsets로 각 문자열의 경계를 저장하는 Arrow 스타일 구조입니다.

### StringArray 내부 구조에 대한 예시

| i (string idx) | string         | UTF-8 bytes       | byte_len | start = offsets[i] | end = offsets[i+1] | values[start:end] (hex)               |
|---------------:|----------------|-------------------|---------:|-------------------:|-------------------:|---------------------------------------|
|              0 | `Hello.`       | `b"Hello."`       |        6 |                  0 |                  6 | `48 65 6C 6C 6F 2E`                   |
|              1 | `I am Kevin.`  | `b"I am Kevin."`  |       10 |                  6 |                 16 | `49 20 61 6D 20 4B 65 76 69 6E 2E`    |
|              2 | `How are you?` | `b"How are you?"` |       12 |                 16 |                 28 | `48 6F 77 20 61 72 65 20 79 6F 75 3F` |

```python
offsets = [0, 6, 16, 28]
>>> values (utf-8) = b"Hello.I am Kevin.How are you?"
>>> values (hex)   = 48 65 6C 6C 6F 2E 49 20 61 6D 20 4B 65 76 69 6E 2E 48 6F 77 20 61 72 65 20 79 6F 75 3F
```

## 2. 왜 offsets + values 구조가 필요한가요?

파이썬 리스트에 문자열을 저장하면 각 문자열은 개별 파이썬 객체로 존재하고, 리스트에는 그 객체에 대한 포인터들이 저장됩니다. 반면 Arrow 스타일에서는 아래처럼 저장합니다.

- `values`는 모든 문자열을 바이트로 이어붙인 연속 버퍼
- `offsets[i]`, `offsets[i+1]`로 i번째 문자열의 시작/끝을 지정

이 구조의 장점은 다음과 같습니다.

- 문자열 데이터가 연속된 버퍼에 저장되어 캐시 친화적입니다.
- 문자열 경계가 offsets로 명확하게 정의되어, 언어/플랫폼 간 공유가 쉽습니다.
- slice/take 같은 연산을 values 복사 없이(또는 최소 복사로) 구현하기 쉬워집니다.

## 3. offsets의 규칙과 physical_length

`offsets`는 int32 버퍼이므로 바이트 길이는 4의 배수여야 합니다.

```python
if len(offsets) % 4 != 0:
    raise ValueError("offsets buffer size must be a multiple of 4 (int32)")
```

`physical_length`는 offsets 엔트리 개수에서 1을 뺀 값입니다.

```python
physical_length = len(offsets) // 4 - 1
```

offsets가 최소 1개 엔트리는 있어야 하므로, `physical_length < 0`이면 오류입니다.

```python
if physical_length < 0:
    raise ValueError("offsets buffer must contain at least one entry")
```

여기서 `physical_length`는 offsets가 표현할 수 있는 실제 문자열 개수(=base 배열의 길이)입니다.

## 4. 논리 길이와 indices의 의미

`StringArray`는 `indices`가 없으면 contiguous 배열이고, 있으면 non-contiguous 뷰입니다.

### contiguous인 경우

- `logical_length = length`
- 그리고 이 `length`는 반드시 `physical_length`와 같아야 합니다.

```python
if indices is None:
    logical_length = length
    if logical_length != physical_length:
        raise ValueError(f"length mismatch: base_length={physical_length}, length argument={length}")
```

즉 contiguous 배열에서는 offsets가 표현하는 문자열 개수와 배열 길이가 정확히 일치해야 합니다.

### non-contiguous 뷰인 경우

indices가 있으면 논리 길이는 indices 엔트리 개수로 결정됩니다.

```python
if len(indices) % 4 != 0:
    raise ValueError("indices buffer size must be a multiple of 4 (int32)")
logical_length = len(indices) // 4
```

이때 offsets와 values는 base 배열의 것을 그대로 공유하고, indices가 논리 인덱스를 physical index로 매핑합니다.

## 5. 인덱스 관리: normalized index와 base index

`StringArray`도 `PrimitiveArray`와 동일한 개념을 사용합니다.

### normalized index

파이썬은 음수 인덱싱을 허용하므로, 내부에서는 먼저 인덱스를 `[0, length)` 범위로 정규화합니다.

- `-1`은 마지막 원소로 변환됩니다.
- `normalize_index(i, self.length)`를 사용합니다.

### base index

base index는 논리 인덱스를 실제 문자열이 존재하는 위치(physical index)로 매핑한 값입니다.

- contiguous 배열이면 base index는 normalized index와 같습니다.
- non-contiguous 뷰이면 base index는 `indices[normalized]`로부터 읽습니다.

이 base index를 이용해 offsets에서 문자열 경계를 찾습니다.

## 6. `__getitem__`: 문자열을 어떻게 읽나요?

정수 인덱싱은 아래 과정을 따릅니다.

1) null이면 None 반환  
2) base index 계산  
3) offsets에서 start/end 추출  
4) values에서 [start:end) 범위를 slice  
5) UTF-8 디코딩 후 문자열 반환

### null 처리와 base index 범위 검사

```python
if self.is_null(key):
    return None

index = self.base_index(key)
if not (0 <= index < self.physical_length):
    raise IndexError(f"base index {index} out of range [0, {self.physical_length})")
```

### offsets로 문자열 경계 계산

```python
start = unpack_int32(self.offsets, index)
end = unpack_int32(self.offsets, index + 1)
```

offsets가 잘못되어 values 범위를 벗어나거나, end < start 같은 이상 상태면 오류입니다.

```python
if start < 0 or end < start or end > len(self.values):
    raise ValueError(
        f"Invalid string slice range: start={start}, end={end}, values_size={len(self.values)}"
    )
```

### 빈 문자열 처리

문자열 길이가 0이면 바로 `""`를 반환합니다.

```python
length = end - start
if length == 0:
    return ""
```

### values 슬라이스 후 UTF-8 디코딩

```python
sub_buffer = self.values.slice(start, length)
return bytes(sub_buffer.data).decode("utf-8")
```

### 슬라이스 인덱싱

`array[start:stop:step]`은 `take(range(...))`로 위임합니다.

```python
start, stop, step = key.indices(self.length)
return self.take(range(start, stop, step))
```

## 7. `__setitem__`이 없는 이유: immutable 지향

`StringArray`에도 `__setitem__`이 없습니다. Arrow 스타일 배열은 보통 immutable(불변)로 취급되며, 이 구현도 같은 방향을 지향합니다.

immutable을 지향하면 다음과 같은 장점이 있습니다.

- 안전한 공유: 여러 뷰가 같은 `values/offsets/validity` 버퍼를 공유해도, 누군가 제자리 수정을 해서 다른 뷰의 의미가 깨지는 문제가 줄어듭니다.
- zero-copy 최적화 단순화: `take`나 slicing 같은 연산을 뷰로 표현할 때 안전성과 설계가 단순해집니다.
- 병렬 처리에 유리: 여러 연산이 동시에 읽기만 하는 상황에서 데이터 경합을 줄일 수 있습니다.

따라서 변경은 in-place mutation이 아니라, builder를 통해 새 배열을 만들거나 `take`로 새 뷰를 만드는 방식으로 유도됩니다.

## 8. take(indices): 선택 연산의 구체적인 동작

`take`는 논리 인덱스 시퀀스를 받아 선택 결과를 `StringArray`로 반환합니다. 핵심 목표는 `values`를 가능한 한 복사하지 않고, 뷰(view)로 표현하는 것입니다.

### 입력이 비어있는 경우

원소가 하나도 없으면 빈 `offsets([0])`와 빈 values로 빈 배열을 구성합니다.

```python
empty_offsets = pack_int32([0])
empty_values = Buffer.from_bytearray(bytearray())
return StringArray(empty_offsets, 0, empty_values, validity=None, indices=None)
```

### 입력 인덱스 정규화와 연속성 판별

```python
normalized = [normalize_index(i, self.length) for i in indices]
is_contiguous_slice = all(normalized[k] + 1 == normalized[k + 1] for k in range(num_items - 1))
```

연속이면 contiguous-slice 최적화를 시도합니다.

## 9. take에서 연속 선택인 경우: contiguous 원본 vs non-contiguous 원본

### contiguous 원본에서 연속 선택

원본이 contiguous이고 연속 구간을 선택하면, 문자열들이 values 버퍼에서 연속 바이트 범위를 차지하므로 values를 byte slice로 자를 수 있습니다. 다만 중요한 차이는 `PrimitiveArray`와 달리 `StringArray`는 offsets가 있기 때문에 offsets도 새로 만들어야 합니다.

#### 1) base 구간 결정

```python
base_start = start
base_end = start + length
```

#### 2) values에서 잘라낼 바이트 범위 계산

```python
byte_start = unpack_int32(self.offsets, base_start)
byte_end = unpack_int32(self.offsets, base_end)
byte_length = byte_end - byte_start
sub_values = self.values.slice(byte_start, byte_length)
# values[byte_start:byte_start + byte_length]로 슬라이싱, values는 memoryview 기반이므로 실제로 복사가 일어나지는 않음
```

#### 3) sub_offsets 생성: 로컬 기준(0부터)로 재정렬

기존 offsets는 base values 기준의 절대 오프셋이므로, sub_values의 시작점을 0으로 맞추기 위해 `byte_start`를 빼서 재계산합니다.

```python
local_offsets: List[int] = []
for i in range(base_start, base_end + 1):
    offset = unpack_int32(self.offsets, i)
    local_offsets.append(offset - byte_start)

sub_offsets = pack_int32(local_offsets)
```

이렇게 하면 sub_values에 대한 offsets는 항상 0부터 시작하는 올바른 레이아웃이 됩니다.

#### 4) validity 슬라이스

```python
sub_validity = self.validity.slice(start, length) if self.validity else None
```

#### 5) 결과 반환

```python
return StringArray(
    offsets=sub_offsets,
    length=length,
    values=sub_values,
    validity=sub_validity,
    indices=None,
)
```

이 경우 결과는 다시 contiguous `StringArray`가 됩니다.

### non-contiguous 원본에서 연속 선택

원본이 이미 indices 기반 뷰이면, 논리적으로 연속이라도 실제 문자열 위치(physical index)는 연속이 아닐 수 있습니다. 이 경우 offsets/values를 새로 구성하지 않고, indices만 슬라이스하여 더 작은 뷰를 만듭니다.

```python
index_offset = start * 4
index_length = length * 4
sub_indices = self.indices.slice(index_offset, index_length)
sub_validity = self.validity.slice(start, length) if self.validity else None
return StringArray(
    offsets=self.offsets,
    length=length,
    values=self.values,
    validity=sub_validity,
    indices=sub_indices,
)
```

이 경우 결과는 offsets/values를 공유하는 non-contiguous 뷰입니다.

## 10. take에서 비연속 선택인 경우: 새 indices로 뷰 구성

연속 선택이 아니라면 values/offsets를 slice로 표현하기 어렵습니다. 이때는 새 indices 버퍼를 만들어 뷰로 표현합니다.

- 원본이 contiguous이면 base_indices는 normalized 자체입니다.
- 원본이 non-contiguous이면 normalized를 한 번 더 indices로 매핑해 physical index를 얻습니다.

```python
base_indices = normalized if self.is_contiguous() else [unpack_int32(self.indices, i) for i in normalized]
new_indices = pack_int32(base_indices)
return StringArray(
    offsets=self.offsets,
    length=len(base_indices),
    values=self.values,
    validity=self.validity,
    indices=new_indices,
)
```

이 분기에서는 validity도 그대로 공유합니다. 즉 선택 결과는 values/offsets/validity를 공유하고 indices만 새로 만들어 뷰를 구성합니다.

## 11. `to_list`: 파이썬 리스트로 변환

`to_list`는 논리 인덱스를 순회하며 null이면 None, 아니면 `self[i]`로 문자열을 디코딩해서 반환합니다.

```python
output = []
for i in range(self.length):
    if self.is_null(i):
        output.append(None)
    else:
        output.append(self[i])
return output
```

## 12. `from_list`와 `StringArrayBuilder`

`from_list`는 builder를 통해 offsets/values/validity를 한 번에 구성합니다.

```python
builder = StringArrayBuilder()
for value in data:
    builder.append(value)
return builder.finish()
```

### `StringArrayBuilder`의 내부 상태

- `offsets: List[int]`는 항상 0으로 시작합니다.
- `data_bytes: bytearray`에 문자열들을 UTF-8로 인코딩하여 이어붙입니다.
- `validity: List[int]`에 0/1을 누적합니다.

```python
self.offsets: List[int] = [0]
self.data_bytes = bytearray()
self.validity: List[int] = []
```

### `append`: 문자열/None 처리

- None이면 validity=0, offsets는 이전 값 유지(문자열 길이 0 추가)
- 문자열이면 UTF-8로 인코딩 후 data_bytes에 추가, validity=1, offsets에 누적 길이 추가

```python
if value is None:
    self.validity.append(0)
    self.offsets.append(self.offsets[-1])
else:
    encoded = value.encode("utf-8")
    self.data_bytes.extend(encoded)
    self.validity.append(1)
    self.offsets.append(len(self.data_bytes))
```

### finish: buffers와 bitmap 구성 후 StringArray 생성

finish에서는 offsets 길이가 항상 `num_items + 1`인지 확인합니다.

```python
num_items = len(self.validity)
if len(self.offsets) != num_items + 1:
    raise ValueError(
        f"offsets length must be num_items + 1, got offsets={len(self.offsets)}, num_items={num_items}"
    )
```

그 다음 offsets를 int32 버퍼로 pack하고, values 버퍼를 만들고, validity bitmap을 생성합니다.

```python
offsets_buffer = pack_int32(self.offsets)
values_buffer = Buffer.from_bytearray(self.data_bytes)
validity_bitmap = Bitmap.from_list(self.validity)
```

마지막으로 contiguous StringArray를 반환합니다.

```python
return StringArray(
    offsets=offsets_buffer,
    length=num_items,
    values=values_buffer,
    validity=validity_bitmap,
    indices=None,
)
```

## 13. StringArray: 3줄 요약

- `StringArray`는 문자열을 UTF-8 `values` 버퍼에 연속 저장하고, `offsets(int32)`로 각 문자열의 경계를 표현하며 null은 `Bitmap`으로 분리합니다.
- `__getitem__`은 base index로 offsets에서 [start, end)를 구한 뒤 values를 slice하여 UTF-8로 디코딩해 문자열을 반환합니다.
- `take`는 연속 선택이면 values를 byte-slice하고 offsets를 로컬 기준으로 재구성하며, 비연속 선택이면 새 indices로 뷰를 만들어 values/offsets를 복사하지 않습니다.
