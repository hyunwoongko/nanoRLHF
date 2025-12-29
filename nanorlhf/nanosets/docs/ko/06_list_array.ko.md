# ListArray

이 문서에서는 `ListArray`가 무엇인지, 왜 필요한지, 그리고 내부 구현이 어떤 설계 의도를 가지는지 설명합니다. `ListArray`는 리스트(가변 길이 시퀀스)를 파이썬 객체 중첩 리스트로 그대로 들고 있지 않고, **child(Array)** 를 하나의 연속 배열로 저장한 뒤, **offsets(int32) 버퍼**로 각 리스트의 경계를 표현하며, 결측치(null)는 **Validity Bitmap**으로 분리하여 관리하는 Arrow 스타일의 리스트 배열 타입입니다.

## 1. ListArray는 무엇인가요?

`ListArray`는 아래 구성 요소로 리스트 컬럼을 표현합니다.

- `child: Array`  
  모든 원소(리스트 내부의 원소들)를 하나의 연속된 1차원 배열로 저장합니다. 예를 들어 `[[1,2], [], [3]]`라면 child는 `[1,2,3]` 같은 형태로 이어붙여집니다.

- `offsets: Buffer`  
  각 리스트가 `child` 안에서 어디서 시작하고 어디서 끝나는지 나타내는 int32 오프셋 배열입니다. 길이는 항상 `base_length + 1`이며, `i`번째 리스트는 `child[offsets[i] : offsets[i+1]]` 범위를 차지합니다.

- `validity: Optional[Bitmap]`  
  결측치 정보를 비트 단위로 저장합니다. 1은 유효, 0은 null을 의미합니다. null이 전혀 없다면 `None`으로 둘 수 있습니다.

- `indices: Optional[Buffer]`  
  `PrimitiveArray`/`StringArray`와 마찬가지로, offsets/child를 복사하지 않고도 임의 선택이나 뷰(view) semantics를 만들기 위한 int32 인덱스 매핑 버퍼입니다. indices가 존재하면 이 배열은 non-contiguous 뷰로 동작하며, 논리 인덱스가 실제 리스트 위치(base index)로 매핑됩니다.

정리하면, `ListArray`는 리스트들의 모든 원소를 child에 모아두고, offsets로 각 리스트의 경계를 저장하는 Arrow 스타일 구조입니다.

### ListArray 내부 구조에 대한 예시

예를 들어 아래 데이터가 있다고 하겠습니다.

- `data = [[1, 2, 3], [], [4, 5], None, [6]]`

이때 `ListArray`는 다음처럼 표현됩니다.

- `child`는 null이 아닌 리스트들의 원소들을 순서대로 이어붙인 배열입니다.
- `offsets`는 각 리스트의 시작/끝(child의 인덱스 범위)을 표현합니다.
- `validity`는 각 row가 null인지 아닌지를 표현합니다.

| i (list idx) | value | validity[i] | start = offsets[i] | end = offsets[i+1] | child[start:end] |
|---:|---|---:|---:|---:|---|
| 0 | `[1, 2, 3]` | 1 | 0 | 3 | `[1, 2, 3]` |
| 1 | `[]` | 1 | 3 | 3 | `[]` |
| 2 | `[4, 5]` | 1 | 3 | 5 | `[4, 5]` |
| 3 | `None` | 0 | 5 | 5 | `[]` (원소를 늘리지 않음) |
| 4 | `[6]` | 1 | 5 | 6 | `[6]` |

```text
offsets = [0, 3, 3, 5, 5, 6]
validity = [1, 1, 1, 0, 1]
child = [1, 2, 3, 4, 5, 6]
```
위에서 i=3은 null이므로 `validity[3]=0`이고 offsets는 이전 값(5)을 그대로 유지합니다. 즉 `[offsets[3], offsets[4])`는 빈 구간이지만, 실제로는 validity가 null이기 때문에 `__getitem__`은 None을 반환합니다.

## 2. 왜 offsets + child 구조가 필요한가요?

파이썬에서 중첩 리스트를 그대로 저장하면, 외부 리스트는 내부 리스트 객체에 대한 포인터들을 들고 있고, 내부 리스트 또한 원소 객체에 대한 포인터들을 들고 있습니다. 반면 Arrow 스타일에서는 아래처럼 저장합니다.

- `child`는 모든 원소를 연속 배열로 이어붙인 저장소
- `offsets[i]`, `offsets[i+1]`로 i번째 리스트의 시작/끝을 지정

이 구조의 장점은 다음과 같습니다.

- 리스트 내부 원소들이 연속된 저장소(child)에 들어가므로, 메모리 접근이 더 예측 가능해집니다.
- offsets로 리스트 경계가 명확하게 정의되어, 언어/플랫폼 간 공유가 쉬워집니다.
- slice/take 같은 연산을 child/offsets 복사 없이(또는 최소 복사로) 구현하기 쉬워집니다.

## 3. offsets의 규칙과 base_length

`offsets`는 int32 버퍼이므로 바이트 길이는 4의 배수여야 합니다.

```python
if len(offsets) % 4 != 0:
    raise ValueError("offsets buffer size must be a multiple of 4 (int32)")
```

`base_length`는 offsets 엔트리 개수에서 1을 뺀 값입니다.

```python
base_length = len(offsets) // 4 - 1
```

offsets가 최소 1개 엔트리는 있어야 하므로, `base_length < 0`이면 오류입니다.

```python
if base_length < 0:
    raise ValueError("offsets buffer must contain at least one entry")
```

여기서 `base_length`는 offsets가 표현할 수 있는 실제 리스트 개수(=base 배열의 길이)입니다.

또한 `ListArray`는 offsets의 마지막 값이 child 길이를 넘어가지 않도록 검증합니다.

```python
total_elems = unpack_int32(offsets, base_length)
if total_elems > len(child):
    raise ValueError(f"offsets refer to {total_elems} child elements, but child length is {len(child)}")
```
즉 offsets는 child의 범위 안에서만 유효한 리스트 경계를 형성해야 합니다.

## 4. 논리 길이와 indices의 의미

`ListArray`는 `indices`가 없으면 contiguous 배열이고, 있으면 non-contiguous 뷰입니다.

### contiguous인 경우

- `logical_length = length`
- 그리고 이 `length`는 반드시 `base_length`와 같아야 합니다.

```python
if indices is None:
    logical_length = length
    if logical_length != base_length:
        raise ValueError(f"length mismatch: base_length={base_length}, length argument={length}")
```
즉 contiguous 배열에서는 offsets가 표현하는 리스트 개수와 배열 길이가 정확히 일치해야 합니다.

### non-contiguous 뷰인 경우

indices가 있으면 논리 길이는 indices 엔트리 개수로 결정됩니다.

```python
if len(indices) % 4 != 0:
    raise ValueError("indices buffer size must be a multiple of 4 (int32)")
logical_length = len(indices) // 4
```
이때 offsets와 child는 base 배열의 것을 그대로 공유하고, indices가 논리 인덱스를 base index로 매핑합니다.

## 5. 인덱스 관리: normalized index와 base index

`ListArray`도 `PrimitiveArray`/`StringArray`와 동일한 개념을 사용합니다.

### normalized index

파이썬은 음수 인덱싱을 허용하므로, 내부에서는 먼저 인덱스를 `[0, length)` 범위로 정규화합니다.

- `-1`은 마지막 원소로 변환됩니다.
- `normalize_index(i, self.length)`를 사용합니다.

### base index

base index는 논리 인덱스를 실제 리스트가 존재하는 위치(base_length 좌표계)로 매핑한 값입니다.

- contiguous 배열이면 base index는 normalized index와 같습니다.
- non-contiguous 뷰이면 base index는 `indices[normalized]`로부터 읽습니다.

이 base index를 이용해 offsets에서 리스트 경계를 찾습니다.

## 6. `__getitem__`: 리스트를 어떻게 읽나요?

정수 인덱싱은 아래 과정을 따릅니다.

1) null이면 None 반환  
2) base index 계산  
3) offsets에서 start/end 추출  
4) child에서 [start:end) 범위를 take  
5) sub_array를 Python list로 변환하여 반환

### null 처리와 base index 범위 검사

```python
if self.is_null(key):
    return None

idx = self.base_index(key)
if not (0 <= idx < self.base_length):
    raise IndexError(f"base index {idx} out of range [0, {self.base_length})")
```

### offsets로 child 범위 계산

```python
start = unpack_int32(self.offsets, idx)
end = unpack_int32(self.offsets, idx + 1)
```

offsets가 잘못되어 child 범위를 벗어나거나, end < start 같은 이상 상태면 오류입니다.

```python
if start < 0 or end < start or end > len(self.child):
    raise ValueError(f"Invalid child range: start={start}, end={end}, child_length={len(self.child)}")
```

### 빈 리스트 처리

리스트 길이가 0이면 바로 `[]`를 반환합니다.

```python
if start == end:
    return []
```

### child에서 슬라이스 후 list 변환

```python
sub_array = self.child.take(range(start, end))
return sub_array.to_list()
```

### 슬라이스 인덱싱

`array[start:stop:step]`은 `take(range(...))`로 위임합니다.

```python
start, stop, step = key.indices(self.length)
return self.take(range(start, stop, step))
```

## 7. `__setitem__`이 없는 이유: immutable 지향

`ListArray`에도 `__setitem__`이 없습니다. Arrow 스타일 배열은 보통 immutable(불변)로 취급되며, 이 구현도 같은 방향을 지향합니다.

immutable을 지향하면 다음과 같은 장점이 있습니다.

- 안전한 공유: 여러 뷰가 같은 offsets/child/validity 버퍼를 공유해도, 누군가 제자리 수정을 해서 다른 뷰의 의미가 깨지는 문제가 줄어듭니다.
- zero-copy 최적화 단순화: `take`나 slicing 같은 연산을 뷰로 표현할 때 안전성과 설계가 단순해집니다.
- 병렬 처리에 유리: 여러 연산이 동시에 읽기만 하는 상황에서 데이터 경합을 줄일 수 있습니다.

따라서 변경은 in-place mutation이 아니라, builder를 통해 새 배열을 만들거나 `take`로 새 뷰를 만드는 방식으로 유도됩니다.

## 8. take(indices): 선택 연산의 구체적인 동작

`take`는 논리 인덱스 시퀀스를 받아 선택 결과를 `ListArray`로 반환합니다. 핵심 목표는 가능한 한 child/offsets를 복사하지 않고, 뷰(view)로 표현하는 것입니다.

### 입력이 비어있는 경우

원소가 하나도 없으면 빈 `offsets([0])`로 빈 배열을 구성합니다.

```python
empty_offsets = pack_int32([0])
return ListArray(
    offsets=empty_offsets,
    length=0,
    child=self.child,
    validity=None,
    indices=None,
)
```
이 경우 child는 기존 child를 그대로 참조합니다.

### 입력 인덱스 정규화와 연속성 판별

```python
normalized = [normalize_index(i, self.length) for i in indices]
is_contiguous_slice = all(normalized[k] + 1 == normalized[k + 1] for k in range(num_items - 1))
```
연속이면 contiguous-slice 최적화를 시도합니다.

## 9. take에서 연속 선택인 경우: contiguous 원본 vs non-contiguous 원본

### contiguous 원본에서 연속 선택

원본이 contiguous이고 연속 구간을 선택하면, 해당 구간이 base_length 좌표계에서도 연속이므로 child 범위도 하나의 연속 구간으로 잡을 수 있습니다. 이때 `ListArray`는 다음을 수행합니다.

- child 범위를 잘라 `new_child`를 만듭니다. (child.take로 선택)
- offsets는 로컬 기준(0부터)로 재정렬해야 하므로 새 offsets를 생성합니다.
- validity는 선택 구간에 맞게 slice합니다.

#### 1) base 구간 결정

```python
base_start = start
base_end = start + length
```

#### 2) child에서 잘라낼 구간 계산 및 new_child 생성

```python
child_start = unpack_int32(self.offsets, base_start)
child_end = unpack_int32(self.offsets, base_end)
new_child = self.child.take(range(child_start, child_end))
```

#### 3) new_offsets 생성: 로컬 기준(0부터)로 재정렬

기존 offsets는 base child 기준의 절대 오프셋이므로, new_child의 시작점을 0으로 맞추기 위해 `child_start`를 빼서 재계산합니다.

```python
local_offsets: List[int] = []
for i in range(base_start, base_end + 1):
    off = unpack_int32(self.offsets, i)
    local_offsets.append(off - child_start)

new_offsets = pack_int32(local_offsets)
```

#### 4) validity 슬라이스

```python
new_validity = self.validity.slice(start, length) if self.validity else None
```

#### 5) 결과 반환

```python
return ListArray(
    offsets=new_offsets,
    child=new_child,
    length=length,
    validity=new_validity,
    indices=None,
)
```
이 경우 결과는 다시 contiguous `ListArray`가 됩니다.

### non-contiguous 원본에서 연속 선택

원본이 이미 indices 기반 뷰이면, 논리적으로 연속이라도 실제 base index는 연속이 아닐 수 있습니다. 이 경우 offsets/child를 새로 구성하지 않고, indices만 슬라이스하여 더 작은 뷰를 만듭니다.

```python
index_offset = start * 4
index_length = length * 4

sub_indices = self.indices.slice(index_offset, index_length)
new_validity = self.validity.slice(start, length) if self.validity else None
return ListArray(
    offsets=self.offsets,
    length=length,
    child=self.child,
    validity=new_validity,
    indices=sub_indices,
)
```
이 경우 결과는 offsets/child를 공유하는 non-contiguous 뷰입니다.

## 10. take에서 비연속 선택인 경우: 새 indices로 뷰 구성

연속 선택이 아니라면 offsets/child를 slice로 표현하기 어렵습니다. 이때는 새 indices 버퍼를 만들어 뷰로 표현합니다.

- 원본이 contiguous이면 base_indices는 normalized 자체입니다.
- 원본이 non-contiguous이면 normalized를 한 번 더 indices로 매핑해 base index를 얻습니다.

```python
base_indices = normalized if self.is_contiguous() else [unpack_int32(self.indices, i) for i in normalized]
new_indices = pack_int32(base_indices)
return ListArray(
    offsets=self.offsets,
    length=len(base_indices),
    child=self.child,
    validity=self.validity,
    indices=new_indices,
)
```
이 분기에서는 validity도 그대로 공유합니다. 즉 선택 결과는 offsets/child/validity를 공유하고 indices만 새로 만들어 뷰를 구성합니다.

## 11. `to_list`: 파이썬 리스트로 변환

`to_list`는 논리 인덱스를 순회하며 null이면 None, 아니면 `self[i]`로 리스트를 구성해서 반환합니다.

```python
outputs = []
for i in range(self.length):
    if self.is_null(i):
        outputs.append(None)
    else:
        outputs.append(self[i])
return outputs
```

## 12. `from_list`와 `ListArrayBuilder`

`from_list`는 builder를 통해 offsets/child/validity를 한 번에 구성합니다.

```python
child_builder = infer_child_builder(data)
builder = ListArrayBuilder(child_builder)
for row in data:
    builder.append(row)

return builder.finish()
```
여기서 핵심은 child의 dtype/빌더를 입력 데이터로부터 추론한다는 점입니다.

### `ListArrayBuilder`의 내부 상태

- `child_builder: ArrayBuilder`는 child 원소들을 누적하는 빌더입니다.
- `offsets: List[int]`는 항상 0으로 시작합니다.
- `validity: List[int]`에 0/1을 누적합니다.
- `length: int`는 append된 row 개수입니다.

```python
self.child_builder = child_builder
self.offsets: List[int] = [0]
self.validity: List[int] = []
self.length: int = 0
```

### `append`: 리스트/None 처리

- None이면 validity=0, offsets는 이전 값 유지(원소 추가 없음), length += 1
- 리스트(Iterable)이면 validity=1, 각 elem을 child_builder에 append, offsets에 누적 child 원소 수 반영, length += 1

또한 문자열/바이트 같은 것은 iterable이지만 리스트 원소로 취급하면 의도와 다르므로 명시적으로 금지합니다.

```python
if value is None:
    self.validity.append(0)
    self.offsets.append(self.offsets[-1])
    self.length += 1
    return self

if isinstance(value, (str, bytes, bytearray)) or not hasattr(value, "__iter__"):
    raise TypeError(
        f"ListArrayBuilder.append expects an iterable (non-string) or None, got {type(value).__name__}"
    )

self.validity.append(1)
start_count = self.offsets[-1]
count = 0
for elem in value:
    self.child_builder.append(elem)
    count += 1

self.offsets.append(start_count + count)
self.length += 1
return self
```

### finish: buffers와 child array, bitmap 구성 후 ListArray 생성

finish에서는 validity 길이와 offsets 길이를 검증합니다.

```python
num_items = self.length
if len(self.validity) != num_items:
    raise ValueError(f"validity length {len(self.validity)} does not match number of items {num_items}")
if len(self.offsets) != num_items + 1:
    raise ValueError(
        f"offsets length must be num_items + 1, got offsets={len(self.offsets)}, num_items={num_items}"
    )
```

그 다음 offsets를 int32 버퍼로 pack하고, child array를 child_builder.finish()로 만들고, validity bitmap을 생성합니다.

```python
offsets_buffer = pack_int32(self.offsets)
child_array = self.child_builder.finish()
validity_bitmap = Bitmap.from_list(self.validity)
```

마지막으로 contiguous ListArray를 반환합니다.

```python
return ListArray(
    offsets=offsets_buffer,
    length=num_items,
    child=child_array,
    validity=validity_bitmap,
    indices=None,
)
```

## 13. infer_child_builder: child_builder를 어떻게 추론하나요?

`ListArray.from_list`는 입력 데이터의 원소 타입에 맞는 child builder를 자동으로 선택하기 위해 `infer_child_builder(rows)`를 사용합니다. 이 함수는 다음 목표를 갖습니다.

- 입력 rows에서 대표 샘플 원소를 하나 찾아 타입을 결정합니다.
- 리스트 원소가 중첩 리스트라면, 재귀적으로 내부 원소 타입을 추론하여 **중첩 ListArrayBuilder**를 구성합니다.
- dict, str, primitive, tensor 등 여러 타입을 지원하고, 타입이 섞이면 명시적으로 오류를 냅니다.

아래는 실제 코드입니다.

```python
def infer_child_builder(rows: List[Optional[Iterable[Any]]]) -> ArrayBuilder:
    from nanorlhf.nanosets.dtype.primitive_array import PrimitiveArrayBuilder
    from nanorlhf.nanosets.dtype.string_array import StringArrayBuilder
    from nanorlhf.nanosets.dtype.list_array import ListArrayBuilder
    from nanorlhf.nanosets.dtype.tensor_array import TensorArrayBuilder

    sample: Any = None
    for row in rows:
        if row is None:
            continue
        for element in row:
            if element is not None:
                sample = element
                break
        if sample is not None:
            break

    if sample is None:
        raise ValueError("Cannot infer element type: all rows are None or empty.")

    if isinstance(sample, (list, tuple)):
        inner_rows: List[Optional[Iterable[Any]]] = []
        for row in rows:
            if row is None:
                continue
            for sub in row:
                if sub is None:
                    inner_rows.append(None)
                elif isinstance(sub, (list, tuple)):
                    inner_rows.append(sub)
                else:
                    raise TypeError(f"Expected nested list elements, found {type(sub).__name__}")
        inner_child_builder = infer_child_builder(inner_rows)
        return ListArrayBuilder(inner_child_builder)

    if isinstance(sample, dict):
        dict_elements: List[Optional[Dict[str, Any]]] = []
        for row in rows:
            if row is None:
                continue
            for element in row:
                if element is None:
                    dict_elements.append(None)
                elif isinstance(element, dict):
                    dict_elements.append(element)
                else:
                    raise TypeError(f"Mixed element types: expected dict, got {type(element).__name__}")

        return get_struct_array_builder_from_rows(dict_elements)

    if isinstance(sample, str):
        for row in rows:
            if row is None:
                continue
            for element in row:
                if element is None:
                    continue
                if not isinstance(element, str):
                    raise TypeError(f"Mixed element types: expected str, got {type(element).__name__}")
        return StringArrayBuilder()

    if isinstance(sample, (bool, int, float)):
        prims: List[Optional[PrimitiveType]] = []
        for row in rows:
            if row is None:
                continue
            for element in row:
                if element is None:
                    prims.append(None)
                    continue
                if isinstance(element, (bool, int, float)):
                    prims.append(element)
                else:
                    raise TypeError(f"Mixed element types: expected primitive, got {type(element).__name__}")

        data_type = infer_primitive_dtype(prims)
        return PrimitiveArrayBuilder(data_type)

    if torch.is_tensor(sample):
        for row in rows:
            if row is None:
                continue
            for element in row:
                if element is None:
                    continue
                if not torch.is_tensor(element):
                    raise TypeError(f"Mixed element types: expected tensor-like, got {type(element).__name__}")
        return TensorArrayBuilder()

    raise TypeError(f"Unsupported element type for list: {type(sample).__name__}")
```

### infer_child_builder의 동작 요약

- 샘플 선택 단계
  - rows를 앞에서부터 훑으며, row가 None이 아니고 element가 None이 아닌 첫 원소를 `sample`로 선택합니다.
  - 전부 None이거나 비어 있으면 타입을 추론할 수 없으므로 오류를 냅니다.

- 분기 처리(대표적으로 아래 순서)
  - `sample`이 `(list, tuple)`이면 중첩 리스트로 보고, 내부 리스트들로 `inner_rows`를 구성한 뒤 재귀 호출하여 내부 child builder를 만든 다음 `ListArrayBuilder(inner_child_builder)`를 반환합니다.
  - `sample`이 `dict`이면 dict만 모아 `get_struct_array_builder_from_rows(...)`로 struct builder를 구성합니다.
  - `sample`이 `str`이면 전체 원소가 `str` 또는 `None`인지 검증한 뒤 `StringArrayBuilder()`를 반환합니다.
  - `sample`이 `(bool, int, float)`이면 전체 원소가 primitive 또는 `None`인지 검증하고, `infer_primitive_dtype(...)`로 dtype을 정한 뒤 `PrimitiveArrayBuilder(dtype)`를 반환합니다.
  - `sample`이 tensor이면 전체 원소가 tensor 또는 `None`인지 검증한 뒤 `TensorArrayBuilder()`를 반환합니다.
  - 그 외 타입은 지원하지 않으므로 오류를 냅니다.

### 왜 “혼합 타입”을 강하게 금지하나요?

`ListArray`는 내부적으로 child가 하나의 `Array`로 표현되어야 하므로, child의 dtype(또는 구조)이 단일하게 정해져야 합니다. 예를 들어 한 row에서는 문자열, 다른 row에서는 숫자가 섞이면 child를 어떤 dtype으로 만들어야 할지 모호해지고, 이후 연산도 불안정해집니다. 그래서 `infer_child_builder`는 샘플로 타입을 정한 뒤, 전체 rows를 순회하면서 해당 타입 규칙을 만족하지 않으면 즉시 예외를 발생시킵니다.

## 14. ListArray: 3줄 요약

- `ListArray`는 모든 리스트 원소를 child 배열에 연속 저장하고, `offsets(int32)`로 각 리스트의 경계를 표현하며 null은 `Bitmap`으로 분리합니다.
- `__getitem__`은 base index로 offsets에서 [start, end)를 구한 뒤 child를 take하여 해당 구간을 파이썬 리스트로 반환합니다.
- `from_list`는 `infer_child_builder`로 child builder를 추론하고, `ListArrayBuilder`가 offsets/validity를 누적한 뒤 finish에서 최종 `ListArray`를 생성합니다.