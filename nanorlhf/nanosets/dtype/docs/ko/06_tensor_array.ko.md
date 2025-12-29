# TensorArray

이 문서에서는 `TensorArray`가 무엇인지, 왜 필요한지, 그리고 내부 구현이 어떤 설계 의도를 가지는지 설명합니다. `TensorArray`는 각 row를 파이썬 객체(예: `torch.Tensor`)로 직접 들고 있으면서도, row 단위의 결측치(null)는 **Validity Bitmap**으로 분리하여 관리하고, 선택(take)/슬라이스를 **indices 기반 view**로 표현할 수 있도록 만든 Arrow 스타일의 텐서 배열 타입입니다.

## 1. TensorArray는 무엇인가요?

`TensorArray`는 다음 구성 요소로 텐서 컬럼을 표현합니다.

- `tensors: List[Optional[torch.Tensor]]`  
  base storage입니다. 각 row는 `torch.Tensor`이거나 `None`(placeholder)일 수 있습니다.

- `validity: Optional[Bitmap]`  
  base row 단위 결측치 정보를 비트 단위로 저장합니다. 1은 유효, 0은 null을 의미합니다. null이 전혀 없다면 `None`으로 둘 수 있습니다.

- `indices: Optional[Buffer]`  
  (int32) 논리 인덱스 → base 인덱스 매핑 버퍼입니다. `indices`가 존재하면 `TensorArray`는 non-contiguous view로 동작하며, 실제 원소 접근 시 `base_index`를 통해 `tensors`의 위치를 결정합니다.

정리하면, `TensorArray`는 텐서 payload를 파이썬 `torch.Tensor` 객체 리스트로 유지하되, null과 선택(view) 메커니즘을 Arrow 스타일로 분리해 관리하는 구조입니다.

### TensorArray 내부 구조에 대한 예시

예를 들어 아래 데이터가 있다고 하겠습니다.

- `data = [t0, None, t2, t3]` (여기서 `t0, t2, t3`는 같은 dtype/device/shape의 텐서)

이때 contiguous `TensorArray`는 다음처럼 해석됩니다.

- `tensors = [t0, None, t2, t3]`
- `validity = [1, 0, 1, 1]` (bitmap)
- `indices = None`
- `length = 4`

만약 `take([3, 0])`로 선택해 view를 만들면, 결과는 offsets/child를 재구성하는 대신 인덱스 매핑으로 표현됩니다.

- `indices = [3, 0]` (int32 buffer)
- `tensors`와 `validity`는 base와 공유
- `length = 2`

즉, 논리적으로는 `[t3, t0]`를 보지만, 물리적으로는 base storage를 공유하면서 `indices`로만 재배열합니다.

## 2. validity 규칙

`validity`는 **base row 단위** 규칙을 따릅니다.

- `validity`가 존재한다면, `len(validity) == base_length`를 만족해야 합니다.
- `validity[i] == 0`이면 base row i는 null이며, 그 row를 읽으면 `None`을 반환합니다.
- `validity`가 `None`이면 모든 base row가 유효하다고 해석합니다.

중요한 점은, view(`indices`가 존재)인 경우에도 validity는 base 기준으로 저장되어 있으므로, 논리 row 접근 시에는:

1) 논리 인덱스를 정규화  
2) `base_index`로 base 인덱스를 구함  
3) base 인덱스 기준으로 null 여부를 해석

이 흐름을 유지합니다.

## 3. 길이(length), base_length, 그리고 indices의 의미

`TensorArray`는 contiguous base storage와 view를 구분합니다.

### Contiguous (indices 없음)

- `base_length = len(tensors)`
- `logical_length = base_length`
- `indices = None`

이 경우 논리 인덱스 = base 인덱스입니다.

### View (indices 있음)

- `base_length = len(tensors)`
- `logical_length = len(indices) // 4`
- `indices`는 논리 인덱스를 base 인덱스로 매핑합니다.

이 경우 `TensorArray`는 base storage를 공유하면서, `indices`만 바꿔서 선택/재배열을 표현합니다.

### indices 버퍼 규칙

- `indices`는 int32 buffer이므로, byte 길이는 반드시 4의 배수여야 합니다.
- 그렇지 않으면 오류입니다.

## 4. `__getitem__`: 원소를 어떻게 읽나요?

`TensorArray.__getitem__`은 int 인덱싱과 slice 인덱싱을 지원합니다.

### 정수 인덱싱

1) 인덱스를 `normalize_index(key, self.length)`로 정규화  
2) null이면 `None` 반환  
3) `base_index`로 base 인덱스 계산  
4) base 인덱스 범위를 검사  
5) `self.tensors[base_idx]` 반환

```python
if isinstance(key, int):
    normalized_idx = normalize_index(key, self.length)
    if self.is_null(normalized_idx):
        return None
    base_idx = self.base_index(normalized_idx)
    if not (0 <= base_idx < self.base_length):
        raise IndexError(...)
    return self.tensors[base_idx]
```

여기서 핵심은, 실제 텐서 저장소는 `tensors`에 있고, view일 때는 `indices`를 통해 base 위치를 먼저 찾아간다는 점입니다.

### 슬라이스 인덱싱

`array[start:stop:step]`은 `take(range(...))`로 위임합니다.

```python
if isinstance(key, slice):
    start, stop, step = key.indices(self.length)
    return self.take(range(start, stop, step))
```

## 5. `take(indices)`: 선택 연산의 구체적인 동작

`TensorArray.take`는 논리 인덱스 시퀀스를 받아 선택 결과를 `TensorArray`로 반환합니다. 목표는 가능한 한 **복사 없이(view)** 표현하는 것입니다.

### 입력이 비어있는 경우

선택할 원소가 없으면:

- `TensorArray([], None, None)`을 반환합니다.

```python
if num_items == 0:
    return TensorArray([], None, None)
```

### 연속성(contiguous slice) 판별

정규화된 인덱스들이 연속인지 검사합니다.

```python
normalized = [normalize_index(i, self.length) for i in indices]
is_contiguous_slice = all(normalized[k] + 1 == normalized[k + 1] for k in range(num_items - 1))
```

### 연속 선택: contiguous base vs view

연속 선택이면 두 경우로 나뉩니다.

- **base가 contiguous인 경우 (`self.is_contiguous() == True`)**
  - 실제 `tensors` 리스트를 슬라이스하여 `sub_tensors`를 새로 만듭니다.
  - validity가 있으면 동일 구간을 `slice`합니다.
  - 결과는 다시 contiguous `TensorArray`가 됩니다(`indices=None`).

```python
sub_tensors = self.tensors[base_start:base_end]
sub_validity = self.validity.slice(base_start, length) if self.validity is not None else None
return TensorArray(sub_tensors, sub_validity, None)
```

- **base가 이미 view인 경우 (`self.is_contiguous() == False`)**
  - `tensors`/`validity`는 그대로 공유합니다.
  - `indices`만 슬라이스해서 더 작은 view를 만듭니다.

```python
sub_indices = self.indices.slice(index_offset, index_length)
return TensorArray(self.tensors, self.validity, sub_indices)
```

즉, view 위에서의 연속 선택은 "물리적으로 연속"이 아닐 수 있으므로, base payload를 재구성하지 않고 indices만 줄입니다.

### 비연속 선택: 새 indices를 만들어 view 구성

선택이 비연속이면, 결과는 항상 indices 기반 view로 표현합니다.

- base가 contiguous면 `base_indices = normalized`
- base가 view면 `base_indices = [unpack_int32(self.indices, i) for i in normalized]`
- `new_indices = pack_int32(base_indices)`로 새 indices 버퍼 생성
- `TensorArray(self.tensors, self.validity, new_indices)` 반환

```python
new_indices = pack_int32(base_indices)
return TensorArray(self.tensors, self.validity, new_indices)
```

이 방식은 텐서 payload를 복사하지 않고도 임의 선택/재배열을 표현할 수 있게 합니다.

## 6. `to_list`: 파이썬 리스트로 변환

`to_list`는 논리 인덱스를 순회하며:

- null이면 `None`
- 아니면 `self[i]`를 통해 텐서 객체를 append

즉, view일 경우에도 `indices` 매핑을 적용한 결과를 그대로 Python 리스트로 materialize합니다.

## 7. `from_list`와 `TensorArrayBuilder`

`TensorArray.from_list(data)`는 builder를 사용해 텐서 배열을 생성합니다.

- `TensorArrayBuilder()`를 만들고
- 각 원소를 `builder.append(x)`로 누적한 뒤
- `builder.finish()`로 최종 array를 반환합니다.

## 8. TensorArrayBuilder: 빌더의 역할과 prototype 규칙

`TensorArrayBuilder`는 다음 상태를 누적합니다.

- `tensors: List[Optional[torch.Tensor]]`  
- `validity: List[int]` (0/1)
- `prototype: Optional[torch.Tensor]`

### prototype이 필요한 이유

`TensorArray`는 한 컬럼의 텐서가 **일관된 dtype/device/shape**를 가진다는 전제를 강하게 유지합니다. 이를 위해 builder는 첫 번째 non-null 텐서를 `prototype`으로 저장하고, 이후 들어오는 텐서가 prototype과 다음이 일치하는지 검사합니다.

- dtype 일치
- device 일치
- shape 일치

일치하지 않으면 즉시 오류를 발생시켜 "혼합 타입/혼합 shape"을 금지합니다.

### `append(value)` 규칙

- `value is None`이면
  - `validity += [0]`
  - `tensors += [None]`

- `value`가 `torch.Tensor`이면
  - prototype과 dtype/device/shape 일치 검사
  - `validity += [1]`
  - `tensors += [value]`

- 그 외 타입이면 오류를 발생시킵니다.

### `finish()` 규칙

- `len(tensors) == len(validity)`인지 검증
- `validity_bitmap = Bitmap.from_list(validity)`
- `TensorArray(tensors, validity_bitmap, None)` 반환 (contiguous)

즉, builder는 null 처리, 텐서 스키마 일관성(prototype), validity 생성까지를 한 곳에서 책임집니다.

## 9. TensorArray: 3줄 요약

- `TensorArray`는 payload를 `List[Optional[torch.Tensor]]`로 유지하면서 row null은 `Bitmap`으로 분리 관리하고, 선택/재배열은 `indices(int32)` 기반 view로 표현합니다.
- `__getitem__`은 논리 인덱스를 정규화한 뒤 null 체크를 하고, view면 `indices`를 통해 base 인덱스로 매핑하여 `tensors[base_idx]`를 반환합니다.
- `take`는 연속 선택이면 (base가 contiguous일 때) 리스트/bitmap 슬라이스로 새 contiguous를 만들고, 그 외에는 새 `indices`를 만들어 zero-copy view로 반환합니다.