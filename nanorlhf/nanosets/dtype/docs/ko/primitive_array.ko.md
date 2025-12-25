# PrimitiveArray

이 문서에서는 `PrimitiveArray`가 무엇인지, 왜 필요한지, 그리고 내부 구현이 어떤 설계 의도를 가지는지 설명합니다. `PrimitiveArray`는 `int32`, `int64`, `float32`, `float64`, `bool` 같은 원시(primitive) 타입 데이터를 연속된 바이너리 버퍼(Buffer)에 저장하고, 결측치(null)는 Validity Bitmap으로 분리하여 관리하는 배열 타입입니다.

## 1. PrimitiveArray는 무엇인가요?

`PrimitiveArray`는 다음 3가지 구성 요소로 데이터를 표현합니다.

- `values: Buffer`  
  실제 값이 들어있는 연속 메모리(바이트열)입니다. 예를 들어 int32라면 원소 하나당 4바이트가 연속적으로 저장됩니다.

- `validity: Optional[Bitmap]`  
  결측치 정보를 비트 단위로 저장합니다. 1은 유효, 0은 null을 의미합니다. 모든 값이 유효하면 `None`으로 둘 수 있어 추가 메모리를 아낄 수 있습니다.

- `indices: Optional[Buffer]`  
  이전부터 쭉 이 시리즈를 봐 오신 분이라면 validity bitmap에서 absolute position index를 정의했던 것을 기억하실 것입니다. 이 indices 배열은 그와 비슷하게, values 버퍼를 0번 위치부터 연속으로 읽지 않고 원하는 부분만 참조할 수 있도록, 즉 뷰(view)처럼 동작할 수 있게 만드는 버퍼입니다. 만약 이 indices 버퍼가 존재하면 이 배열은 비연속(non-contiguous) 뷰로 동작합니다.

정리하면, `PrimitiveArray`는 값과 결측치와 (필요시) 참조 인덱스를 분리하여 저장하는 Arrow 스타일의 배열 구조입니다.

## 2. 왜 values를 Buffer에 바이너리로 저장하나요?

파이썬 리스트에 `int`를 저장하면 내부적으로는 `PyObject*` 포인터가 저장되므로, 값이 메모리에 흩어지고 오버헤드가 커집니다. 반면 `PrimitiveArray`는 원시 타입 값을 정해진 크기(예: int32=4바이트)로 연속된 바이트 버퍼에 저장합니다.

이 구조는 다음과 같은 장점이 있습니다.

- 연속 메모리로 인해 캐시 효율이 좋아집니다.
- 벡터화(SIMD)와 같은 저수준 최적화에 유리합니다.
- 언어/플랫폼 간 호환이 쉬워집니다(바이트 레이아웃이 명확함).

## 3. struct를 왜 쓰나요?

`PrimitiveArray`는 파이썬 값(정수/실수/불리언)을 바이트 버퍼에 저장하고, 다시 읽어오는 과정이 필요합니다. 여기서 `struct`는 이 타입은 몇 바이트이며 어떤 규칙으로 바이트를 해석할지를 지정해서 pack/unpack을 수행합니다.

- `struct.pack_into(fmt, buffer, offset, value)`  
  버퍼의 특정 위치에 값을 바이너리로 기록합니다.

- `struct.unpack_from(fmt, buffer, offset)`  
  버퍼의 특정 위치에서 값을 바이너리로 읽어옵니다.

이때 `FMT[dtype] = (fmt, item_size)` 형태로 dtype별 포맷과 원소 크기를 관리합니다. 예를 들어 int32는 일반적으로 4바이트이며, float64는 8바이트입니다.

## 4. 인덱스 관리

`PrimitiveArray`의 핵심은 논리적으로는 길이가 `length`인 배열처럼 보이지만, 내부적으로는 contiguous 배열일 수도 있고(indices가 없음), indices가 있는 non-contiguous 뷰일 수도 있다는 점입니다. 
이때 인덱스 관련 개념을 먼저 정리해 두면 이후 `__getitem__`과 `take` 등의 메서드를 이해하기가 훨씬 쉽습니다.

### normalized index

파이썬은 `arr[-1]`처럼 음수 인덱스를 허용합니다. 따라서 내부 구현에서는 먼저 인덱스를 항상 `[0, length)` 범위로 정규화합니다. 이를 여기서는 normalized index라고 부릅니다.

예를 들어 길이가 5이면 다음이 성립합니다.

- `-1 -> 4`
- `-2 -> 3`

코드에서는 `normalize_index(idx, self.length)`로 이를 수행합니다.

### base index

base index는 논리 인덱스(사용자 입장에서의 idx)가 실제 values 버퍼에서 어느 원소를 가리키는지 나타내는 인덱스입니다.

- contiguous 배열이면 논리 인덱스와 실제 위치가 동일하므로 `base_index(idx) = normalized_idx` 입니다.
- non-contiguous 뷰이면 indices가 논리 인덱스를 실제 위치로 매핑하므로 `base_index(idx) = indices[normalized_idx]` 입니다.

구현은 다음과 같은 형태입니다.

```python
def base_index(self, idx: int) -> int:
    normalized_idx = normalize_index(idx, self.length)
    if self.is_contiguous():
        return normalized_idx
    return unpack_int32(self.indices, normalized_idx)
```

이 개념 하나로, 이후 모든 값 읽기/쓰기에서 동일한 규칙을 적용할 수 있습니다.

## 5. PrimitiveArray의 길이(length)

`PrimitiveArray.__init__`의 중요한 포인트는 논리적 길이(logical length)를 어떻게 표현하는가 입니다.

- `indices`가 없으면 이 배열은 slice 된 적 없는 배열로, 논리적 길이는 `length`가 됩니다.
- `indices`가 있으면 논리적 길이는 `len(indices) // 4`가 됩니다. indices는 int32 배열이므로 엔트리 하나는 4바이트이고, 따라서 바이트 길이를 4로 나눈 값이 원소 개수입니다.

즉, indices가 존재하면 이 배열은 values를 그대로 들고 있으면서 indices가 가리키는 위치만 읽는 뷰가 되고, 논리적 길이는 indices에 의해 결정됩니다.

## 6. `__getitem__`: 값을 어떻게 읽나요?

`__getitem__`은 정수 인덱스와 슬라이스를 처리합니다.

```python
def __getitem__(self, key: Union[int, slice]):
    if isinstance(key, int):
        if self.is_null(key):
            return None
        offset = self.base_index(key) * self.item_size
        return struct.unpack_from(self.fmt, self.values.data, offset)[0]

    if isinstance(key, slice):
        start, stop, step = key.indices(self.length)
        return self.take(range(start, stop, step))
```

정수 인덱싱의 동작은 다음 순서입니다.

1) 해당 위치가 null인지 확인합니다.
2) null이면 `None`을 반환합니다.
3) null이 아니면 `base_index(key)`로 실제 values 위치(원소 단위)를 구합니다.
4) `offset = base_index(key) * item_size`로 바이트 오프셋을 계산합니다.
5) `struct.unpack_from`으로 bytes를 값으로 해석합니다.

예를 들어 `item_size = 4`(int32)이고 `base_index(key) = 10`이면 `offset = 40`이며, bytes의 40번째 바이트부터 4바이트를 읽어야 합니다.

슬라이스(`array[start:stop:step]`)는 내부적으로 `take(range(...))`로 위임하여 처리합니다. (`take` 메서드에 대해서는 아래에서 후술합니다.)

## 7. `__setitem__`이 없는 이유: immutable 지향

`PrimitiveArray`에는 `__setitem__`이 없습니다. 이는 이 구현이 Apache Arrow의 설계 철학을 따라 배열을 기본적으로 immutable(불변)로 다루는 방향을 지향하기 때문입니다.

불변 구조를 지향하면 다음과 같은 장점이 있습니다.

- 안전한 공유: 여러 배열 뷰가 동일한 `values`/`validity` 버퍼를 공유하더라도, 누군가가 값을 바꿔서 다른 뷰의 의미가 깨지는 문제가 발생하지 않습니다.
- zero-copy 최적화: `slice`나 `take` 같은 연산을 "Zero-copy 뷰"로 표현하는 전략이 훨씬 단순하고 안전해집니다.
- 병렬 처리에 유리: 동일한 데이터를 여러 스레드/연산이 동시에 읽어도 데이터 경합(race)이 줄어듭니다.

따라서 `PrimitiveArray`는 값을 제자리에서 수정하기보다는, `from_list`, `builder.finish`, `take`, `slice` 같은 방식으로 아예 새로운 배열을 생성하거나 zero-copy 뷰를 만들어내는 방식을 권장합니다.

## 8. `take(indices)`: 선택 연산의 구체적인 동작

```python
def take(self, indices: Sequence[int]):
    num_items = len(indices)
    if num_items == 0:
        return PrimitiveArray(self.dtype, 0, values=Buffer.from_bytearray(bytearray(0)))

    normalized = [normalize_index(i, self.length) for i in indices]
    is_contiguous_slice = all(normalized[k] + 1 == normalized[k + 1] for k in range(num_items - 1))

    if is_contiguous_slice:
        start = normalized[0]
        length = num_items
        if self.is_contiguous():
            byte_offset = start * self.item_size
            byte_length = length * self.item_size
            sub_values = self.values.slice(byte_offset, byte_length)
            sub_validity = self.validity.slice(start, length) if self.validity else None
            return PrimitiveArray(self.dtype, length, sub_values, sub_validity)
        else:
            index_offset = start * 4
            index_length = length * 4
            sub_indices = self.indices.slice(index_offset, index_length)
            return PrimitiveArray(self.dtype, length, self.values, self.validity, sub_indices)
    else:
      base_indices = normalized if self.is_contiguous() else [unpack_int32(self.indices, i) for i in normalized]
      new_indices = pack_int32(base_indices)
      return PrimitiveArray(self.dtype, len(base_indices), self.values, self.validity, new_indices)
```

`take`는 원소를 특정 인덱스들로 선택한 결과를 새로운 `PrimitiveArray`로 반환합니다. 여기서 중요한 점은 values 버퍼를 복사하지 않고(zero-copy), 결과를 뷰(view)로 표현하려고 한다는 것입니다.

### 입력: 논리 인덱스의 리스트 또는 시퀀스

`take`의 입력인 `indices`는 사용자 관점에서의 논리 인덱스들입니다. 즉, contiguous든 non-contiguous든 상관없이 우선 입력은 논리 인덱스로 들어온다고 생각하시면 됩니다.

### 입력 인덱스 정규화

먼저 모든 인덱스를 `[0, length)` 범위로 정규화합니다.

```python
normalized = [normalize_index(i, self.length) for i in indices]
```

이 시점부터는 음수 처리 걱정 없이 동일한 로직으로 진행할 수 있습니다.

### 연속 슬라이스인지 판별

정규화된 인덱스들이 완전히 연속 증가인지 확인합니다.

```python
is_contiguous_slice = all(normalized[k] + 1 == normalized[k + 1] for k in range(num_items - 1))
```

여기서 말하는 연속은 논리 인덱스 기준 연속입니다. 예를 들어 `[3, 4, 5, 6]`은 연속이고, `[3, 5, 8]`은 연속이 아닙니다.

### 연속 슬라이스인 경우

연속이면 `start = normalized[0]`, `length = num_items`로 요약됩니다. 이제 핵심 분기는 원본이 contiguous인지 여부입니다.

#### 원본이 contiguous인 경우: values/validity를 직접 슬라이스

contiguous 원본에서는 논리 인덱스가 곧 values에서의 위치이므로, values도 실제로 연속 구간입니다. 
따라서 values를 바이트 단위로 슬라이스하고 validity도 동일 범위로 슬라이스하여 결과를 만들 수 있습니다.

```python
byte_offset = start * self.item_size
byte_length = length * self.item_size
sub_values = self.values.slice(byte_offset, byte_length)
# values 버퍼를 values[byte_offset:byte_offset + byte_length]로 자릅니다.
sub_validity = self.validity.slice(start, length) if self.validity else None
# values와 validity 내부는 memoryview로 데이터를 관리하기에 데이터 복사는 일어나지 않습니다.
return PrimitiveArray(self.dtype, length, sub_values, sub_validity)
```

이 경우 결과는 indices가 필요 없는 contiguous 배열이 됩니다.

#### 원본이 non-contiguous인 경우: indices만 슬라이스하여 더 작은 뷰 만들기

non-contiguous 원본에서는 논리적으로 연속인 구간이더라도 values에서 실제로 연속일 보장이 없습니다. 왜냐하면 values는 공유된 원본 버퍼이고, 실제 위치는 indices가 결정하기 때문입니다.

따라서 이 경우에는 values를 자르지 않고, indices 버퍼에서 해당 구간만 잘라서 결과 뷰를 만듭니다. indices는 int32 엔트리로 이루어져 있으므로 1개당 4바이트입니다.

```python
index_offset = start * 4
index_length = length * 4
sub_indices = self.indices.slice(index_offset, index_length)
return PrimitiveArray(self.dtype, length, self.values, self.validity, sub_indices)
```

이 경우 결과는 여전히 non-contiguous 뷰이며 values/validity를 원본과 공유합니다.

### 연속 슬라이스가 아닌 경우

인덱스가 점프하거나 중복되거나 임의 순서라면 values를 슬라이스로 표현할 수 없습니다. 이때는 결과를 새로운 indices 매핑으로 표현합니다.

여기서 중요한 점은 결과 indices가 논리 인덱스가 아니라 values에서의 실제 위치(base index)를 담아야 한다는 것입니다.

- 원본이 contiguous이면 실제 위치가 논리 인덱스와 동일하므로 base index는 `normalized` 자체입니다.
- 원본이 non-contiguous이면 `normalized`는 원본 논리 인덱스이므로, 이를 한 번 더 indices로 매핑해서 실제 위치를 얻어야 합니다.

```python
base_indices = normalized if self.is_contiguous() else [unpack_int32(self.indices, i) for i in normalized]
```

이제 `base_indices`는 values 버퍼에서 실제로 읽어야 하는 원소 위치 목록입니다. 이를 int32 버퍼로 패킹하여 새 indices를 만들고, values/validity는 공유한 채 결과를 반환합니다.

```python
new_indices = pack_int32(base_indices)
return PrimitiveArray(self.dtype, len(base_indices), self.values, self.validity, new_indices)
```

이 설계의 성질은 다음과 같습니다.

- values 버퍼는 절대 새롭게 복사하지 않습니다.
- take를 여러 번 적용해도 결과는 indices 기반의 뷰로 누적될 수 있습니다.

## 9. `to_list`: 파이썬 리스트로 변환

`to_list`는 모든 원소를 순회하면서 null이면 `None`, 아니면 `struct.unpack_from`으로 값을 읽어 파이썬 값으로 반환합니다.

```python
output = []
for i in range(self.length):
    if self.is_null(i):
        output.append(None)
    else:
        offset = self.base_index(i) * self.item_size
        value = struct.unpack_from(self.fmt, self.values.data, offset)[0]
        output.append(value)
return output
```

이 함수는 외부로 내보내는 용도로는 편리하지만, `struct.unpack_from`에 의해 파이썬 객체를 생성하므로 대규모 데이터에서는 비용이 큽니다.

## 10. `from_list`: 파이썬 리스트로부터 생성하기

`from_list`는 입력 리스트의 dtype을 추론하거나(`infer_primitive_dtype`) 사용자가 지정한 dtype을 사용합니다. 
그리고 후술할 `PrimitiveArrayBuilder`를 사용하여 다음을 수행합니다.

- `values` 목록과 `validity` 목록을 분리하여 누적합니다.
- 마지막에 `struct.pack_into`로 values를 바이트 버퍼에 기록합니다.
- validity는 `Bitmap.from_list`로 변환합니다(전부 유효하면 `None`일 수 있습니다).

dtype별로 입력 값 타입을 엄격히 처리하는 이유는 다음과 같습니다.

- 정수 dtype에 float이 들어오면 의도치 않은 손실/변환이 발생할 수 있으므로 오류로 처리합니다.
- bool은 내부적으로 정수로도 표현 가능하므로 일부 경우 허용되지만, BOOL dtype에서는 bool만 허용합니다.
- int32 범위 검사를 통해 overflow를 방지합니다.

## 11. `infer_primitive_dtype`: dtype 추론
```python
def infer_primitive_dtype(values: List[Optional[PrimitiveType]]) -> DataType:
    saw_float = False
    saw_int = False
    saw_bool = False

    for value in values:
        if value is None:
            continue
        if isinstance(value, bool):
            saw_bool = True
            continue
        if isinstance(value, float):
            saw_float = True
        elif isinstance(value, int):
            saw_int = True
        else:
            raise ValueError(f"Unsupported primitive type: {type(value).__name__}")

    if saw_float:
        return FLOAT64
    if saw_int:
        return INT64
    if saw_bool:
        return BOOL

    raise ValueError("Cannot infer primitive dtype from empty or unsupported values")
```
`infer_primitive_dtype` 함수는 입력 값들의 타입을 검사하여 적절한 `DataType`을 추론합니다.
실제 Arrow는 더 복잡하게 추론하겠지만 여기에서는 심플하게 float > int > bool 우선순위로 dtype을 결정합니다.
만약 [1, 3.14, True]가 들어오면 이 리스트는 float64로 추론됩니다.

## 12. `PrimitiveArrayBuilder`: `pack_into로` 버퍼 구성하기

Builder는 파이썬 값들을 바로 bytes로 쌓지 않고, 우선 `values` 리스트와 `validity` 리스트를 모아둔 다음, 마지막에 한 번에 버퍼를 구성합니다.

핵심은 다음 과정입니다.

1) `raw_buffer = bytearray(num_items * item_size)` 를 미리 확보합니다.
2) `offset`을 `item_size`만큼 증가시키며 `struct.pack_into`로 값들을 기록합니다.
3) (필요한 경우) validity 비트맵을 생성합니다. `Bitmap.from_list`는 `self.validity`가 None이면 그대로 None을 반환합니다.

```python
raw_buffer = bytearray(num_items * self.item_size)

offset = 0
for value in self.values:
    struct.pack_into(self.fmt, raw_buffer, offset, value)
    offset += self.item_size

buffer = Buffer.from_bytearray(raw_buffer)
validity = Bitmap.from_list(self.validity)
return PrimitiveArray(self.dtype, num_items, buffer, validity, indices=None)
```

만약 validity가 존재하는 경우 null 값의 경우에도 values 버퍼에는 자리값이 필요합니다. 
그래서 BOOL이면 `False`, 그 외 숫자면 `0`을 넣고, 실제 null 여부는 validity가 결정합니다.

## 12. PrimitiveArray: 3줄 요약

- `PrimitiveArray`는 원시 타입 값을 `Buffer`에 연속된 바이너리로 저장하고, null은 `Bitmap`으로 분리합니다.
- `base_index`는 논리 인덱스를 values에서의 실제 위치로 변환하며, contiguous면 그대로, non-contiguous면 indices를 통해 매핑합니다.
- Arrow의 철학처럼 immutable을 지향하여 `__setitem__`을 제공하지 않으며, `take`는 연속 선택은 슬라이스로, 임의 선택은 새 indices로 표현해 values를 복사하지 않고 뷰를 구성합니다.
