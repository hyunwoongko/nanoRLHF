# StructArray

이 문서에서는 `StructArray`가 무엇인지, 왜 필요한지, 그리고 내부 구현이 어떤 설계 의도를 가지는지 설명합니다. `StructArray`는 각 row를 파이썬 `dict` 객체로 그대로 들고 있지 않고, **필드별 child Array들을 컬럼 형태로 저장**한 뒤, row 단위의 결측치(null)는 **Validity Bitmap**으로 분리하여 관리하는 Arrow 스타일의 구조체(struct) 배열 타입입니다.

## 1. Columnar(컬럼 지향) 구조를 사용하나요?

`StructArray`를 이해하는 핵심은, 데이터가 **row-oriented(행 지향)** 이 아니라 **columnar(열 지향)** 으로 저장된다는 점입니다.

### Row-oriented vs Columnar

- **Row-oriented**
  - 한 row(예: dict)가 하나의 객체로 묶여 저장됩니다.
  - 예: `[{name: Kevin, age: 30}, {name: Ada, age: 31}, ...]`
  - row 단위로 읽고 쓰는 데는 자연스럽지만, 특정 필드만 반복 처리할 때는 메모리 접근이 산발적일 수 있습니다.

- **Columnar**
  - 필드별로 값을 따로 모아 "열(column)" 단위로 저장합니다.
  - 예: `name = [Kevin, Ada, ...]`, `age = [30, 31, ...]`
  - 동일 필드의 값들이 연속적으로 배치되어, 특정 필드 연산에 유리합니다.

`StructArray`는 columnar 방식을 사용합니다. 즉, row dict를 그대로 저장하는 대신, `children`에 **필드별 컬럼**을 저장합니다.

### Columnar 구조(=children 구조)의 장점

- **필드 단위 처리에 유리**
  - 예를 들어 `age`만 대상으로 필터링/통계/변환을 할 때, `age` 컬럼만 연속 접근하면 됩니다.

- **캐시 친화적**
  - 동일 필드의 값들이 연속 배치되어, CPU 캐시 히트율이 좋아질 가능성이 큽니다.

- **타입별 최적화가 쉬움**
  - `name`은 `StringArray`, `age`는 `PrimitiveArray`, `tags`는 `ListArray`, `meta`는 nested `StructArray`처럼
    각 필드를 타입에 맞는 표현으로 저장할 수 있습니다.

- **선택(take) 같은 연산을 일관되게 적용 가능**
  - row 선택은 사실상 각 컬럼에 같은 인덱스 선택을 적용하는 것이므로,
    `children`에 동일한 `take`를 적용하면 row 정렬이 유지됩니다.

- **언어/런타임 간 공유에 유리**
  - "필드별 배열 + 메타데이터" 형태는 Arrow 계열 포맷과 잘 맞아,
    다양한 시스템에서 동일한 구조로 공유/교환하기 쉽습니다.

## 2. StructArray는 무엇인가요?

`StructArray`는 아래 구성 요소로 struct 컬럼을 표현합니다.

- `field_names: List[str]`  
  struct가 가지는 필드 이름들의 리스트입니다. 예를 들어 `name`, `age` 같은 필드가 될 수 있습니다.

- `children: List[Array]`  
  각 필드에 대응하는 child array들의 리스트입니다. `children[i]`는 `field_names[i]` 필드의 컬럼 데이터를 저장합니다.  
  즉, row 단위로 dict를 저장하는 대신, 필드별 컬럼을 각각 별도의 `Array`로 저장합니다.

- `validity: Optional[Bitmap]`  
  row 단위 결측치 정보를 비트 단위로 저장합니다. 1은 유효, 0은 null을 의미합니다. null이 전혀 없다면 `None`으로 둘 수 있습니다.

정리하면, `StructArray`는 row를 dict로 들고 있지 않고, 필드별 컬럼(child arrays)로 분해해 저장하는 구조입니다.

### StructArray 내부 구조에 대한 예시

예를 들어 아래 rows가 있다고 하겠습니다.

- `rows = [{name: Kevin, age: 30}, None, {name: Ada, age: None}]`

이때 `StructArray`는 다음처럼 표현됩니다.

- `field_names = [name, age]`
- `children`는 필드별 컬럼입니다.
  - `children[name]`는 `[Kevin, None, Ada]` (예: `StringArray`)
  - `children[age]`는 `[30, None, None]` (예: `PrimitiveArray`)
- `validity`는 row 단위로 `[1, 0, 1]` 입니다.

| row i | row value | validity[i] | child[name][i] | child[age][i] |
|---:|---|---:|---|---|
| 0 | `{name: Kevin, age: 30}` | 1 | `Kevin` | `30` |
| 1 | `None` | 0 | `None` | `None` |
| 2 | `{name: Ada, age: None}` | 1 | `Ada` | `None` |

여기서 중요한 점은, row가 null이면 struct 전체가 null이므로 `StructArray.__getitem__`은 `None`을 반환합니다. 동시에 builder 구현상 row가 null이면 모든 child에도 `None`이 append되어, child arrays의 길이 정합성이 유지됩니다.

## 3. 길이(length) 규칙과 children 정합성

`StructArray`의 길이는 children의 길이로 정의됩니다.

- children이 비어 있지 않으면, `length = len(children[0])` 입니다.
- 그리고 모든 child는 동일한 길이를 가져야 합니다.

즉, `children[i]`들의 길이가 하나라도 다르면 오류입니다. 이는 row 정렬이 깨지는 것을 방지하기 위한 강한 불변 조건입니다.

children이 비어 있으면 `length = 0`입니다.

## 4. validity 규칙

`validity`가 존재한다면, `len(validity) == length`를 만족해야 합니다.

- `validity[i] == 0`이면 row i는 struct 전체가 null입니다.
- `validity[i] == 1`이면 row i는 유효하며, `__getitem__`은 children 값을 모아 dict를 구성합니다.

`validity`가 `None`이면 모든 row가 유효하다고 해석합니다.

## 5. 필드 이름 조회: name → index 매핑

`StructArray`는 필드 이름을 빠르게 찾기 위해 내부에 다음 매핑을 저장합니다.

- `name_to_index: Dict[str, int] = {name: i for i, name in enumerate(field_names)}`

이를 통해 `field_index(name)`는 name에 대응하는 child index를 반환합니다.

또한 `check_field_index`는 field index 범위를 검사하는 유틸리티입니다.

## 6. `__getitem__`: row를 어떻게 읽나요?

`StructArray.__getitem__`은 int 인덱싱과 slice 인덱싱을 지원합니다.

### 정수 인덱싱

1) null이면 `None` 반환  
2) 인덱스를 `normalize_index(key, self.length)`로 정규화  
3) `field_names`와 `children`를 순회하며 `child[normalized_idx]` 값을 모아 dict 생성  
4) dict 반환

즉, row를 저장해두지 않고 읽을 때마다 children에서 값을 읽어 row dict를 구성합니다.

```python
if isinstance(key, int):
    if self.is_null(key):
        return None

    normalized_idx = normalize_index(key, self.length)

    row: Dict[str, Any] = {}
    for name, child in zip(self.field_names, self.children):
        row[name] = child[normalized_idx]
    return row
```

### 슬라이스 인덱싱

`array[start:stop:step]`은 `take(range(...))`로 위임합니다.

```python
if isinstance(key, slice):
    start, stop, step = key.indices(self.length)
    return self.take(range(start, stop, step))
```

## 7. `take(indices)`: 선택 연산의 구체적인 동작

`StructArray.take`는 논리 인덱스 시퀀스를 받아 선택 결과를 `StructArray`로 반환합니다. 핵심 목표는 children에 대해 동일한 선택을 적용해, row 정렬을 유지하면서 새로운 struct를 만드는 것입니다.

### 입력이 비어있는 경우

선택할 원소가 없으면:

- 각 child에 대해 `child.take([])`를 호출해 빈 child로 만듭니다.
- `new_validity = None`
- `StructArray(field_names, new_children, new_validity)`를 반환합니다.

```python
if num_items == 0:
    new_children = [child.take([]) for child in self.children]
    new_validity = None
    return StructArray(self.field_names, new_children, new_validity)
```

### 연속성(contiguous slice) 판별

`normalized`를 만든 뒤, 연속 구간인지 검사합니다.

```python
normalized = [normalize_index(i, self.length) for i in indices]
is_contiguous_slice = all(
    normalized[k] + 1 == normalized[k + 1] for k in range(num_items - 1)
)
```

### validity 구성 로직

- `self.validity is None`이면 `new_validity = None`
- validity가 있고, 선택이 연속이면 `validity.slice(start, num_items)`로 빠르게 슬라이스
- validity가 있고, 선택이 비연속이면 각 선택 인덱스에 대해 `is_null`을 호출해 bits를 새로 만들고 `Bitmap.from_list(bits)`로 구성

```python
if self.validity is None:
    new_validity = None
else:
    if is_contiguous_slice:
        start = normalized[0]
        new_validity = self.validity.slice(start, num_items)
    else:
        bits: List[int] = []
        for src_i in normalized:
            bits.append(0 if self.is_null(src_i) else 1)
        new_validity = Bitmap.from_list(bits)
```

### children 구성 로직

children은 항상 `child.take(normalized)`로 동일한 인덱스 선택을 적용합니다.

```python
new_children = [child.take(normalized) for child in self.children]
return StructArray(self.field_names, new_children, new_validity)
```

이 방식은 struct의 핵심 불변 조건인 children 길이 동일성을 자연스럽게 보장합니다.

## 8. `to_list`: 파이썬 리스트로 변환

`to_list`는 전체 row를 순회하며:

- null이면 `None`
- 아니면 각 필드 값을 children에서 읽어 dict를 구성하고 append

즉, `__getitem__`의 row 구성 로직을 명시적으로 펼친 형태입니다.

## 9. `from_list`: 파이썬 rows로부터 StructArray 생성

`StructArray.from_list(rows, strict_keys=False)`는 builder를 사용해 struct를 생성합니다.

- `get_struct_array_builder_from_rows(rows)`로 builder를 만들고
- 각 row를 `builder.append(row)`로 누적한 뒤
- `builder.finish()`로 최종 array를 반환합니다.

`strict_keys`는 row dict에 예상하지 못한 키가 들어왔을 때 에러를 낼지 여부를 제어하는 옵션입니다.

## 10. `StructArrayBuilder`: 빌더의 역할

`StructArrayBuilder`는 struct를 만들기 위해 다음 상태를 누적합니다.

- `field_names`: 필드 목록
- `child_builders`: 필드별 builder 목록
- `strict_keys`: 예상하지 못한 키를 허용할지 여부
- `validity`: row 단위 0/1
- `length`: row 개수

### `append(row)`: row 누적 규칙

- row가 `None`이면
  - `validity += [0]`
  - 모든 child builder에 `None`을 append
  - `length += 1`

- row가 dict이면
  - strict_keys가 켜져 있으면 예상 밖 키를 검사
  - `validity += [1]`
  - 각 필드에 대해 `value = row.get(name, None)`를 읽어 child builder에 append
  - `length += 1`

이 설계로 인해, struct row가 null인 경우에도 children의 길이 정합성이 유지됩니다.

### `finish()`: 최종 StructArray 생성

- `children = [b.finish() for b in child_builders]`
- `validity_bitmap = Bitmap.from_list(validity)` (단, length가 0이면 `None`)
- `StructArray(field_names, children, validity_bitmap)` 반환

## 11. `get_struct_array_builder_from_rows`: struct 스키마를 어떻게 추론하나요?

`get_struct_array_builder_from_rows(rows)`는 rows를 보고 다음을 수행합니다.

1) 모든 row dict를 훑어 등장한 key들을 `inner_names`에 수집합니다.  
   수집 순서는 rows를 순회하며 처음 등장한 순서를 따릅니다.

2) `inner_columns: Dict[str, List[Optional[Any]]]`를 만들어, 각 필드에 대해 열(column) 데이터를 구성합니다.  
   row에 해당 키가 없으면 `None`으로 채웁니다.

3) 각 필드 column에 대해 `inference_builder_for_column(values)`를 호출해 child builder를 만듭니다.

4) `StructArrayBuilder(inner_names, inner_child_builders, strict_keys=False)`를 반환합니다.

즉, struct는 필드 스키마를 row들의 key 집합에서 만들고, 각 필드 타입은 컬럼 단위로 추론합니다.

## 12. `inference_builder_for_column`: 필드 타입을 어떻게 결정하나요?

`inference_builder_for_column(values)`는 필드 하나의 column 값 리스트를 보고 builder를 결정합니다.

- 샘플 값 하나를 찾습니다. (첫 non-null)
- 샘플이 `None`이면 `StringArrayBuilder()`를 반환합니다.  
  즉, 전부 null인 필드는 기본적으로 string builder로 처리됩니다.

- 샘플이 dict이면
  - 전체 값이 dict 또는 None인지 검증하고
  - `get_struct_array_builder_from_rows(values)`로 nested struct builder를 만듭니다.

- 샘플이 list/tuple이면
  - 전체 값이 list/tuple 또는 None인지 검증하고
  - `infer_child_builder(values)`로 list의 child builder를 만든 뒤
  - `ListArrayBuilder(child_builder)`를 반환합니다.

- 샘플이 str이면
  - 전체 값이 str 또는 None인지 검증하고
  - `StringArrayBuilder()`를 반환합니다.

- 샘플이 primitive(bool/int/float)이면
  - 전체 값이 primitive 또는 None인지 검증하고
  - `infer_primitive_dtype(values)`로 dtype을 정한 뒤
  - `PrimitiveArrayBuilder(dtype)`를 반환합니다.

- 샘플이 tensor이면
  - 전체 값이 tensor 또는 None인지 검증하고
  - `TensorArrayBuilder()`를 반환합니다.

- 그 외는 지원하지 않아 오류를 발생 시킵니다.

이 로직은 struct의 각 필드가 단일한 array 표현으로 저장되어야 한다는 전제(필드 단위 타입 일관성)를 강하게 유지합니다.

## 13. StructArray: 3줄 요약

- `StructArray`는 columnar 구조로 `field_names`와 필드별 `children`을 저장하며 row null은 `Bitmap`으로 분리합니다.
- `__getitem__`은 row i에 대해 각 child의 i번째 값을 모아 dict를 구성하고, row가 null이면 `None`을 반환합니다.
- `take`는 children에 동일한 선택을 적용하고, validity는 연속 선택이면 slice, 비연속 선택이면 새 bitmap을 만들어 반환합니다.